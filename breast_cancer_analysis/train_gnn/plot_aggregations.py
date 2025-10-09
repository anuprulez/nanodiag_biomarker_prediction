from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import json
import umap
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

from sklearn.preprocessing import label_binarize

from omegaconf.omegaconf import OmegaConf

# Global styling (adjust as you like; calls remain the same)
sns.set_theme(style="whitegrid", context="talk")


def _as_path(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_radar(config):
    """
    models: dict of {model_name: list/np.array of per-class accuracies}
            Accuracies can be in [0,1] or [0,100]; auto-scaled to [0,1].
    class_labels: list of class names (length = number of classes)
    """
    # Use your real values. Model_A uses the numbers you provided.
    metric_keys = [
        "te_f1_macro",
        "te_f1_weighted",
        "te_f1_micro",
        "te_precision",
        "te_recall",
    ]
    models = ["pna", "gcn", "gsage", "gatv2", "gtran"]
    metric_labels = {
        "te_f1_macro": "F1 (macro)",
        "te_f1_weighted": "F1 (weighted)",
        "te_f1_micro": "F1 (micro)",
        "te_precision": "Precision",
        "te_recall": "Recall",
    }

    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)

    agg_models_perf = dict()
    for model in models:
        filename = f"{config.p_plot}all_metrics_{model}.json"
        with open(filename, "r") as f:
            agg_models_perf[model] = json.load(f)
    print(agg_models_perf)
    models = list(agg_models_perf.keys())
    # ===== 2) Prepare data for radar plot =====
    N = len(models)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close the loop

    def values_for_metric(key):
        vals = [agg_models_perf[m][key] for m in models]
        vals = np.array(vals, dtype=float)
        return np.concatenate([vals, vals[:1]])

    # ==== 5) Plot ====
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)  # start at top
    ax.set_theta_direction(-1)  # clockwise

    # Put model names on axes (corners)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, models, fontsize=11)

    # Radial limits and ticks (scores assumed in 0..1)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
    ax.set_rlabel_position(0)

    # Plot each metric as a polygon; legend shows metrics
    for k in metric_keys:
        vals = values_for_metric(k)
        ax.plot(angles, vals, linewidth=2, label=metric_labels.get(k, k))
        ax.fill(angles, vals, alpha=0.10)

    # Title, legend, layout
    plt.title(
        "Model Comparison (Models on Axes, Metrics in Legend)", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=False)

    plt.tight_layout()
    path = plot_local_path / f"Radar_plot_aggregated_edges_{n_edges}_epochs_{n_epo}.pdf"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_radar_runs(config):
    """
    Aggregate per-run metrics for each model, average the 5 test metrics across runs,
    and plot a radar chart comparing models on averaged metrics.

    Expects files at:
      {config.p_base}/runs/{run_id}/all_metrics_{model}.json
    for run_id in [1..config.n_runs] and model in ["pna","gcn","gsage","gatv2","gtran"].
    """
    # ---- Settings ----
    metric_keys = [
        "te_f1_macro",
        "te_f1_weighted",
        "te_f1_micro",
        "te_precision",
        "te_recall",
    ]
    models = ["GATv2", "GCN", "GraphSage", "GraphTransformer"]

    metric_labels = {
        "te_f1_macro": "F1 (macro)",
        "te_f1_weighted": "F1 (weighted)",
        "te_f1_micro": "F1 (micro)",
        "te_precision": "Precision",
        "te_recall": "Recall",
    }

    plot_local_path = _as_path(config.p_plot)
    p_base = getattr(config, "p_base", "./")
    n_edges = getattr(config, "n_edges", 0)
    n_epo = getattr(config, "n_epo", 0)
    n_runs = getattr(config, "n_runs", 5)
    dpi = getattr(config, "dpi", 200)

    # ---- 1) Load & aggregate across runs per model ----
    agg_models_perf_avg = {}   # model -> {metric_key: mean_over_runs}

    for model in models:
        metrics_list = []
        # Collect all runs for this model
        for item in range(n_runs):
            run_id = item + 1
            file_path = f"{p_base}runs/{run_id}/all_metrics_{model}.json"
            with open(file_path, "r") as f:
                values = json.load(f)
                metrics_list.append({
                    "te_f1_macro": values["te_f1_macro"],
                    "te_f1_weighted": values["te_f1_weighted"],
                    "te_f1_micro": values["te_f1_micro"],
                    "te_precision": values["te_precision"],
                    "te_recall": values["te_recall"],
                })
            print(f"Run: {item + 1}, model {model}: {metrics_list[item]}")
        print(f"Aggregated model {model}")

        # Compute scalar averages for the 5 test metrics across runs
        agg_models_perf_avg[model] = {}
        for k in metric_keys:
            vals = [float(m[k]) for m in metrics_list if k in m]
            print(f"Metric: {k}, model: {model}, vals: {vals}")
            agg_models_perf_avg[model][k] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    print("Averaged for models")
    print(agg_models_perf_avg)
    # Keep model order consistent with what we actually loaded
    models = list(agg_models_perf_avg.keys())

    # ---- 2) Prepare data for radar plot (models = axes) ----
    N = len(models)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close the loop

    def values_for_metric(key):
        vals = [agg_models_perf_avg[m][key] for m in models]
        vals = np.array(vals, dtype=float)
        return np.concatenate([vals, vals[:1]])

    # ---- 3) Plot ----
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)  # start at top
    ax.set_theta_direction(-1)      # clockwise

    # Put model names on axes (corners)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, models, fontsize=11)

    # Radial limits and ticks (scores in 0..1)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=11)
    ax.set_rlabel_position(0)

    # Plot each metric as a polygon; legend shows metrics
    for k in metric_keys:
        vals = values_for_metric(k)
        if np.all(np.isnan(vals)):
            continue
        ax.plot(angles, vals, linewidth=2, label=metric_labels.get(k, k))
        ax.fill(angles, vals, alpha=0.10)

    # Title, legend, layout
    plt.title(
        f"Model Comparison (Averaged over {n_runs} runs) — Models on Axes, Metrics in Legend",
        fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=False)

    plt.tight_layout()
    out_path = plot_local_path / f"Radar_plot_averaged_edges_{n_edges}_epochs_{n_epo}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")


def plot_mean_std_loss_acc(config):
    """
    Plot mean and standard deviation across multiple runs of training.
    metrics_list: list of dicts with keys 'tr_loss', 'te_loss', 'val_acc', 'te_acc'.
    """
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = config.dpi

    n_runs = 5
    chosen_model = "GCN"
    metrics_list = []

    for item in range(n_runs):
        run_id = item + 1
        file_path = f"{config.p_base}runs/{run_id}/all_metrics_{chosen_model}.json"
        with open(file_path) as f:
            metrics_list.append(json.load(f))
    print(metrics_list)

    # Convert lists of arrays to stacked numpy arrays (shape: [n_runs, n_epochs])
    tr_loss = np.array([np.asarray(m["tr_loss"]) for m in metrics_list])
    te_loss = np.array([np.asarray(m["te_loss"]) for m in metrics_list])
    val_loss = np.array([np.asarray(m["val_loss"]) for m in metrics_list])

    tr_acc = np.array([np.asarray(m["tr_acc"]) for m in metrics_list])
    te_acc = np.array([np.asarray(m["te_acc"]) for m in metrics_list])
    val_acc = np.array([np.asarray(m["val_acc"]) for m in metrics_list])

    # Mean and standard deviation across runs
    tr_loss_mean, tr_loss_std = tr_loss.mean(axis=0), tr_loss.std(axis=0)
    te_loss_mean, te_loss_std = te_loss.mean(axis=0), te_loss.std(axis=0)
    val_loss_mean, val_loss_std = val_loss.mean(axis=0), val_loss.std(axis=0)

    tr_acc_mean, tr_acc_std = tr_acc.mean(axis=0), tr_acc.std(axis=0)
    val_acc_mean, val_acc_std = val_acc.mean(axis=0), val_acc.std(axis=0)
    te_acc_mean, te_acc_std = te_acc.mean(axis=0), te_acc.std(axis=0)

    x_val = np.arange(1, n_epo + 1)
    xticks = np.arange(1, n_epo + 1, 2)

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_val, tr_loss_mean, linewidth=2, label="Train loss", color="tab:orange")
    ax.fill_between(x_val, tr_loss_mean - tr_loss_std, tr_loss_mean + tr_loss_std, alpha=0.2, color="tab:orange")

    ax.plot(x_val, te_loss_mean, linewidth=2, label="Test loss", color="tab:green")
    ax.fill_between(x_val, te_loss_mean - te_loss_std, te_loss_mean + te_loss_std, alpha=0.2, color="tab:green")

    ax.plot(x_val, val_loss_mean, linewidth=2, label="Val loss", color="tab:blue")
    ax.fill_between(x_val, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color="tab:blue")

    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(f"Mean ± Std Loss over epochs: {chosen_model}")
    plt.xticks(xticks)
    ax.legend(loc="best")
    ax.grid(True)
    _save(fig, plot_local_path / f"Model_loss_mean_std_{n_edges}_links_{n_epo}_epochs_{chosen_model}.pdf", dpi)

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_val, tr_acc_mean, linewidth=2, label="Train accuracy", color="tab:orange")
    ax.fill_between(x_val, tr_acc_mean - tr_acc_std, tr_acc_mean + tr_acc_std, alpha=0.2, color="tab:orange")

    ax.plot(x_val, te_acc_mean, linewidth=2, label="Test accuracy", color="tab:green")
    ax.fill_between(x_val, te_acc_mean - te_acc_std, te_acc_mean + te_acc_std, alpha=0.2, color="tab:green")

    ax.plot(x_val, val_acc_mean, linewidth=2, label="Val accuracy", color="tab:blue")
    ax.fill_between(x_val, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.2, color="tab:blue")

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_title(f"Mean ± Std Accuracy over epochs: {chosen_model}")
    plt.xticks(xticks)
    ax.legend(loc="best")
    ax.grid(True)
    _save(fig, plot_local_path / f"Model_accuracy_mean_std_{n_edges}_links_{n_epo}_epochs_{chosen_model}.pdf", dpi)


def plot_xai_nodes_raw_values(config):
    chosen_model = config.model_type
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)
    topk = 10
    cmap = "viridis"  # cividis
    path_pred_LP = config.p_base + f"pred_likely_pos_no_training_genes_probes_bc_{chosen_model}.csv"
    #path_pred_LP = config.p_base + f"pred_negatives_{chosen_model}.csv"
    path_signals = config.p_base + "combined_pos_neg_signals_bc.csv"
    df_pred_LP = pd.read_csv(path_pred_LP, sep="\t")
    print(df_pred_LP)
    df_signals = pd.read_csv(path_signals, sep="\t")
    
    xai_nodes = df_pred_LP[: topk]
    print("XAI nodes")
    print(xai_nodes)
    list_nodes = xai_nodes["test_gene_names"].tolist()
    print(list_nodes)
    print("Top signals")
    df_top_signals = df_signals[list_nodes]
    print(df_top_signals)
    group_labels = ["Breast Cancer"] * 50 + ["Normal"] * 30
    df_top_signals["Group"] = group_labels
    group_colors = {"Breast Cancer": "darkred", "Normal": "steelblue"}
    row_colors = df_top_signals["Group"].map(group_colors)

    print(df_top_signals)

    # Drop the group column before plotting
    data = df_top_signals.drop(columns=["Group"])

    # ----------------------------
    # Plot heatmap
    # ----------------------------
    sns.set(style="white", font_scale=1.0)

    g = sns.clustermap(
        data,
        row_colors=row_colors,
        cmap=cmap, #RdBu_r Blues
        col_cluster=False,
        row_cluster=False,
        linewidths=0.3,
        figsize=(8, 8),
        cbar_pos=None,  # we'll add our own colorbar outside
    )

    # --- remove row labels and ticks ---
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.tick_params(left=False)

    # force labels to exist and set rotation
    labels = list(data.columns)

    # --- widen right margin to host legends ---
    g.fig.subplots_adjust(right=0.80, top=0.95, bottom=0.05)

    # --- add colorbar (heatmap scale) outside on the right ---
    mappable = g.ax_heatmap.collections[0]  # QuadMesh of the heatmap
    cax = g.fig.add_axes([1.1, 0.25, 0.02, 0.50])  # [left, bottom, width, height] in fig coords
    cb = g.fig.colorbar(mappable, cax=cax)
    cb.set_label("β-value", rotation=90, labelpad=10)

    # --- add group legend (row_colors) outside under the colorbar ---
    handles = [Patch(facecolor=group_colors[k], label=k) for k in ["Breast Cancer", "Normal"]]
    g.fig.legend(
        handles=handles,
        title="Group",
        loc="center left",
        bbox_to_anchor=(1.05, 1), #(0.85, 0.12),  # under the colorbar
        frameon=False,
)

    # Titles and formatting
    plt.suptitle(f"DNA methylation of predicted top {topk} likely positive features (probe_genes) \n(Breast Cancer vs Normal Patients)", fontsize=14, y=1.03)
    plt.tight_layout()
    path = plot_local_path / f"Top_likely_predicted_genes_edges_{n_edges}_links_{n_epo}_epochs_{chosen_model}.pdf"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    #plot_radar_runs(config)
    #plot_mean_std_loss_acc(config)
    plot_xai_nodes_raw_values(config)
