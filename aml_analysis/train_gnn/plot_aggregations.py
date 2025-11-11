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


def plot_radar_runs_multiple(config):
    """
    Aggregate per-run metrics for each model, average the 5 test metrics across runs,
    and generate:
      1) Improved overlay radar (models on axes, metrics in legend)
      2) Small-multiples radars (one radar per model, metrics on axes)
      3) Metric-normalized heatmap (models x metrics)

    Expects files at:
      {config.p_base}/runs/{run_id}/all_metrics_{Model}.json
    for run_id in [1..config.n_runs] and Model in ["PNA","GATv2","GCN","GraphSage","GraphTransformer"].
    """

    # ---- Settings ----
    metric_keys = [
        "te_f1_macro",
        "te_f1_weighted",
        "te_f1_micro",
        "te_precision",
        "te_recall",
    ]
    metric_labels = {
        "te_f1_macro": "F1 (macro)",
        "te_f1_weighted": "F1 (weighted)",
        "te_f1_micro": "F1 (micro)",
        "te_precision": "Precision",
        "te_recall": "Recall",
    }
    models = ["PNA", "GATv2", "GCN", "GraphSage", "GraphTransformer"]

    # Plot toggles & options
    MAKE_OVERLAY_RADAR = True
    MAKE_SMALL_MULTIPLES = True
    MAKE_HEATMAP = True

    SORT_MODELS = True            # sort models for overlay/heatmap
    SORT_BY = "mean"              # "mean" or one of metric_keys
    NORMALIZE_FOR_HEATMAP = True  # per-metric min-max to [0,1]

    plot_local_path = Path(getattr(config, "p_plot", "./plots"))
    p_base = getattr(config, "p_base", "./")
    n_edges = getattr(config, "n_edges", 0)
    n_epo = getattr(config, "n_epo", 0)
    n_runs = getattr(config, "n_runs", 5)
    dpi = getattr(config, "dpi", 200)

    plot_local_path.mkdir(parents=True, exist_ok=True)

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
                    "te_f1_macro": float(values["te_f1_macro"]),
                    "te_f1_weighted": float(values["te_f1_weighted"]),
                    "te_f1_micro": float(values["te_f1_micro"]),
                    "te_precision": float(values["te_precision"]),
                    "te_recall": float(values["te_recall"]),
                })
            print(f"Run: {run_id}, model {model}: {metrics_list[item]}")
        print(f"Aggregated model {model}")

        # Compute scalar averages for the 5 test metrics across runs
        agg_models_perf_avg[model] = {}
        for k in metric_keys:
            vals = [m[k] for m in metrics_list if k in m]
            print(f"Metric: {k}, model: {model}, vals: {vals}")
            agg_models_perf_avg[model][k] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    print("Averaged for models")
    print(agg_models_perf_avg)

    # Optionally sort models to make the overlay/heatmap easier to read
    def model_score(m):
        if SORT_BY == "mean":
            return float(np.nanmean([agg_models_perf_avg[m][k] for k in metric_keys]))
        else:
            return agg_models_perf_avg[m][SORT_BY]

    if SORT_MODELS:
        models = sorted(agg_models_perf_avg.keys(), key=model_score, reverse=True)
    else:
        models = list(agg_models_perf_avg.keys())

    # Utility: build array [len(models) x len(metrics)]
    def build_matrix():
        mat = np.zeros((len(models), len(metric_keys)), dtype=float)
        for i, m in enumerate(models):
            for j, k in enumerate(metric_keys):
                mat[i, j] = agg_models_perf_avg[m][k]
        return mat

    mat = build_matrix()

    # -----------------------------
    # 2) Improved Overlay Radar
    # -----------------------------
    if MAKE_OVERLAY_RADAR:
        N = len(models)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.concatenate([angles, angles[:1]])  # close the loop

        def values_for_metric(key):
            vals = np.array([agg_models_perf_avg[m][key] for m in models], dtype=float)
            return np.concatenate([vals, vals[:1]])

        # Distinct markers per metric to cut overlap ambiguity
        metric_markers = ['o', 's', '^', 'D', 'P']

        plt.figure(figsize=(8.8, 8.8))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)   # start at top
        ax.set_theta_direction(-1)       # clockwise

        # Model labels on axes
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, models, fontsize=11)

        # Radial limits and ticks (scores in 0..1)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=10)
        ax.set_rlabel_position(0)

        # Cleaner grid
        ax.grid(True, linewidth=0.6, alpha=0.5)

        # Color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(metric_keys)))

        # Plot each metric polygon
        for idx, k in enumerate(metric_keys):
            vals = values_for_metric(k)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(angles, vals, linewidth=2.2, marker=metric_markers[idx],
                    markersize=5, label=metric_labels.get(k, k))
            ax.fill(angles, vals, alpha=0.10)

        # Title (reduced pad to cut extra gap)
        ax.set_title(
            f"Model Comparison (Averaged over {n_runs} runs)\nModels on Axes, Metrics in Legend",
            fontsize=13, pad=10
        )

        # Legend outside
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)

        # Trim extra whitespace around the polar plot
        plt.tight_layout(pad=0.8)
        plt.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.82)

        out_path = plot_local_path / f"Radar_overlay_avg_edges_{n_edges}_epochs_{n_epo}.pdf"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved overlay radar -> {out_path}")

    # -------------------------------------
    # 3) Small-Multiples: one radar / model
    # -------------------------------------
    if MAKE_SMALL_MULTIPLES:
        M = len(models)
        L = len(metric_keys)

        angles = np.linspace(0, 2*np.pi, L, endpoint=False)
        angles = np.concatenate([angles, angles[:1]])

        n_cols = min(M, 5)  # 5 models → 5 columns
        n_rows = int(np.ceil(M / n_cols))

        fig, axs = plt.subplots(
            n_rows, n_cols,
            subplot_kw=dict(polar=True),
            figsize=(3.6*n_cols, 3.6*n_rows)
        )
        axs = np.atleast_2d(axs)

        # unify metric tick labels around all small plots
        spoke_labels = [metric_labels[k] for k in metric_keys]

        for i, m in enumerate(models):
            r, c = divmod(i, n_cols)
            ax = axs[r, c]

            vals = np.array([agg_models_perf_avg[m][k] for k in metric_keys], dtype=float)
            vals = np.concatenate([vals, vals[:1]])

            ax.plot(angles, vals, linewidth=2)
            ax.fill(angles, vals, alpha=0.15)

            # Grid, ticks, labels
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.2, 0.4,  0.6, 0.8]) # [0.25, 0.5, 0.75]
            ax.set_yticklabels(["0.2", "0.4",  "0.6", "0.8"], fontsize=8) # []".25", ".5", ".75"
            ax.grid(True, linewidth=0.5, alpha=0.5)
            ax.set_thetagrids(angles[:-1] * 180/np.pi, spoke_labels, fontsize=8)

            ax.set_title(m, fontsize=11, pad=8)

        # Hide any empty subplots
        total_axes = n_rows*n_cols
        for j in range(M, total_axes):
            r, c = divmod(j, n_cols)
            axs[r, c].axis("off")

        fig.suptitle(f"Per-Model Performance (averaged over {n_runs} runs)", fontsize=13, y=1.02)
        plt.tight_layout(pad=0.8)
        plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)

        out_path_sm = plot_local_path / f"Radar_small_multiples_edges_{n_edges}_epochs_{n_epo}.pdf"
        plt.savefig(out_path_sm, dpi=dpi, bbox_inches="tight")
        print(f"Saved small-multiples radars -> {out_path_sm}")

    # ----------------------------
    # 4) Metric-Normalized Heatmap
    # ----------------------------
    if MAKE_HEATMAP:
        A = mat.copy()  # models x metrics

        if NORMALIZE_FOR_HEATMAP:
            # Per-metric min-max normalization to [0,1] for contrast
            mins = np.nanmin(A, axis=0)
            maxs = np.nanmax(A, axis=0)
            span = np.clip(maxs - mins, 1e-12, None)
            A_norm = (A - mins) / span
        else:
            # Normalize globally for luminance calculation only
            gmin, gmax = np.nanmin(A), np.nanmax(A)
            gspan = max(gmax - gmin, 1e-12)
            A_norm = (A - gmin) / gspan

        # Use a colormap with light low-end so text stays visible for small values
        cmap = plt.cm.YlGnBu  # low = light yellow, high = dark blue
        #cmap = plt.cm.magma_r
        #cmap = plt.cm.Greys_r
        fig, ax = plt.subplots(figsize=(1.2*len(metric_keys)+2.5, 0.6*len(models)+2.5))
        im = ax.imshow(A_norm if NORMALIZE_FOR_HEATMAP else A_norm, aspect="auto",
                    interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

        # axis tick labels
        ax.set_xticks(np.arange(len(metric_keys)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([metric_labels[k] for k in metric_keys], rotation=30, ha="right")
        ax.set_yticklabels(models)

        # annotate with values (use original agg values for numbers)
        for i in range(len(models)):
            for j in range(len(metric_keys)):
                val = float(agg_models_perf_avg[models[i]][metric_keys[j]])
                # Luminance-based text color for contrast (ITU-R BT.601)
                r, g, b, _ = cmap(A_norm[i, j])
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = "black" if luminance > 0.55 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color=txt_color)

        # optional: add faint grid lines to separate cells
        ax.set_xticks(np.arange(-.5, len(metric_keys), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_title(f"Models vs Metrics", pad=10)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Metric score", rotation=90, va="center", labelpad=12)

        plt.tight_layout()
        out_path_hm = plot_local_path / f"Heatmap_avg_edges_{n_edges}_epochs_{n_epo}.pdf"
        plt.savefig(out_path_hm, dpi=dpi, bbox_inches="tight")
        print(f"Saved heatmap -> {out_path_hm}")


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
    models = ["PNA", "GATv2", "GCN", "GraphSage", "GraphTransformer"]

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
    n_runs = getattr(config, "n_runs", 2)
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
    #ax.set_ylim(0.0, 1.0)
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
    chosen_model = "PNA"
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
    chosen_model = "PNA"
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)
    topk = 300
    cmap = "cividis"  # cividis viridis  inferno
    path_pred_LP = config.p_base + f"pred_likely_pos_no_training_genes_probes_aml_{chosen_model}.csv"
    #path_pred_LP = config.p_base + f"pred_negatives_{chosen_model}.csv"
    path_signals = config.p_base + "combined_pos_neg_signals_aml.csv"
    df_pred_LP = pd.read_csv(path_pred_LP, sep="\t")
    df_signals = pd.read_csv(path_signals, sep="\t")
    
    xai_nodes = df_pred_LP[: topk]
    list_nodes = xai_nodes["test_gene_names"].tolist()
    df_top_signals = df_signals[list_nodes]

    group_labels = ["Day0"] * 20 + ["Day8"] * 20
    df_top_signals["Group"] = group_labels
    group_colors = {"Day0": "darkred", "Day8": "steelblue"}
    row_colors = df_top_signals["Group"].map(group_colors)

    # Drop the group column before plotting
    data = df_top_signals.drop(columns=["Group"])
    sns.set(style="white", font_scale=1.0)

    g = sns.clustermap(
        data,
        row_colors=row_colors,
        cmap=cmap,
        col_cluster=False,
        row_cluster=False,
        linewidths=0.3,
        figsize=(8, 8),
        cbar_pos=None,
    )

    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.tick_params(left=False)

    g.fig.subplots_adjust(top=0.92, right=0.80, left=0.06, bottom=0.06)

    # --- add colorbar (heatmap scale) outside on the right ---
    mappable = g.ax_heatmap.collections[0]  # QuadMesh of the heatmap
    cax = g.fig.add_axes([1.1, 0.25, 0.02, 0.50])  # [left, bottom, width, height] in fig coords
    cb = g.fig.colorbar(mappable, cax=cax)
    cb.set_label("β-value", rotation=90, labelpad=10)

    # --- add group legend (row_colors) outside under the colorbar ---
    handles = [Patch(facecolor=group_colors[k], label=k) for k in ["Day0", "Day8"]]
    g.fig.legend(
        handles=handles,
        title="Group",
        loc="center left",
        frameon=False,
    )
    # Titles and formatting
    g.ax_heatmap.set_title(f"DNA methylation of predicted top {topk} likely positive features (probe_genes) \n(Day0 vs Day8 Patients)", pad=4, fontsize=14)
    g.ax_col_dendrogram.set_visible(False)
    plt.tight_layout()
    #path = plot_local_path / f"Top_negative_predicted_genes_edges_{n_edges}_links_{n_epo}_epochs_{chosen_model}.pdf"
    path = plot_local_path / f"Top_likely_predicted_genes_edges_{n_edges}_links_{n_epo}_epochs_{chosen_model}.pdf"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages


def _find_run_file(base, run_id, chosen_model):
    """Try a few plausible filenames/locations for each run; return first that exists."""
    candidates = [
        os.path.join(base, f"runs/{run_id}/pred_likely_pos_no_training_genes_probes_aml_{chosen_model}.csv"),
        #os.path.join(base, f"runs/{run_id}/pred_negatives_{chosen_model}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No pred file found for run {run_id}. Tried: {candidates}")


def plot_top_nodes_correlation(config, df_signals, df_lp):
    
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    df_out_genes.iloc[:, 2] = df_out_genes.iloc[:, 2].astype(float)
    df_seed_nodes = df_out_genes[df_out_genes.iloc[:, 2] > 0.0]
    print("Seed nodes df")
    print(df_seed_nodes)
    seed_nodes = df_seed_nodes.iloc[:, 1].tolist()
    print(f"Seed nodes: {seed_nodes[:5], len(seed_nodes)}")
    print("df_signals")
    print(df_signals)
    seed_nodes = ["cg09242307_SOX5", 
                  "cg10990959_SOX5",
                  "cg02147465_ZBTB20",
                  "cg13697223_GALNTL6", 
                  "cg06126815_PON2",
                  "cg09962458_SIPA1L1",
                  "cg16036046_RIT2"
                  ]
    df_seed = df_signals[seed_nodes]
    print("Seed signals")
    print(df_seed)
    print("Top LP signals")
    top_lp = "cg13985132_LOC390595"
    df_lp = df_signals[[top_lp]]
    print(df_lp)

    df_seed = df_seed.apply(pd.to_numeric, errors='coerce')
    df_lp = df_lp.apply(pd.to_numeric, errors='coerce')

    # --- Create output directory ---
    p_corr = f"{config.p_plot}correlation_plots"
    os.makedirs(p_corr, exist_ok=True)

    # --- Compute correlations ---
    correlation_results = {}

    for lp_col in df_lp.columns:
        correlations = []
        for seed_col in df_seed.columns:
            corr = df_lp[lp_col].corr(df_seed[seed_col])
            correlations.append(corr)
        correlation_results[lp_col] = correlations

    # --- Convert to DataFrame for easier analysis ---
    df_corr = pd.DataFrame(correlation_results, index=df_seed.columns)
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("crest", as_cmap=True)
    print("Plotting correlation plots between seeds and top LP genes")
    # --- Plot correlations for each LP signal ---
    for lp_col in df_lp.columns:
        corr_series = df_corr[lp_col].sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        bar_colors = sns.color_palette("viridis", len(corr_series))
        sns.barplot(
            x=corr_series.index,
            y=corr_series.values,
            palette=bar_colors
        )

        plt.xticks(rotation=90, fontsize=9)
        plt.yticks(fontsize=10)
        plt.ylabel("Pearson correlation", fontsize=12)
        plt.xlabel("Seed signals", fontsize=12)
        plt.title(f"Correlation of {lp_col} with XAI subgraph seed signals", fontsize=14, weight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Save high-quality PNG and PDF
        plt.savefig(f"{p_corr}/{lp_col}_correlation.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{p_corr}/{lp_col}_correlation.pdf", bbox_inches='tight')
        plt.close()

    print(f"✅ Correlation plots saved to {p_corr} folder.")


def plot_xai_nodes_raw_values_averaged_runs(config):
    # ---- Fixed model & runs ----
    chosen_model = "PNA"  # enforce PNA as requested
    n_runs = 5

    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)
    topk = 20
    cmap = "cividis"  # cividis / viridis / inferno

    # Signals matrix (patients x features), shared across runs
    path_signals = os.path.join(config.p_base, "combined_pos_neg_signals_aml.csv")
    df_signals = pd.read_csv(path_signals, sep="\t")

    # ------------------------------
    # 1) Collect top lists per run
    # ------------------------------
    per_run_lists = []
    for run_id in range(1, n_runs + 1):
        path_pred = _find_run_file(config.p_base, run_id, chosen_model)
        df_pred = pd.read_csv(path_pred, sep="\t")
        print("All: ", df_pred)
        N = [5]
        LP = [2]
        P = [1]
        df_pred = df_pred[df_pred["pred_labels"].isin(LP)]
        print(df_pred)
        if "test_gene_names" not in df_pred.columns:
            raise KeyError(f"'test_gene_names' column not found in {path_pred}")
        per_run_lists.append(df_pred["test_gene_names"].tolist())

    # ------------------------------
    # 2) Build consensus: by frequency, then average rank
    # ------------------------------
    counts = defaultdict(int)
    rank_sums = defaultdict(float)

    for run_list in per_run_lists:
        # limit to topk per run before counting/ranking
        sub = run_list[:topk]
        for rank, feat in enumerate(sub):
            counts[feat] += 1
            rank_sums[feat] += rank

    all_feats = list(counts.keys())
    # average rank where present; (lower avg rank is better)
    avg_rank = {f: (rank_sums[f] / counts[f]) for f in all_feats}

    # Sort: primary = frequency desc, secondary = avg rank asc
    consensus_sorted = sorted(
        all_feats,
        key=lambda f: (-counts[f], avg_rank[f])
    )
    consensus_features = consensus_sorted[:topk]

    # ------------------------------
    # 3) Slice signals for consensus features
    # ------------------------------
    missing = [f for f in consensus_features if f not in df_signals.columns]
    if missing:
        # Keep only those available; warn in console
        print(f"[WARN] {len(missing)} consensus features not in signals; ignoring a few like: {missing[:5]} ...")
    consensus_features = [f for f in consensus_features if f in df_signals.columns]
    df_top_signals = df_signals[consensus_features].copy()


    plot_top_nodes_correlation(config, df_signals, df_top_signals)

    # ------------------------------
    # 4) Groups & cosmetics
    # ------------------------------
    # Adjust these counts if your cohort sizes differ
    group_labels = ["Day0"] * 20 + ["Day8"] * 20
    if len(group_labels) != len(df_top_signals):
        raise ValueError(
            f"Group label length ({len(group_labels)}) doesn't match data rows ({len(df_top_signals)}). "
            "Update the group_labels to your actual cohort sizes."
        )

    df_top_signals["Group"] = group_labels
    group_colors = {"Day0": "#dd8452", "Day8": "#4c72b0"}
    hue_order = ['Day0', 'Day8']
    row_colors = df_top_signals["Group"].map(group_colors)

    data = df_top_signals.drop(columns=["Group"])
    sns.set(style="white", font_scale=1.0)
    pred_type = "likely_positive" #"negative" # likely_positive
    out_path_heatmap = plot_local_path / f"heatmap_top_{pred_type}_predicted_genes_edges_{n_edges}_links_{n_epo}_epochs_{chosen_model}_5runs.pdf"
    out_path_voilin = plot_local_path / f"violin_top_{pred_type}_predicted_genes_edges_{n_edges}_links_{n_epo}_epochs_{chosen_model}_5runs.pdf"

    # ------------------------------
    # 5) Output: multi-page PDF
    # ------------------------------
    #out_path = plot_local_path / f"Top_likely_predicted_genes_edges_{n_edges}_links_{n_epo}_epochs_{chosen_model}_5runs.pdf"
    with PdfPages(out_path_heatmap) as pdf:
        # ===== Page 1: Heatmap (consensus features) =====
        g = sns.clustermap(
            data,
            row_colors=row_colors,
            cmap=cmap,
            col_cluster=False,
            row_cluster=False,
            linewidths=0.3,
            figsize=(8, 8),
            cbar_pos=None,
        )
        g.ax_heatmap.set_yticklabels([])
        g.ax_heatmap.tick_params(left=False)
        g.fig.subplots_adjust(top=0.85, right=0.80, left=0.06, bottom=0.06)

        # Colorbar
        mappable = g.ax_heatmap.collections[0]
        # Move colorbar closer by decreasing the 'left' value from 1.10 → ~0.88–0.92
        cax = g.fig.add_axes([0.88, 0.1, 0.02, 0.50])  # [left, bottom, width, height]
        cb = g.fig.colorbar(mappable, cax=cax)
        cb.set_label("β-value", rotation=90, labelpad=14)  # small extra pad for readability

        # Legend
        handles = [Patch(facecolor=group_colors[k], label=k) for k in ["Day0", "Day8"]]
        g.fig.legend(handles=handles, title="Group", loc="center left", frameon=False)

        # Title
        g.ax_heatmap.set_title(
            f"DNA methylation of consensus top {len(consensus_features)} likely positive features (PNA, 5 runs)\n"
            "(Day0 vs Day8 patients)",
            pad=4, fontsize=14
        )
        g.ax_col_dendrogram.set_visible(False)

        pdf.savefig(g.fig, dpi=dpi, bbox_inches="tight")
        plt.close(g.fig)

    print(f"Saved multipage PDF (PNA aggregated over 5 runs) to: {out_path_heatmap}")


    with PdfPages(out_path_voilin) as pdf:

        title_font = 24
        x_y_font = 16
        mean_annot_font = 14
        # ===== Page 2: Violin plot (day0 vs day8 per feature) =====
        long_df = (
            df_top_signals
            .melt(id_vars="Group", var_name="Feature", value_name="Beta")
            .dropna()
        )

        # limit to first N for readability
        max_features_for_violin = 30
        features_for_violin = consensus_features #[:max_features_for_violin]
        vdf = long_df[long_df["Feature"].isin(features_for_violin)].copy()

        vdf["Group"] = pd.Categorical(vdf["Group"], categories=["Day8", "Day0"], ordered=True)
        vdf["Feature"] = pd.Categorical(vdf["Feature"], categories=features_for_violin, ordered=True)

        plt.figure(figsize=(max(12, 0.4 * len(features_for_violin)), 8))
        ax = sns.violinplot(
            data=vdf,
            x="Feature",
            y="Beta",
            hue="Group",
            hue_order=hue_order,
            cut=0,
            inner='quartile',
            split=True,
            palette=group_colors
        )

        # Overlay mean markers + lines to emphasize mean differences
        sns.pointplot(
            data=vdf,
            x="Feature",
            y="Beta",
            hue="Group",
            dodge=0.4,
            join=True,
            markers="o",
            linestyles="-",
            errorbar=None,
            ax=ax
        )

        # Clean up duplicate legends
        _, labels = ax.get_legend_handles_labels()
        handles = [Patch(facecolor=group_colors[k], label=k) for k in ["Day0", "Day8"]]
        ax.legend(handles[:2], labels[:2], title="Conditions", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.02))

        # Axis labels and title
        ax.set_ylim(0, 1.3)
        ax.set_yticks(np.arange(0, 1.3, 0.2))
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
        ax.set_xlabel("Feature (probe_gene)", fontsize=x_y_font)
        ax.set_ylabel("β-value", fontsize=x_y_font)
        ax.set_title(
            f"Distribution and mean differences per feature (β-values): Day0 vs Day8",
            pad=10,
            fontsize=title_font
        )
        ax.tick_params(axis="y", labelsize=x_y_font)
        ax.tick_params(axis="x", labelsize=x_y_font)
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Annotate per-feature mean difference (Day0 − Day8)
        group_means = (
            vdf.groupby(["Feature", "Group"])["Beta"]
            .mean()
            #.reagindex(features_for_violin if hasattr(pd.core.groupby.generic.SeriesGroupBy, "__len__") else None)  # safe no-op
            .unstack("Group")
        )
        # Safely compute ymax per feature for text placement
        ymax_per_feature = vdf.groupby("Feature")["Beta"].max()
        for i, feat in enumerate(features_for_violin):
            if feat not in group_means.index:
                continue
            mu_day0 = group_means.loc[feat].get("Day0", np.nan)
            mu_day8 = group_means.loc[feat].get("Day8", np.nan)
            if np.isnan(mu_day0) or np.isnan(mu_day8):
                continue
            diff = mu_day0 - mu_day8
            ytxt = ymax_per_feature.get(feat, vdf["Beta"].max()) + 0.02
            ax.text(i, ytxt, f"Δμ={diff:.3f}", ha="center", va="bottom", fontsize=mean_annot_font, rotation=90)

        pdf.savefig(ax.figure, dpi=dpi, bbox_inches="tight")
        plt.close(ax.figure)

    print(f"Saved multipage PDF (PNA aggregated over 5 runs) to: {out_path_voilin}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

def plot_positive_xai_nodes_raw_values(config):
    """
    Violin plot of top 20 positive CpG features comparing Day0 vs Day8.
    Each half of violin uses group color; means and Δμ are annotated.
    """
    base_path = Path(config.p_plot)

    title_font = 24
    x_y_font = 16
    mean_annot_font = 14

    # --- Load data ---
    df_nebit_features = pd.read_csv(base_path / "df_nebit_dnam_features_aml_PNA.csv", sep=",")
    df_signals = pd.read_csv(base_path / "combined_pos_neg_signals_aml.csv", sep="\t")
    df_seed_genes = pd.read_csv(base_path / "out_gene_rankings_aml.csv", sep=" ")

    df_seed_genes.columns = ["names", "associations", "labels"]
    df_seed_genes = df_seed_genes[df_seed_genes["labels"] == 1]
    seed_names = df_seed_genes["names"].tolist()

    random.shuffle(seed_names)

    seed_signals = df_signals[seed_names]

    # --- Select top 20 features and split into groups ---
    topk_features = 20
    df = seed_signals.iloc[:, :topk_features].copy()
    group1 = df.iloc[:20].assign(Group="Day0")
    group2 = df.iloc[20:].assign(Group="Day8")

    df_melted = pd.concat([group1, group2], axis=0).melt(
        id_vars="Group", var_name="CpG", value_name="Value"
    )

    # --- Colors and ordering ---
    hue_order = ["Day0", "Day8"]  # legend order: blue, orange
    group_colors = {"Day0": "#dd8452", "Day8": "#4c72b0"}

    df_melted["Group"] = pd.Categorical(df_melted["Group"], categories=hue_order, ordered=True)
    df_melted["CpG"] = pd.Categorical(df_melted["CpG"], categories=list(df.columns), ordered=True)

    # --- Plot ---
    sns.set(style="white", font_scale=1.0)
    plt.figure(figsize=(max(12, 0.4 * topk_features), 8))
    ax = plt.gca()

    # split violins colored by group
    ax = sns.violinplot(
        data=df_melted,
        x="CpG",
        y="Value",
        hue="Group",
        hue_order=hue_order,
        split=True,
        inner="quartile",
        cut=0,
        linewidth=1,
        palette=[group_colors[g] for g in hue_order]
    )

    # overlay mean lines and markers
    sns.pointplot(
        data=df_melted,
        x="CpG",
        y="Value",
        hue="Group",
        hue_order=hue_order,
        dodge=0.4,
        join=True,
        markers="o",
        linestyles="-",
        errorbar=None,
        palette=[group_colors[g] for g in hue_order],
        ax=ax
    )

    # clean duplicate legend
    _, labels = ax.get_legend_handles_labels()
    handles = [Patch(facecolor=group_colors[k], label=k) for k in ["Day0", "Day8"]]
    ax.legend(handles[-2:], labels[-2:], title="Conditions", frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 1.02))

    # --- Compute per-feature means ---
    group_means = (
        df_melted.groupby(["CpG", "Group"])["Value"]
        .mean()
        .unstack("Group")
    )

    # annotate per-group means at markers
    point_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
    point_lines = point_lines[-len(hue_order):]
    for line, grp in zip(point_lines, hue_order):
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        for xi, yi in zip(xdata, ydata):
            ax.annotate(
                f"{yi:.3f}",
                (xi, yi),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=x_y_font,
                color=group_colors[grp],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
            )

    # Δμ (Day0 − Day8) above each CpG
    ymax_per_cpg = df_melted.groupby("CpG")["Value"].max()
    for i, cpg in enumerate(df.columns):
        if cpg not in group_means.index:
            continue
        mu_day0 = group_means.loc[cpg].get("Day0", np.nan)
        mu_day8 = group_means.loc[cpg].get("Day8", np.nan)
        if np.isnan(mu_day0) or np.isnan(mu_day8):
            continue
        dmu = mu_day0 - mu_day8
        ytxt = ymax_per_cpg.get(cpg, df_melted["Value"].max()) + 0.02
        ax.text(i, ytxt, f"Δμ={dmu:.3f}", ha="center", va="bottom", fontsize=mean_annot_font, rotation=90)

    # --- Cosmetics ---
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.arange(0, 1.3, 0.2))
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_ylabel("β-value", fontsize=x_y_font)
    ax.set_xlabel("Feature (probe_gene)", fontsize=x_y_font)
    plt.xticks(rotation=90, ha="center", fontsize=x_y_font)
    ax.tick_params(axis="y", labelsize=x_y_font)

    title_main = "Distribution and mean differences per feature (β-values): Day0 vs Day8"
    ax.set_title(f"{title_main}", fontsize=title_font, pad=10)

    plt.tight_layout()
    out_path = base_path / "violin_positive_seeds_day0_day8.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")



if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    plot_xai_nodes_raw_values_averaged_runs(config)
    plot_positive_xai_nodes_raw_values(config)
    #plot_radar_runs_multiple(config)
    #plot_radar_runs(config)
    #plot_mean_std_loss_acc(config)
    #plot_xai_nodes_raw_values(config)
