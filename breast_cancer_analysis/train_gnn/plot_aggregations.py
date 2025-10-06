from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    chosen_model = "GraphTransformer"
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

    x_val = np.arange(n_epo)

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
    ax.legend()
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
    ax.legend(loc="best")
    ax.grid(True)
    _save(fig, plot_local_path / f"Model_accuracy_mean_std_{n_edges}_links_{n_epo}_epochs_{chosen_model}.pdf", dpi)


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    #plot_radar(config)
    plot_mean_std_loss_acc(config)
