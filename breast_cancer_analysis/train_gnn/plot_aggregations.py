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
    metric_keys = ["te_f1_macro", "te_f1_weighted", "te_f1_micro", "te_precision", "te_recall"]
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
    '''def to_ordered_list(d, keys):
        return [d[k] for k in keys]

    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False)
    # close the loop by appending the first angle at the end
    angles = np.concatenate([angles, angles[:1]])

    # ===== 3) Plot =====
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Nice grid & limits
    ax.set_theta_offset(np.pi / 2)     # start at the top
    ax.set_theta_direction(-1)         # go clockwise
    ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics_labels)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)

    # Plot each model
    for name, mdict in agg_models_perf.items():
        values = to_ordered_list(mdict, metrics_labels)
        values = np.array(values)
        values = np.concatenate([values, values[:1]])  # close the polygon
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.12)

    # Legend & title
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=False)
    plt.title("Classifier Comparison — Radar Plot", fontsize=14, pad=20)

    plt.tight_layout()
    #plt.savefig("radar_classifiers.png", dpi=200)
    #plt.show()'''
    N = len(models)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close the loop

    def values_for_metric(key):
        vals = [agg_models_perf[m][key] for m in models]
        vals = np.array(vals, dtype=float)
        return np.concatenate([vals, vals[:1]])

    # ==== 5) Plot ====
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)     # start at top
    ax.set_theta_direction(-1)         # clockwise

    # Put model names on axes (corners)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, models, fontsize=11)

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
    plt.title("Model Comparison (Models on Axes, Metrics in Legend)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=False)

    plt.tight_layout()
    path = plot_local_path / f"radar_plot_edges_{n_edges}_epochs_{n_epo}.pdf"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    plot_radar(config)




