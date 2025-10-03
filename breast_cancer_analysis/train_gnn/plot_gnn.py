from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import networkx as nx
import torch
import umap
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

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


def plot_loss_acc(n_epo, tr_loss, te_loss, tr_acc, val_acc, te_acc, config):
    """
    Signature unchanged. Saves two PDFs and (as before) shows the last figure.
    """
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    # Preserve original behavior: override n_epo from config
    n_epo = getattr(config, "n_epo", n_epo)
    dpi = getattr(config, "dpi", 200)

    tr_loss = np.asarray(tr_loss)
    te_loss = np.asarray(te_loss)
    val_acc = np.asarray(val_acc)
    tr_acc = np.asarray(tr_acc)
    te_acc = None if te_acc is None else np.asarray(te_acc)

    x_val = np.arange(n_epo)

    # --- Training loss
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_val, tr_loss, linewidth=2)
    ax.plot(x_val, te_loss, linewidth=2)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.grid(True)
    plt.legend(["Training", "Test"])
    ax.set_title(f"Model loss over epochs: {config.model_type}")
    _save(
        fig,
        plot_local_path
        / f"Model_loss_{n_edges}_links_{n_epo}_epochs_{config.model_type}.pdf",
        dpi,
    )

    # Validation/Test/Train accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_val, val_acc, linewidth=2, label="Validation")
    ax.plot(x_val, tr_acc, linewidth=2, label="Train")
    if te_acc is not None:
        ax.plot(x_val, te_acc, linewidth=2, label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title(f"Model accuracy: {config.model_type}")
    _save(
        fig,
        plot_local_path
        / f"Model_accuracy_{n_edges}_links_{n_epo}_epochs_{config.model_type}.pdf",
        dpi,
    )


def plot_confusion_matrix(
    true_labels, predicted_labels, config, classes=[1, 2, 3, 4, 5]
):
    """
    Signature unchanged. Adds robust label handling and optional normalization via config.normalize if present.
    """
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)
    normalize = getattr(config, "normalize", None)  # 'true' | 'pred' | 'all' | None

    y_true = np.asarray([int(x) + 1 for x in true_labels])
    y_pred = np.asarray([int(x) + 1 for x in predicted_labels])

    # If user passed an explicit classes list, respect it; otherwise infer from data
    class_ticks = (
        classes if classes is not None else sorted(np.unique(np.r_[y_true, y_pred]))
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_ticks, normalize=normalize)

    fig, ax = plt.subplots(figsize=(8, 6))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_ticks,
        yticklabels=class_ticks,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    title = f"Confusion Matrix: {config.model_type}"
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    ax.grid(False)  # heatmaps look cleaner without overlaid grid
    _save(
        fig,
        plot_local_path
        / f"Confusion_matrix_NPPI_{n_edges}_epochs_{n_epo}_{config.model_type}.pdf",
        dpi,
    )


def plot_precision_recall(y_true, y_scores, config):
    """
    Multiclass Precision-recall with per-class & micro AP.
    """
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    labels_sorted = np.unique(y_true)
    if y_scores.ndim != 2 or y_scores.shape[1] != len(labels_sorted):
        raise ValueError(
            f"y_scores must be (n_samples, n_classes={len(labels_sorted)}). Got {y_scores.shape}."
        )

    y_true_bin = label_binarize(y_true, classes=labels_sorted)

    precision: Dict[Union[int, str], np.ndarray] = {}
    recall: Dict[Union[int, str], np.ndarray] = {}
    avg_precision: Dict[Union[int, str], float] = {}

    # per-class
    for i, lab in enumerate(labels_sorted):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        precision[str(lab)], recall[str(lab)] = p, r
        avg_precision[str(lab)] = average_precision_score(
            y_true_bin[:, i], y_scores[:, i]
        )

    # micro
    p_micro, r_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    precision["micro"], recall["micro"] = p_micro, r_micro
    avg_precision["micro"] = average_precision_score(
        y_true_bin, y_scores, average="micro"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in [str(l) for l in labels_sorted]:
        ax.plot(
            recall[k],
            precision[k],
            linewidth=2,
            label=f"Class {k} (AP = {avg_precision[k]:.2f})",
        )
    ax.plot(
        recall["micro"],
        precision["micro"],
        linestyle="--",
        linewidth=2,
        label=f"Micro-average (AP = {avg_precision['micro']:.2f})",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Multiclass Precision–Recall Curve: {config.model_type}")
    ax.legend(loc="best")
    ax.grid(True)
    _save(
        fig,
        plot_local_path
        / f"Precision_recall_{n_edges}_N_Epochs_{n_epo}_{config.model_type}.pdf",
        dpi,
    )


def analyse_ground_truth_pos(model, compact_data, out_genes, all_pred, config):
    """
    Produces histogram & KDE PDFs.
    """
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = config.n_epo
    dpi = getattr(config, "dpi", 200)

    # Ground-truth positives (column 2 > 0), IDs in column 0
    ground_truth_pos_genes = out_genes[out_genes.iloc[:, 2] > 0]
    ground_truth_pos_gene_ids = ground_truth_pos_genes.iloc[:, 0].tolist()

    # Test indices from boolean mask
    test_mask = np.asarray(compact_data.test_mask).astype(bool)
    test_index = np.nonzero(test_mask)[0].tolist()

    masked_pos_genes_ids = sorted(
        set(ground_truth_pos_gene_ids).intersection(set(test_index))
    )

    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    all_pred = out.argmax(dim=1).detach().cpu().numpy()

    masked_p_pos_labels = all_pred[masked_pos_genes_ids]
    df_p_labels = pd.DataFrame({"pred_labels": masked_p_pos_labels})

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    g = sns.histplot(data=df_p_labels, x="pred_labels", discrete=True, ax=ax)
    ax.set_xlabel("Predicted classes")
    ax.set_ylabel("Count")
    ax.set_title("Masked positive genes predicted into different classes.")
    g.set_xticks(sorted(df_p_labels["pred_labels"].unique()))
    plt.tight_layout()
    ax.grid(True, axis="y")
    _save(
        fig,
        plot_local_path
        / f"Histogram_positive__NPPI_{n_edges}_epochs_{n_epo}_{config.model_type}.pdf",
        dpi,
    )

    # KDE (add jitter because labels are discrete)
    fig, ax = plt.subplots(figsize=(8, 6))
    jitter = np.random.RandomState(0).normal(scale=0.05, size=len(df_p_labels))
    sns.kdeplot(x=df_p_labels["pred_labels"] + jitter, ax=ax, bw_adjust=0.6)
    ax.set_xlabel("Predicted classes (jittered)")
    ax.set_ylabel("Density")
    ax.set_title("Masked positive genes predicted into different classes.")
    plt.tight_layout()
    ax.grid(True)
    _save(
        fig,
        plot_local_path
        / f"KDE_positive__NPPI_{n_edges}_epochs_{n_epo}_{config.model_type}.pdf",
        dpi,
    )


def plot_features(features, labels, config, title, flag):
    """
    UMAP visualization of feature vectors.
    """
    plot_local_path = _as_path(config.p_plot)
    n_neighbors = getattr(config, "n_neighbors", 15)
    min_dist = getattr(config, "min_dist", 0.1)
    metric = getattr(config, "metric", "euclidean")
    umap_random_state = getattr(config, "umap_random_state", 42)
    dpi = getattr(config, "dpi", 200)

    labels = [int(item) for item in labels]
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=umap_random_state,
    )
    embeddings = reducer.fit_transform(np.asarray(features))
    df = pd.DataFrame(
        {"UMAP1": embeddings[:, 0], "UMAP2": embeddings[:, 1], "Label": labels}
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="Label",
        data=df,
        s=50,
        alpha=0.9,
        ax=ax,
        palette=config.color_palette,
    )
    ax.set_title(title)
    ax.legend(title="Label", loc="best", frameon=True)
    _save(
        fig,
        plot_local_path / f"UMAP_nedbit_dnam_features_{flag}_{config.model_type}.pdf",
        dpi,
    )


def plot_node_embed(features, labels, pred_labels, config, feature_type):
    """
    UMAP of node embeddings
    """
    plot_local_path = _as_path(config.p_plot)
    n_neighbors = getattr(config, "n_neighbors", 15)
    min_dist = getattr(config, "min_dist", 0.1)
    metric = getattr(config, "metric", "euclidean")
    umap_random_state = getattr(config, "umap_random_state", 42)
    dpi = getattr(config, "dpi", 200)

    labels = [int(item) for item in labels]

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=umap_random_state,
    )
    embeddings = reducer.fit_transform(np.asarray(features))
    df = pd.DataFrame(
        {"UMAP1": embeddings[:, 0], "UMAP2": embeddings[:, 1], "Label": labels}
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="Label",
        data=df,
        s=50,
        alpha=1.0,
        ax=ax,
        palette=config.color_palette,
    )
    ax.set_title(f"Embeddings UMAP of last {feature_type} layer: {config.model_type}")
    ax.legend(title="Class", loc="best", frameon=True)
    _save(
        fig,
        plot_local_path
        / f"UMAP_node_embeddings_{n_neighbors}_{min_dist}_{feature_type}_{config.model_type}.pdf",
        dpi,
    )


def plot_feature_importance(data, node_mask, mean_mask, xai_node, config):
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = getattr(config, "n_epo", config.n_epo)
    dpi = getattr(config, "dpi", 200)
    num_features = data.num_node_features
    # Assign groups
    n_nedbit_features = len(config.keep_feature_names.split(","))
    n_bc_features = 50
    n_normal_features = 30
    group_ids = np.zeros(num_features, dtype=int)
    group_ids[:n_nedbit_features] = 0  # Group of nedbit features
    group_ids[n_nedbit_features : n_nedbit_features + n_bc_features] = (
        1  # Group of breast cancer patients
    )
    group_ids[n_nedbit_features + n_bc_features :] = 2  # Group of normal patients
    group_names = ["Nedbit", "Breast cancer", "Normal"]
    # Collect distributions per group
    distributions = []
    labels = []
    means = []
    for g in range(3):
        idx = np.where(group_ids == g)[0]
        distributions.append(
            node_mask[:, idx].flatten()
        )  # all node importances for features in group
        labels.append(group_names[g])
        means.append(mean_mask[idx].mean())  # mean across features in group

    # Plot violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=distributions)
    plt.xticks(range(3), labels)
    plt.ylabel("Feature importance")
    plt.title("Feature importance distributions per group (Nedbit, BC, Normal)")

    # Overlay means
    for i, m in enumerate(means):
        plt.scatter(
            i, m, color="red", marker="o", zorder=5, label="Mean" if i == 0 else ""
        )

    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    _save(
        fig,
        plot_local_path
        / f"Feature_importance_violin_node_{xai_node}_edges_{n_edges}_epo_{n_epo}_{config.model_type}.pdf",
        dpi,
    )
    plt.close()


def draw_xai_local_graph(G, sorted_ranking, idx_global, ei_sub, nodes_sub, config):
    """
    Draw neighbourhood of chosen node
    """
    """
    Draw the explanation subgraph:
      - Nodes: only those in `sorted_ranking[:config.show_num_neighbours]`
      - Edges: only those provided in `ei_sub` (PyG edge_index, shape [2, E])
      - If `nodes_sub` is provided, it's an array/list mapping local -> global node ids.
    """
    new_nodes = []
    legend_elements = []
    node_color_map = {}
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = getattr(config, "n_epo", config.n_epo)
    dpi = getattr(config, "dpi", 200)

    s_rankings_draw = sorted_ranking[: config.show_num_neighbours]
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    df_plotted_nodes = df_out_genes[df_out_genes.iloc[:, 0].isin(s_rankings_draw)]
    df_seed_nodes = df_out_genes[
        (df_out_genes.iloc[:, 0].isin(s_rankings_draw) & df_out_genes.iloc[:, 2] > 0.0)
    ]
    lst_seed_nodes = df_seed_nodes.iloc[:, 0].tolist()

    # Build node list + colors
    for n, data in G.nodes(data=True):
        if n not in s_rankings_draw:
            continue
        new_nodes.append(n)
        if n == idx_global:
            node_color_map[n] = "red"       # explainee node
        elif n in lst_seed_nodes:
            node_color_map[n] = "green"     # seed node
        else:
            node_color_map[n] = "tab:blue"  # others
        node_name = df_plotted_nodes[df_plotted_nodes.iloc[:, 0] == n]
        if not node_name.empty:
            label = f"{node_name.iloc[0, 0]}:{node_name.iloc[0, 1]}"
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", label=label, markersize=5)
            )

    # Normalize ei_sub to numpy
    if isinstance(ei_sub, torch.Tensor):
        ei_np = ei_sub.detach().cpu().numpy()
    else:
        ei_np = np.asarray(ei_sub)

    assert ei_np.shape[0] == 2, "ei_sub must be shape [2, E]"

    # If nodes_sub is provided, map local -> global; else assume ei_sub uses global IDs already
    def to_global(i):
        return nodes_sub[i]
    # Build H with ONLY the edges supplied by ei_sub and ONLY among `new_nodes`
    new_nodes_set = set(new_nodes)
    H = nx.Graph()
    H.add_nodes_from(new_nodes_set)

    # Add edges (remove self-loops and duplicates naturally via Graph)
    src, dst = ei_np[0], ei_np[1]
    for u_loc, v_loc in zip(src, dst):
        u, v = to_global(u_loc), to_global(v_loc)
        if u == v:
            continue  # drop self-loops
        if u in new_nodes_set and v in new_nodes_set:
            H.add_edge(u, v)

    # If some nodes in `new_nodes` became isolated after filtering by ei_sub, keep them (already added above)

    # Colors in H's node order
    colors_in_order = [node_color_map.get(node, "tab:blue") for node in H.nodes()]

    # Layout on the small graph (stable seed for reproducibility)
    pos = nx.spring_layout(H, seed=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(H, pos=pos, with_labels=True, node_color=colors_in_order, ax=ax)

    # Side legend
    if legend_elements:
        plt.legend(
            handles=legend_elements,
            title="Nodes",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

    plt.title(f"Explanation subgraph of seed node: {idx_global}")
    plt.grid(True)

    # Save
    _save(
        fig,
        plot_local_path
        / f"Explanation_local_subgraph_node_{idx_global}_edges_{n_edges}_epo_{n_epo}_{config.model_type}.pdf",
        dpi,
    )
    plt.close()
    return s_rankings_draw


def draw_xai_global_graph(G, sorted_ranking, idx_global, config):
    """
    Draw neighbourhood of chosen node
    """
    new_nodes = []
    legend_elements = []
    node_color_map = {}
    plot_local_path = _as_path(config.p_plot)
    n_edges = config.n_edges
    n_epo = getattr(config, "n_epo", config.n_epo)
    dpi = getattr(config, "dpi", 200)

    s_rankings_draw = sorted_ranking[: config.show_num_neighbours]
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    df_plotted_nodes = df_out_genes[df_out_genes.iloc[:, 0].isin(s_rankings_draw)]
    df_seed_nodes = df_out_genes[
        (df_out_genes.iloc[:, 0].isin(s_rankings_draw) & df_out_genes.iloc[:, 2] > 0.0)
    ]
    print("Seed nodes in plotted nodes")
    print(df_seed_nodes)
    lst_seed_nodes = df_seed_nodes.iloc[:, 0].tolist()

    for enum, n in enumerate(G.nodes(data=True)):
        if n[0] not in s_rankings_draw:
            continue
        new_nodes.append(n[0])
        # assign color
        if n[0] == idx_global:
            node_color_map[n[0]] = "red"  # exlainable node
        elif n[0] in lst_seed_nodes:
            node_color_map[n[0]] = "green"  # seed node
        else:
            node_color_map[n[0]] = "tab:blue"  # others
        node_name = df_plotted_nodes[df_plotted_nodes.iloc[:, 0] == n[0]]
        label = f"{node_name.iloc[0, 0]}:{node_name.iloc[0, 1]}"
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", label=label, markersize=5)
        )
    K = G.subgraph(new_nodes)
    pos = nx.spring_layout(K)
    # extract colors in K's node order
    colors_in_order = [node_color_map[node] for node in K.nodes()]
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(K, pos=pos, with_labels=True, node_color=colors_in_order)
    plt.legend(
        handles=legend_elements,
        title="Nodes",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.title(f"Explanation subgraph of seed node :{idx_global}")
    plt.grid(True)
    # save subgraph plot
    _save(
        fig,
        plot_local_path
        / f"Explaination_global_subgraph_node_{idx_global}_edges_{n_edges}_epo_{n_epo}_{config.model_type}.pdf",
        dpi,
    )
    plt.close()
