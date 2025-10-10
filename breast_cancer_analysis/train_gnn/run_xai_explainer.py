import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer

from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader

from omegaconf.omegaconf import OmegaConf

import gnn_network
import plot_gnn
import utils


class LogitsOnly(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base  # your PNA model that returns (logits, penultimate)

    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)
        # handle either tuple (logits, penult) or plain logits
        if isinstance(out, tuple):
            out = out[0]
        return out


def load_model(model_path, data, chosen_model):
    device = "cpu"
    model = utils.choose_model(config, data, chosen_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def explain_candiate_gene(model, dataset, xai_node, G, chosen_model, config):
    """
    Explain xai_node using a sampled neighborhood from NeighborLoader (no full-graph ops).
    """
    assert G is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks_for_seed = config.masks_for_seed
    explainer_epochs = config.explainer_epochs
    neighbour_predictions = config.neighbour_predictions
    explanation_nodes_ratio = config.explanation_nodes_ratio
    y = dataset.y
    nodes_names = list(G.nodes)

    # Map likely positives (kept from your flow; we still only explain xai_node)
    nodes_with_idxs = {}
    for i, node in enumerate(G):
        # use [1] for extracting only likely positives
        # use [1, 2, 3, 4, 5] for extracting negatives
        if y[i] in neighbour_predictions:  # [0, 1, 2, 3, 4]
            nodes_with_idxs[node] = i
    print("[+]", len(nodes_with_idxs), "likely positive nodes found in the graph")

    if xai_node not in nodes_with_idxs:
        raise ValueError(f"xai_node '{xai_node}' not found among likely positives.")

    idx_global = int(nodes_with_idxs[xai_node])

    print(f"idx_global: {idx_global}")

    # Sample neighborhood with NeighborLoader
    loader = NeighborLoader(
        dataset,
        num_neighbors=config.explain_neighbors_spread,
        input_nodes=torch.tensor([idx_global], dtype=torch.long),
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        subgraph_type=config.graph_subtype,
    )

    # Pull exactly one sampled subgraph for this seed
    sub_data = next(iter(loader))
    model = model.to(device)
    x_sub = sub_data.x.to(device)
    ei_sub = sub_data.edge_index.to(device)
    n_id_global = sub_data.n_id.cpu()
    idx_local = 0  # seed first

    print(f"# Global ids: {len(n_id_global)}")

    model.eval()
    with torch.no_grad():
        out_sub, *_ = model(x_sub, ei_sub)
        predictions_sub = out_sub.argmax(dim=1).cpu()  # local predictions
    print("Predictions (subgraph) done!")

    # --------- Aggregate explainer masks on SUBGRAPH ----------
    edge_mask = torch.zeros(ei_sub.shape[1], dtype=torch.float32)
    node_mask = np.zeros((masks_for_seed, x_sub.shape[1]))
    explanation = None
    wrapped = LogitsOnly(model)
    for seed_run in range(masks_for_seed):
        print(f"seed run: {seed_run + 1}/{masks_for_seed}")
        explainer = Explainer(
            model=wrapped,
            algorithm=GNNExplainer(epochs=explainer_epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )
        explanation = explainer(x_sub, ei_sub, index=idx_local)
        edge_mask += explanation.edge_mask.detach().cpu()
        node_mask[seed_run, :] = explanation.node_mask[0].detach().cpu().numpy()

    print(f"Node mask: {node_mask.shape}")

    edge_mask /= float(masks_for_seed)
    mean_node_mask = np.mean(node_mask, axis=0)

    # Rank candidates from SUBGRAPH
    n_sub_nodes = sub_data.num_nodes
    n_sub_edges = ei_sub.shape[1]
    num_nodes_target = max(1, int(round(n_sub_nodes * float(explanation_nodes_ratio))))
    print(
        f"Subgraph nodes={n_sub_nodes}, edges={n_sub_edges}, nodes target={num_nodes_target}"
    )

    values, edge_sel_idx = torch.topk(edge_mask, k=n_sub_edges)
    print("Top edges selected:", len(edge_sel_idx))

    # name mapping: local -> global -> name
    local_to_name = {
        loc: nodes_names[int(n_id_global[loc])] for loc in range(n_sub_nodes)
    }

    print("Computing edge rankings ...")
    ranking = compute_rankings(
        xai_node,
        values,
        edge_sel_idx,
        local_to_name,
        ei_sub,
        predictions_sub,
        neighbour_predictions,
        num_nodes_target,
    )
    sorted_ranking = sorted(
        ranking, key=lambda c: (ranking[c][0], ranking[c][1]), reverse=True
    )

    # Plot neighbours and feature importances
    print("Drawing neighbourhood with local G")
    s_rankings_draw = plot_gnn.draw_xai_local_graph(G, sorted_ranking, idx_global, ei_sub, local_to_name, chosen_model, config)
    print("Drawing neighbourhood with global G")
    plot_gnn.draw_xai_global_graph(G, sorted_ranking, idx_global, chosen_model, config)
    plot_gnn.plot_feature_importance(data, node_mask, mean_node_mask, xai_node, chosen_model, config)
    return s_rankings_draw


def compute_rankings(
    xai_node,
    values,
    edge_sel_idx,
    local_to_name,
    ei_sub,
    predictions_sub,
    neighbour_predictions,
    num_nodes_target,
):
    """
    Compute rankings of links over iterations of explanations
    """
    explained_name = xai_node
    candidates = {explained_name: {}}
    candidate_predictions = {}
    seen_genes = set()

    for k_i in range(len(edge_sel_idx)):
        e_idx = int(edge_sel_idx[k_i])
        src_loc = int(ei_sub[0, e_idx].item())
        trg_loc = int(ei_sub[1, e_idx].item())

        src_name = local_to_name[src_loc]
        trg_name = local_to_name[trg_loc]

        src_pred = int(predictions_sub[src_loc].item())
        trg_pred = int(predictions_sub[trg_loc].item())

        if src_name != explained_name:
            seen_genes.add(src_name)
        if trg_name != explained_name:
            seen_genes.add(trg_name)

        # Your original logic considered [0,1] as P/LP
        if src_pred in neighbour_predictions:
            candidates[explained_name][src_name] = candidates[explained_name].get(
                src_name, 0.0
            ) + float(values[k_i].item())
            candidate_predictions[src_name] = src_pred
        if trg_pred in neighbour_predictions:
            candidates[explained_name][trg_name] = candidates[explained_name].get(
                trg_name, 0.0
            ) + float(values[k_i].item())
            candidate_predictions[src_name] = trg_pred

        if len(seen_genes) >= num_nodes_target:
            break
    candiates_xai = candidates[xai_node]
    print(
        f"sorted candiates_xai: {dict(sorted(candiates_xai.items(), key=lambda item: float(item[1]), reverse=True))}"
    )
    ranking = {}
    for cand_name, score in candidates[explained_name].items():
        if cand_name not in ranking:
            ranking[cand_name] = [1, float(score)]
        else:
            ranking[cand_name][0] += 1
            ranking[cand_name][1] += float(score)
    print(f"Rankings: {ranking}")
    return ranking


def collect_pred_labels(config):
    print("Collecting datasets ...")
    df_test_probe_genes = pd.read_csv(config.p_test_probe_genes, sep=",")
    df_test_probe_genes.columns = ["test_gene_ids", "test_gene_names"]
    df_labels = df_test_probe_genes
    print("Test dataframe reconstructed")
    print(df_labels)
    true_labels = torch.load(config.p_true_labels, weights_only=False)
    pred_labels = torch.load(config.p_pred_labels, weights_only=False)
    pred_probs = torch.load(config.p_best_class_pred_probs, weights_only=False)
    true_labels = [int(item) + 1 for item in true_labels]
    pred_labels = [int(item) + 1 for item in pred_labels]

    df_labels["labels"] = true_labels
    df_labels["pred_labels"] = pred_labels
    df_labels["pred_probs"] = pred_probs

    print(df_labels)

    pred_pos = df_labels[
        (df_labels["labels"].isin([1])) & (df_labels["pred_labels"].isin([1]))
    ]

    pred_likely_pos = df_labels[
        (df_labels["labels"].isin([2, 3, 4, 5])) & (df_labels["pred_labels"].isin([2]))
    ]

    pred_negatives = df_labels[
        (df_labels["labels"].isin([3, 4, 5]))
        & (df_labels["pred_labels"].isin([3, 4, 5]))
    ]

    pred_negatives.to_csv(config.p_pred_negatives, sep="\t", index=None)
    pred_likely_pos.to_csv(config.p_pred_likely_pos, sep="\t", index=None)
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)

    print(df_out_genes)

    l_genes = list()
    l_probes = list()
    l_gene_ids = list()
    for i, row in pred_likely_pos.iterrows():
        rvals = row.values[1].split("_")
        l_probes.append(rvals[0])
        l_genes.append(rvals[1])
        match = df_out_genes[df_out_genes.iloc[:, 1] == row.values[1]]
        gene_id = match[0].values[0]
        l_gene_ids.append(gene_id)

    pred_likely_pos["genes"] = l_genes
    pred_likely_pos["probes"] = l_probes
    pred_likely_pos["gene_ids"] = l_gene_ids

    df_tr_probe_genes = pd.read_csv(config.p_train_probe_genes)
    training_node_ids = df_tr_probe_genes["tr_gene_ids"].tolist()

    tr_probes_genes = df_out_genes[df_out_genes.iloc[:, 0].isin(training_node_ids)]

    print(
        f"training_node_ids: {len(training_node_ids)}, tr_probes_genes {len(tr_probes_genes)}"
    )

    tr_genes = list()
    tr_probes = list()
    for i, row in tr_probes_genes.iterrows():
        rvals = row.values[1].split("_")
        tr_probes.append(rvals[0])
        tr_genes.append(rvals[1])

    tr_probes_genes["genes"] = tr_genes
    tr_probes_genes["probes"] = tr_probes

    pred_likely_pos = pred_likely_pos[
        ~pred_likely_pos["genes"].isin(tr_probes_genes["genes"])
    ]
    pred_likely_pos = pred_likely_pos[
        ~pred_likely_pos["probes"].isin(tr_probes_genes["probes"])
    ]

    print("Pred likely pos with no training genes/probes")
    pred_likely_pos = pred_likely_pos.sort_values(by=["pred_probs"], ascending=False)
    pred_likely_pos.to_csv(
        config.p_pred_likely_pos_no_training_genes_probes, sep="\t", index=None
    )


def get_node_names_links(n_nodes, xai_node, config):
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    df_out_links = pd.read_csv(config.p_out_links, sep=" ", header=None)
    df_plotted_nodes = df_out_genes[df_out_genes.iloc[:, 0].isin(n_nodes)]
    df_xai_node_links_first = df_out_links[df_out_links.iloc[:, 0] == xai_node]
    df_xai_node_links_second = df_out_links[df_out_links.iloc[:, 1] == xai_node]

    df_xai_node_links_plotted_node = df_out_links[
        (df_out_links.iloc[:, 0].isin(n_nodes) & (df_out_links.iloc[:, 1] == xai_node))
    ]
    print(f"Plotted {n_nodes} nodes")
    print("Dataframe of plotted nodes")
    print(df_plotted_nodes)
    print()
    print("All nodes for xai node when xai is in first column")
    print(df_xai_node_links_first)
    print()
    print("All nodes for xai node when xai is in second column")
    print(df_xai_node_links_second)
    df_xai_node_links_first.to_csv(
        f"{config.p_data}df_xai_node_links_first_{xai_node}.csv", index=None
    )
    df_xai_node_links_second.to_csv(
        f"{config.p_data}df_xai_node_links_second_{xai_node}.csv", index=None
    )
    print("All links between xai node and links cache")
    print(df_xai_node_links_plotted_node)


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    plot_local_path = config.p_plot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(config.p_torch_data, weights_only=False)
    chosen_model = config.best_trained_model
    model = load_model(config.p_torch_model, data, chosen_model)
    node_i = 1775 # 2569 #1775 #2569 #7478 # 68 #7868
    # Plot examples: 7868 (LP); 7149 (RN); 68 (LN)
    collect_pred_labels(config)
    #print(f"Creating graph with all nodes ...")
    #G = to_networkx(data, node_attrs=["x"], to_undirected=True)
    # Collect dataframes for different axes
    #p_nodes = explain_candiate_gene(model, data, node_i, G, chosen_model, config)
    #get_node_names_links(p_nodes, node_i, config)