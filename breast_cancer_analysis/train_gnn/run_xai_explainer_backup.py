import pandas as pd
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from omegaconf.omegaconf import OmegaConf

import gnn_network


def load_model(model_path, data):
    device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)
    model = gnn_network.GPNA(config, data)
    # model = model.to(device)
    print(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def ensure_bool(m):
    return m if m.dtype == torch.bool else m.bool()


def load_local_neighbourhood(xai_node, data):
    seen_explain_batches = dict()
    explainer_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.test_mask),  # seed nodes = train
        num_neighbors=[15, 10],  # list(config.num_neighbors),
        batch_size=1,
        shuffle=False,
        num_workers=8,  # config.num_workers,
        pin_memory=True,
        directed=True,
    )
    for tr_idx, batch in enumerate(explainer_loader):
        global_ids = batch.n_id.cpu()
        where = (global_ids == xai_node).nonzero(as_tuple=True)[0]
        if where.numel():
            local_i = int(where[0])
        if xai_node == global_ids[0]:
            seen_explain_batches[xai_node] = (
                batch.x.detach().cpu(),
                batch.edge_index.detach().cpu(),
                batch.y.detach().cpu(),
                global_ids.clone(),  # n_id mapping
                local_i,
            )
            break
    return seen_explain_batches


import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer


@torch.no_grad()
def _predict(model, x, edge_index):
    model.eval()
    return model(x, edge_index).argmax(dim=1)


def predict_candidate_genes_gnn_explainer_khop_GPT(
    model,
    data,  # PyG Data with data.x, data.edge_index, data.y
    xai_node: int,  # GLOBAL node id to explain
    path: str,  # where to save the plot (pdf/png)
    num_hops: int = 1,  # GNN receptive field depth
    masks_for_seed: int = 1,  # averaging multiple runs; for GNNExplainer deterministic data this is optional
    explanation_nodes_ratio: float = 1.0,  # how many nodes to keep via edge importance coverage
    id_to_name: Optional[Dict[int, str]] = None,  # optional global-id -> display-name
    device: Optional[torch.device] = None,
) -> Tuple[List[int], Dict[int, Tuple[int, float]]]:
    """
    Deterministically:
      1) builds k-hop subgraph around `xai_node`,
      2) runs GNNExplainer on the *local index*,
      3) ranks candidate neighbor nodes by edge importance,
      4) plots with original (global) node ids/names.
    Returns:
      - sorted list of candidate global ids,
      - ranking dict {global_id: (hit_count, total_importance)}.
    """
    device = (
        "cpu"  # device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- 1) Deterministic k-hop subgraph (LOCAL relabeling) ----
    full_x, full_ei, full_y = data.x, data.edge_index, data.y
    print("K-hop subgraph ...")
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=int(xai_node),
        num_hops=num_hops,
        edge_index=full_ei,
        relabel_nodes=True,
    )
    # subset: 1D tensor [n_sub] mapping LOCAL -> GLOBAL ids
    # mapping: scalar/local index where the original node appears in subgraph
    local_idx = int(mapping)
    sub_x = full_x[subset].to(device)
    sub_ei = sub_ei.to(device)
    sub_y = full_y[subset].to(device)
    print("Explanation using K-hop subgraph ...")
    # ---- 2) Explain on LOCAL index ----
    explainer = Explainer(
        model=model.to(device).eval(),
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification", task_level="node", return_type="log_probs"
        ),
    )

    # mean_mask = torch.zeros(sub_ei.size(1), device=device)
    mean_mask = torch.zeros(sub_ei.shape[1]).to("cpu")

    for _ in range(masks_for_seed):
        explanation = explainer(sub_x, sub_ei, index=local_idx)
        # mean_mask += explanation.edge_mask.to(device)
        mean_mask += explanation.edge_mask.to("cpu")
    # mean_mask /= max(1, masks_for_seed)  # [num_edges_sub]
    mean_mask = torch.div(mean_mask, masks_for_seed)
    print("Shape of mean mask: {}".format(mean_mask.shape))

    """
    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=200), explanation_type='model', node_mask_type='attributes', \
                              edge_mask_type='object', model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs',),)
            
            #explanation = explainer(x, edge_index, index=idx)
            explanation = explainer(subg_x, subg_edge_index, index=local_mapping_idx)
    """

    # ---- 3) Rank candidate nodes by edge importances (LOCAL → GLOBAL) ----
    # We’ll pick edges in descending importance until we have ~num_nodes_to_cover nodes.
    num_nodes_to_cover = math.ceil(sub_x.size(0) * float(explanation_nodes_ratio))
    values, edge_idx_sorted = torch.sort(mean_mask, descending=True)

    # prediction on subgraph (optional filter)
    preds = _predict(model, sub_x, sub_ei)  # [n_sub]

    seed_local = local_idx
    covered = set([seed_local])
    # Score per LOCAL node
    local_score = {}

    for epos in edge_idx_sorted.tolist():
        src = int(sub_ei[0, epos])
        dst = int(sub_ei[1, epos])

        # accumulate importance to both incident nodes (excluding seed itself for candidates)
        for node_l in (src, dst):
            if node_l == seed_local:
                continue
            local_score[node_l] = local_score.get(node_l, 0.0) + float(
                values[edge_idx_sorted == epos]
            )

        covered.add(src)
        covered.add(dst)
        if len(covered) >= num_nodes_to_cover:
            break

    # Convert LOCAL → GLOBAL; optionally to names
    ranking = {}  # global_id -> (hit_count, total_importance)
    for node_l, score in local_score.items():
        g_id = int(subset[node_l])
        ranking[g_id] = (1, score)  # hit_count=1 here (single seed)

    sorted_global = sorted(
        ranking.keys(), key=lambda gid: (ranking[gid][0], ranking[gid][1]), reverse=True
    )

    # ---- 4) Plot with original node ids/names using NetworkX ----
    # Build a NetworkX graph in LOCAL space, but label nodes by GLOBAL ids/names.
    G_sub = nx.Graph()
    n_sub = sub_x.size(0)
    G_sub.add_nodes_from(range(n_sub))
    G_sub.add_edges_from(sub_ei.t().tolist())

    # Labels: original global ids or provided names
    if id_to_name is None:
        labels = {i: str(int(subset[i])) for i in range(n_sub)}  # show global ids
    else:
        labels = {
            i: id_to_name.get(int(subset[i]), str(int(subset[i]))) for i in range(n_sub)
        }

    # Node styling: seed red; 1-hop blue; 2-hop green; else lightgray
    # hop distance on undirected view for clarity
    hop_d = dict(
        nx.single_source_shortest_path_length(
            G_sub.to_undirected(), seed_local, cutoff=num_hops
        )
    )
    node_colors = []
    for i in range(n_sub):
        if i == seed_local:
            node_colors.append("red")
        elif hop_d.get(i, 99) == 1:
            node_colors.append("tab:blue")
        elif hop_d.get(i, 99) == 2:
            node_colors.append("tab:green")
        else:
            node_colors.append("lightgray")

    # Edge widths from mean_mask (LOCAL order!)
    em = mean_mask.detach().cpu()
    em = (em - em.min()) / (em.max() - em.min() + 1e-12)
    edge_widths = [0.5 + 3.0 * float(em[k]) for k in range(sub_ei.size(1))]

    pos = nx.spring_layout(G_sub, seed=0)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G_sub,
        pos,
        with_labels=False,
        node_color=node_colors,
        node_size=120,
        width=edge_widths,
    )
    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=6)
    nx.draw_networkx_nodes(
        G_sub, pos, nodelist=[seed_local], node_color=["red"], node_size=300
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return sorted_global, ranking


def predict_candidate_genes_gnn_explainer(
    model,
    dataset,
    path,
    xai_node,
    explanation_nodes_ratio=1,
    masks_for_seed=10,
    G=None,
    num_pos="all",
):
    # sub_x, sub_edge_index, sub_labels, g_ids, local_i = sub_dataset[xai_node]
    # labels = data.y
    # print("Explaining...")
    # print(sub_x)
    # print(sub_edge_index)
    # print(sub_labels)
    # print(g_ids)
    # print(local_i)
    x = dataset.x
    labels = dataset.y
    edge_index = dataset.edge_index

    ranking = {}
    candidates = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    # nodes_names = list(G.nodes)

    i = 0
    """for node in G:
        if labels[i] == 1:
            if node == xai_node:
                print(node, dataset.x[i], dataset.y[i])
            nodes_with_idxs[node] = i
        i += 1
    print('[+]', len(nodes_with_idxs), 'likely positive nodes found in the graph')"""

    # sub_nodes_with_idxs = dict(itertools.islice(nodes_with_idxs.items(), 1))
    sub_nodes_with_idxs = {
        "8337": 8337
    }  # {k: v for k, v in nodes_with_idxs.items() if v == xai_node}
    print(sub_nodes_with_idxs)
    # Get the subgraphs of a positive nodes
    for node in sub_nodes_with_idxs:
        idx = sub_nodes_with_idxs[node]
        print("Node idx: {}".format(idx))
        subg_nodes, subg_edge_index, local_mapping_idx, _ = k_hop_subgraph(
            idx, 1, edge_index, relabel_nodes=True
        )
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]
        print(
            f"Node {node}, {idx}: {subg_nodes}, {subg_edge_index}, {local_mapping_idx}"
        )

    # Get explanations of all the positive genes
    nodes_explained = 0
    num_pos = len(sub_nodes_with_idxs)
    print(sub_nodes_with_idxs, num_pos)

    subg_x = x[subg_nodes]
    print(f"print(subg_x): {subg_x}, {subg_x.shape}")
    print(f"print(subg_edge_index): {subg_edge_index}, {subg_edge_index.shape}")
    print(subg_x)
    # subg_x = subg_x.to(device)
    # subg_nodes = subg_nodes.to(device)
    # subg_edge_index = subg_edge_index.to(device)
    # model = model.to(device)

    for node in tqdm(sub_nodes_with_idxs):
        idx = sub_nodes_with_idxs[node]

        candidates[node] = {}

        mean_mask = torch.zeros(subg_edge_index.shape[1]).to("cpu")

        for i in range(masks_for_seed):
            print(f"seed run: {i}/{masks_for_seed}")
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=200),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="node",
                    return_type="log_probs",
                ),
            )

            # explanation = explainer(x, edge_index, index=idx)
            explanation = explainer(subg_x, subg_edge_index, index=local_mapping_idx)
            mean_mask += explanation.edge_mask.to("cpu")

        mean_mask = torch.div(mean_mask, masks_for_seed)
        print("Shape of mean mask: {}".format(mean_mask.shape))
        num_nodes = int(round(subg_numnodes_d[idx][0] * explanation_nodes_ratio))
        print("Number of nodes: {}".format(num_nodes))
        print(sub_nodes_with_idxs)
        print()
        print(subg_numnodes_d)
        values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1])
        print("Indices: ", len(values), len(indices))
        print("Number of selected edges: {}".format(len(indices)))

        model.eval()
        # out = model(data.x, data.edge_index)
        out = model(subg_x, subg_edge_index)
        predictions = out.argmax(dim=1)
        print("Predictions done!")

        seen_genes = set()

        for i in range(len(indices)):
            src = subg_edge_index[0][indices[i]]
            trgt = subg_edge_index[1][indices[i]]

            src_name = nodes_names[src]
            trgt_name = nodes_names[trgt]

            src_pred = predictions[src]
            trgt_pred = predictions[trgt]

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)
            if trgt_name != node:
                seen_genes.add(trgt_name)

            # if src_pred == 1: # LP
            if src_pred in [0, 1]:  # P, LP
                if src_name not in candidates[node]:
                    candidates[node][src_name] = values[i]
                else:
                    candidates[node][src_name] += values[i]

            # if trgt_pred == 1: # LP
            if src_pred in [0, 1]:  # P, LP
                if trgt_name not in candidates[node]:
                    candidates[node][trgt_name] = values[i]
                else:
                    candidates[node][trgt_name] += values[i]

            # when the seen geens set reaches the num_nodes threshold
            # break the loop
            if len(seen_genes) >= num_nodes:
                break

    for seed in candidates:
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate].item()]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate].item()

    sorted_ranking = sorted(
        ranking, key=lambda x: (ranking[x][0], ranking[x][1]), reverse=True
    )

    print(sorted_ranking, len(sorted_ranking))

    s_rankings_explained_node = [xai_node] + sorted_ranking
    s_rankings_draw = s_rankings_explained_node[:10]
    new_nodes = []
    for enum, n in enumerate(G.nodes(data=True)):
        if n[0] not in s_rankings_draw:
            continue
        new_nodes.append(n[0])

    k = G.subgraph(new_nodes)
    pos = nx.spring_layout(k)
    nx.draw(k, pos=pos, with_labels=True)
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    return new_nodes, s_rankings_explained_node


def collect_pred_labels(config):
    print("Collecting datasets ...")
    # df_nebit_features = pd.read_csv(config.p_nedbit_dnam_features, sep=",")
    df_test_probe_genes = pd.read_csv(config.p_test_probe_genes, sep=",")
    # probe_gene_list = df_test_probe_genes.iloc[:, 1].tolist()
    # df_nebit_features_test = df_nebit_features[df_nebit_features["name"].isin(probe_gene_list)]
    # df_nebit_features_test.reset_index(drop=True, inplace=True)
    # feature_name = df_nebit_features_test.iloc[:, 0]
    # labels = df_nebit_features_test.iloc[:, -1]
    # df_labels = pd.DataFrame(zip(feature_name.tolist(), labels.tolist()), columns=["feature_name", "labels"])
    df_test_probe_genes.columns = ["test_gene_ids", "test_gene_names"]
    df_labels = df_test_probe_genes

    print("Test dataframe reconstructed")
    print(df_labels)

    true_labels = torch.load(config.p_true_labels, weights_only=False)
    pred_labels = torch.load(config.p_pred_labels, weights_only=False)
    pred_probs = torch.load(config.p_pred_probs, weights_only=False)
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


def get_node_names_links(n_nodes, ranked_nodes, config):
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    df_plotted_nodes = df_out_genes[df_out_genes.iloc[:, 0].isin(n_nodes)]
    print(f"Plotted {n_nodes} nodes")
    print("Dataframe of plotted nodes")
    print(df_plotted_nodes)


import copy, math, torch, networkx as nx, matplotlib.pyplot as plt
from torch_geometric.loader import NeighborLoader
from torch_geometric.explain import Explainer, GNNExplainer


def make_bounded_subgraph(data, global_idx, num_neighbors=[10, 5], directed=True):
    # Single seed ⇒ seed is local index 0
    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([int(global_idx)]),
        num_neighbors=num_neighbors,
        batch_size=1,
        shuffle=False,
        directed=directed,
    )
    batch = next(iter(loader))
    sub_x, sub_ei, subset = batch.x, batch.edge_index, batch.n_id  # local→global
    local_idx = 0
    return sub_x, sub_ei, subset, local_idx


def predict_candidate_genes_gnn_explainer_cpu(
    model,
    data,
    xai_node,
    path,
    num_neighbors=[10, 5],
    masks_for_seed=1,
    explanation_nodes_ratio=1.0,
):
    # ---- bounded subgraph around the seed ----
    sub_x, sub_ei, subset, local_idx = make_bounded_subgraph(
        data, xai_node, num_neighbors=num_neighbors, directed=True
    )

    # ---- move explainer work to CPU ----
    model_cpu = copy.deepcopy(model).cpu().eval()
    sub_x, sub_ei = sub_x.cpu(), sub_ei.cpu()

    explainer = Explainer(
        model=model_cpu,
        algorithm=GNNExplainer(epochs=200),  # fewer epochs is fine here
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",  # keep; see note below
        model_config=dict(
            mode="multiclass_classification", task_level="node", return_type="log_probs"
        ),
    )

    mean_mask = torch.zeros(sub_ei.size(1))
    for _ in range(masks_for_seed):
        expn = explainer(sub_x, sub_ei, index=local_idx)  # local index (0)
        mean_mask += expn.edge_mask.cpu()
    mean_mask /= max(1, masks_for_seed)

    # ---- simple candidate ranking from edge scores ----
    k_nodes = math.ceil(sub_x.size(0) * float(explanation_nodes_ratio))
    vals, eidx = torch.sort(mean_mask, descending=True)

    covered = {local_idx}
    local_score = {}
    for pos in eidx.tolist():
        u, v = int(sub_ei[0, pos]), int(sub_ei[1, pos])
        for n in (u, v):
            if n == local_idx:
                continue
            local_score[n] = local_score.get(n, 0.0) + float(mean_mask[pos])
        covered.add(u)
        covered.add(v)
        if len(covered) >= k_nodes:
            break

    # LOCAL → GLOBAL ids
    ranking = {int(subset[n]): (1, score) for n, score in local_score.items()}
    sorted_global = sorted(
        ranking, key=lambda g: (ranking[g][0], ranking[g][1]), reverse=True
    )

    plot_topk_nodes_from_pyg_subgraph(
        sub_edge_index=sub_ei,
        subset=subset,
        local_seed=local_idx,
        k=10,  # show top 25 by explainer score
        edge_mask=mean_mask,
        path=path,
        directed=False,
    )

    # ---- plot with original global ids ----
    """import numpy as np
    G = nx.Graph()
    n_sub = sub_x.size(0)
    G.add_nodes_from(range(n_sub))
    G.add_edges_from(sub_ei.t().tolist())

    labels = {i: str(int(subset[i])) for i in range(n_sub)}  # show GLOBAL ids
    hop = dict(nx.single_source_shortest_path_length(G, local_idx, cutoff=len(num_neighbors)))
    node_colors = ["red" if i==local_idx else
                   ("tab:blue" if hop.get(i, 99)==1 else
                    "tab:green" if hop.get(i, 99)==2 else "lightgray")
                   for i in range(n_sub)]
    em = (mean_mask - mean_mask.min()) / (mean_mask.max() - mean_mask.min() + 1e-12)
    widths = [0.5 + 3.0*float(em[k]) for k in range(sub_ei.size(1))]

    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=120, width=widths)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    nx.draw_networkx_nodes(G, pos, nodelist=[local_idx], node_color=["red"], node_size=300)
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()"""

    return sorted_global, ranking


def plot_topk_nodes_from_pyg_subgraph(
    sub_edge_index,  # torch.LongTensor [2, E] (LOCAL indices)
    subset=None,  # torch.LongTensor [n_sub]; LOCAL -> GLOBAL ids (optional, for labels)
    local_seed=None,  # int or None; local index of seed to force-include
    k=10,  # how many nodes to show
    edge_mask=None,  # torch.Tensor [E] or None; explainer edge scores
    path="topk_subgraph.pdf",
    directed=False,
):
    """
    Builds a NetworkX graph from a LOCAL PyG subgraph and draws only the top-K nodes.
    Scoring:
      - If edge_mask is provided: node_score = sum of incident edge scores
      - Else: node_score = degree (on undirected view)
    Labels show GLOBAL ids if `subset` is provided, else LOCAL ids.
    """
    # 1) Build NX graph
    G = nx.DiGraph() if directed else nx.Graph()
    edges = sub_edge_index.t().tolist()
    # Get n_sub from largest index in edges
    n_sub = (sub_edge_index.max().item() if sub_edge_index.numel() > 0 else -1) + 1
    G.add_nodes_from(range(n_sub))
    G.add_edges_from(edges)

    # Use undirected graph for scoring "connectedness"
    Gu = G.to_undirected()

    # 2) Compute node scores
    if edge_mask is not None:
        edge_mask = edge_mask.detach().cpu()
        # normalize to [0,1] for nicer widths/colors later
        em = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-12)
        # sum edge score to incident nodes
        node_score = {i: 0.0 for i in range(n_sub)}
        for eid, (u, v) in enumerate(edges):
            s = float(em[eid])
            node_score[u] += s
            node_score[v] += s
    else:
        # degree-based score
        node_score = dict(Gu.degree())

    # 3) Pick top-K nodes (force-include seed if provided)
    top = sorted(node_score.keys(), key=lambda i: node_score[i], reverse=True)
    if local_seed is not None and local_seed not in top[:k]:
        # ensure seed is part of the set
        top = [local_seed] + [i for i in top if i != local_seed]
    top_nodes = set(top[:k])

    # Induce subgraph on top-K nodes
    H = Gu.subgraph(top_nodes).copy()

    # If very sparse, optionally keep only the largest connected component
    if H.number_of_nodes() > 0 and not nx.is_connected(H):
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()

    # 4) Edge widths (optional, if edge_mask provided)
    widths = None
    if edge_mask is not None:
        # map each H-edge back to its eid in edges list to fetch normalized score
        eid_map = {tuple(e): i for i, e in enumerate(edges)}
        eid_map.update(
            {(v, u): i for (u, v), i in eid_map.items()}
        )  # handle undirected lookup
        em = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-12)
        widths = [0.5 + 3.0 * float(em[eid_map[(u, v)]]) for u, v in H.edges()]

    # 5) Labels: GLOBAL ids if given, else LOCAL
    if subset is not None:
        labels = {i: str(int(subset[i])) for i in H.nodes()}  # original/global ids
    else:
        labels = {i: str(i) for i in H.nodes()}  # local ids

    # Highlight seed
    node_colors = []
    for i in H.nodes():
        if local_seed is not None and i == local_seed:
            node_colors.append("red")
        else:
            node_colors.append("tab:blue")

    pos = nx.spring_layout(H, seed=0)
    plt.figure(figsize=(7, 6))
    nx.draw(
        H,
        pos,
        with_labels=False,
        node_color=node_colors,
        node_size=180,
        width=widths if widths is not None else 1.0,
    )
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=7)
    if local_seed is not None and local_seed in H:
        nx.draw_networkx_nodes(
            H, pos, nodelist=[local_seed], node_color=["red"], node_size=320
        )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    plot_local_path = config.p_plot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(config.p_torch_data, weights_only=False)
    # data = data.to(device)
    model_path = config.p_torch_model
    # f"{config.p_model}trained_model_edges_{config.n_edges}_epo_{config.n_epo}.ptm"
    model = load_model(model_path, data)
    # model = model.to(device)
    node_i = 8337
    path = plot_local_path + "subgraph_{}.pdf".format(node_i)
    """G = to_networkx(data,
                    node_attrs=['x'],
                    to_undirected=True)"""
    # collect_pred_labels(config)
    # seen_explain_batches = load_local_neighbourhood(node_i, data)
    # local_xai_graph = seen_explain_batches[node_i]
    G = None
    # plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer(model, data, path, node_i, explanation_nodes_ratio=1, masks_for_seed=config.exp_epo, G=G, num_pos='all')
    # plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer_khop_GPT(model, data, node_i, path, num_hops=1)
    plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer_cpu(
        model,
        data,
        node_i,
        path,
        num_neighbors=[100],
        masks_for_seed=config.exp_epo,
        explanation_nodes_ratio=1.0,
    )
    get_node_names_links(plotted_nodes, ranked_nodes, config)
