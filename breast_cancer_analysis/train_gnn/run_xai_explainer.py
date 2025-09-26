import pandas as pd
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer
from omegaconf.omegaconf import OmegaConf

import gnn_network


def load_model(model_path, data):
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #data = data.to(device)
    model = gnn_network.GPNA(config, data)
    #model = model.to(device)
    print(model)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return model


def ensure_bool(m):
    return m if m.dtype == torch.bool else m.bool()


def load_local_neighbourhood(xai_node, data):
    seen_explain_batches = dict()
    explainer_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.test_mask),  # seed nodes = train
        num_neighbors=[15, 10], #list(config.num_neighbors),
        batch_size=1,
        shuffle=False,
        num_workers=8, #config.num_workers,
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
                global_ids.clone(),   # n_id mapping
                local_i
            )
            break
    return seen_explain_batches

def make_bounded_subgraph(data, global_idx, num_neighbors=[10,5], directed=True):
    # Single seed ⇒ seed is local index 0
    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([int(global_idx)]),
        num_neighbors=num_neighbors,
        batch_size=1,
        shuffle=False,
        directed=directed
    )
    batch = next(iter(loader))
    sub_x, sub_ei, subset = batch.x, batch.edge_index, batch.n_id  # local→global
    local_idx = 0
    return sub_x, sub_ei, subset, local_idx


@torch.no_grad()
def _predict(model, x, edge_index):
    model.eval()
    return model(x, edge_index).argmax(dim=1)

# Assumes you already have this helper from earlier:
# def make_bounded_subgraph(data, global_idx, num_neighbors=[10,5], directed=True):
#     ... returns sub_x, sub_ei, subset, local_idx
#     # subset: LOCAL -> GLOBAL ids (1D tensor)
#     # local_idx: seed's LOCAL index (0 for single-seed NeighborLoader)

def predict_candidate_genes_gnn_explainer_show_all_conn(
    model,
    dataset,                # PyG Data
    path,                   # output figure path
    xai_node,               # GLOBAL node id to explain
    explanation_nodes_ratio=0.3,  # fraction of nodes to keep in the final plot (0..1]
    masks_for_seed=3,       # average multiple explainer runs
    num_neighbors=(30, 10), # keep small to avoid huge subgraphs
    run_on_cpu=False,        # run explainer/model copy on CPU to avoid CUDA OOM
    expl_epochs=80,         # fewer epochs for speed/robustness
    directed=True
):
    # ---- 1) Bounded subgraph around the GLOBAL seed ----
    sub_x, sub_ei, subset, local_idx = make_bounded_subgraph(
        dataset, xai_node, num_neighbors=list(num_neighbors), directed=directed
    )
    # sub_x: [n_sub, F] (LOCAL)
    # sub_ei: [2, E_sub] (LOCAL)
    # subset: [n_sub] LOCAL -> GLOBAL ids
    # local_idx: int (seed LOCAL id; 0 for single-seed NeighborLoader)

    # Choose device for the explainer/model
    if run_on_cpu:
        device = torch.device("cpu")
        model_expl = copy.deepcopy(model).to(device).eval()
        sub_x_dev, sub_ei_dev = sub_x.cpu(), sub_ei.cpu()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_expl = model.to(device).eval()
        sub_x_dev, sub_ei_dev = sub_x.to(device), sub_ei.to(device)

    # ---- 2) GNNExplainer on the LOCAL index ----
    explainer = Explainer(
        model=model_expl,
        algorithm=GNNExplainer(epochs=int(expl_epochs)),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

    mean_mask = torch.zeros(sub_ei_dev.size(1), device=device)
    for i in range(masks_for_seed):
        # print(f"[Explainer] run {i+1}/{masks_for_seed}")
        expn = explainer(sub_x_dev, sub_ei_dev, index=int(local_idx))
        mean_mask += expn.edge_mask.to(device)
    mean_mask /= max(1, masks_for_seed)           # [E_sub]

    # ---- 3) Aggregate edge importance to node scores (LOCAL), pick top-K nodes ----
    # Normalize edge scores for plotting widths later
    em = (mean_mask - mean_mask.min()) / (mean_mask.max() - mean_mask.min() + 1e-12)

    # Node score = sum of incident edge scores
    n_sub = sub_x.size(0)
    node_score = {i: 0.0 for i in range(n_sub)}
    edges_list = sub_ei.t().tolist()
    for eid, (u, v) in enumerate(edges_list):
        s = float(em[eid].item())
        node_score[u] += s
        node_score[v] += s

    # Always include seed (local_idx)
    k_keep = max(1, math.ceil(n_sub * float(explanation_nodes_ratio)))
    ranked_local = sorted(node_score.keys(), key=lambda i: node_score[i], reverse=True)
    if local_idx not in ranked_local[:k_keep]:
        ranked_local = [local_idx] + [i for i in ranked_local if i != local_idx]
    keep_local_nodes = set(ranked_local[:k_keep])

    # ---- 4) Build a NetworkX graph (LOCAL), then induce subgraph on top-K nodes ----
    # We include ALL edges among the kept nodes (so connections among them show up, not only seed links).
    G_local = nx.DiGraph() if directed else nx.Graph()
    G_local.add_nodes_from(range(n_sub))
    G_local.add_edges_from(edges_list)

    H_local = G_local.subgraph(keep_local_nodes).copy()

    # Optional: keep the largest connected component for clarity (on undirected view)
    if H_local.number_of_nodes() > 0:
        Gu = H_local.to_undirected()
        if not nx.is_connected(Gu):
            cc = max(nx.connected_components(Gu), key=len)
            H_local = H_local.subgraph(cc).copy()

    # ---- 5) Plot with GLOBAL ids as labels ----
    # Map LOCAL -> GLOBAL via `subset`
    labels = {i: str(int(subset[i])) for i in H_local.nodes()}  # global ids
    # Edge widths proportional to explainer scores (need e->eid map)
    eid_map = {tuple(e): i for i, e in enumerate(edges_list)}
    if not directed:
        eid_map.update({(v, u): i for (u, v), i in eid_map.items()})

    widths = []
    for (u, v) in H_local.edges():
        eid = eid_map.get((u, v), None)
        w = 1.0
        if eid is not None:
            w = 0.5 + 3.0 * float(em[eid].item())
        widths.append(w)

    # Color seed red, others blue
    node_colors = ["red" if n == local_idx else "tab:blue" for n in H_local.nodes()]

    pos = nx.spring_layout(H_local.to_undirected(), seed=0)
    plt.figure(figsize=(8, 6))
    nx.draw(H_local, pos,
            with_labels=False,
            node_color=node_colors,
            node_size=180,
            width=[]) #width=widths
    nx.draw_networkx_labels(H_local, pos, labels=labels, font_size=7)
    if local_idx in H_local:
        nx.draw_networkx_nodes(H_local, pos, nodelist=[local_idx], node_color=["red"], node_size=320)
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # ---- 6) Return the selected GLOBAL node ids (in ranking order) and full ranked list ----
    selected_globals = [int(subset[i]) for i in ranked_local if i in H_local.nodes()]
    full_rank_globals = [int(subset[i]) for i in ranked_local]
    return selected_globals, full_rank_globals



def predict_candidate_genes_gnn_explainer(model, dataset, path, xai_node, explanation_nodes_ratio=1, \
                                          masks_for_seed=10, G=None, num_pos='all'):
    
    #sub_x, sub_edge_index, sub_labels, g_ids, local_i = sub_dataset[xai_node]
    #labels = data.y
    #print("Explaining...")
    #print(sub_x)
    #print(sub_edge_index)
    #print(sub_labels)
    #print(g_ids)
    #print(local_i)
    num_neighbors = [100, 50]
    sub_x, sub_ei, subset, local_idx = make_bounded_subgraph(
        dataset, xai_node, num_neighbors=num_neighbors, directed=True
    )

    x           = dataset.x
    labels      = dataset.y
    edge_index  = dataset.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    #nodes_names = list(G.nodes)
    
    i = 0
    '''for node in G:
        if labels[i] == 1:
            if node == xai_node:
                print(node, dataset.x[i], dataset.y[i])
            nodes_with_idxs[node] = i
        i += 1
    print('[+]', len(nodes_with_idxs), 'likely positive nodes found in the graph')'''

    #sub_nodes_with_idxs = dict(itertools.islice(nodes_with_idxs.items(), 1))
    sub_nodes_with_idxs = {"8337": 8337} #{k: v for k, v in nodes_with_idxs.items() if v == xai_node}
    print(sub_nodes_with_idxs)
    # Get the subgraphs of a positive nodes
    '''for node in sub_nodes_with_idxs:
        idx = sub_nodes_with_idxs[node]
        print("Node idx: {}".format(idx))
        subg_nodes, subg_edge_index, local_mapping_idx, _ = k_hop_subgraph(idx, 1, edge_index, relabel_nodes=True)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]
        print(f"Node {node}, {idx}: {subg_nodes}, {subg_edge_index}, {local_mapping_idx}")'''

    # Get explanations of all the positive genes
    nodes_explained = 0
    num_pos = len(sub_nodes_with_idxs)
    print(sub_nodes_with_idxs, num_pos)
    
    
    print(f"print(subg_x): {sub_x}, {sub_x.shape}")
    print(f"print(subg_edge_index): {sub_ei}, {sub_ei.shape}")
    #subg_x = subg_x.to(device)
    #subg_nodes = subg_nodes.to(device)
    #subg_edge_index = subg_edge_index.to(device)
    #model = model.to(device)

    for node in tqdm(sub_nodes_with_idxs):
        idx = sub_nodes_with_idxs[node]

        candidates[node] = {}

        mean_mask = torch.zeros(sub_ei.shape[1]).to('cpu')

        for i in range(masks_for_seed):
            print(f"seed run: {i}/{masks_for_seed}")
            explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=200), explanation_type='model', node_mask_type='attributes', \
                              edge_mask_type='object', model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs',),)
            
            #explanation = explainer(x, edge_index, index=idx)
            explanation = explainer(sub_x, sub_ei, index=local_idx)
            mean_mask += explanation.edge_mask.to('cpu')

        mean_mask = torch.div(mean_mask, masks_for_seed)
        print("Shape of mean mask: {}".format(mean_mask.shape))
        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))
        print("Number of nodes: {}".format(num_nodes))
        print(sub_nodes_with_idxs)
        print()
        print(subg_numnodes_d)
        values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1])
        print("Indices: ", len(values), len(indices))
        print("Number of selected edges: {}".format(len(indices)))

        model.eval()
        #out = model(data.x, data.edge_index)
        out = model(sub_x, sub_ei)
        predictions = out.argmax(dim=1)
        print("Predictions done!")

        seen_genes = set()

        for i in range(len(indices)):
            src     = sub_ei[0][indices[i]]
            trgt    = sub_ei[1][indices[i]]

            src_name    = nodes_names[src]
            trgt_name   = nodes_names[trgt]

            src_pred    = predictions[src]
            trgt_pred   = predictions[trgt]

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)
            if trgt_name != node:
                seen_genes.add(trgt_name)

            #if src_pred == 1: # LP
            if src_pred in [0, 1]: # P, LP
                if src_name not in candidates[node]:
                    candidates[node][src_name] = values[i]
                else:
                    candidates[node][src_name] += values[i]

            #if trgt_pred == 1: # LP
            if src_pred in [0, 1]: # P, LP
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
    
    sorted_ranking  = sorted(ranking, key=lambda x: (ranking[x][0], ranking[x][1]), reverse=True)

    print(sorted_ranking, len(sorted_ranking))

    s_rankings_explained_node = [xai_node] + sorted_ranking
    s_rankings_draw = s_rankings_explained_node[:10]
    new_nodes = []
    for enum, n in enumerate(G.nodes(data=True)):
            if n[0] not in s_rankings_draw: continue
            new_nodes.append(n[0])

    k = G.subgraph(new_nodes)
    pos = nx.spring_layout(k)
    nx.draw(k, pos=pos, with_labels = True)
    plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    return new_nodes, s_rankings_explained_node


def collect_pred_labels(config):
    print("Collecting datasets ...")
    #df_nebit_features = pd.read_csv(config.p_nedbit_dnam_features, sep=",")
    df_test_probe_genes = pd.read_csv(config.p_test_probe_genes, sep=",")
    #probe_gene_list = df_test_probe_genes.iloc[:, 1].tolist()
    #df_nebit_features_test = df_nebit_features[df_nebit_features["name"].isin(probe_gene_list)]
    #df_nebit_features_test.reset_index(drop=True, inplace=True)
    #feature_name = df_nebit_features_test.iloc[:, 0]
    #labels = df_nebit_features_test.iloc[:, -1]
    #df_labels = pd.DataFrame(zip(feature_name.tolist(), labels.tolist()), columns=["feature_name", "labels"])
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

    pred_pos = df_labels[(df_labels["labels"].isin([1])) & \
                                    (df_labels["pred_labels"].isin([1]))]
    
    pred_likely_pos = df_labels[(df_labels["labels"].isin([2, 3, 4, 5])) & \
                                    (df_labels["pred_labels"].isin([2]))]
    
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

    print(f"training_node_ids: {len(training_node_ids)}, tr_probes_genes {len(tr_probes_genes)}")

    tr_genes = list()
    tr_probes = list()
    for i, row in tr_probes_genes.iterrows():
        rvals = row.values[1].split("_")
        tr_probes.append(rvals[0])
        tr_genes.append(rvals[1])

    tr_probes_genes["genes"] = tr_genes
    tr_probes_genes["probes"] = tr_probes

    pred_likely_pos = pred_likely_pos[~pred_likely_pos["genes"].isin(tr_probes_genes["genes"])]
    pred_likely_pos = pred_likely_pos[~pred_likely_pos["probes"].isin(tr_probes_genes["probes"])]

    print("Pred likely pos with no training genes/probes")
    pred_likely_pos = pred_likely_pos.sort_values(by=["pred_probs"], ascending=False)
    pred_likely_pos.to_csv(config.p_pred_likely_pos_no_training_genes_probes, sep="\t", index=None)


def get_node_names_links(n_nodes, ranked_nodes, xai_node, config):
    df_out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    df_out_links = pd.read_csv(config.p_out_links, sep=" ", header=None)
    df_plotted_nodes = df_out_genes[df_out_genes.iloc[:, 0].isin(n_nodes)]
    df_xai_node_links = df_out_links[df_out_links.iloc[:, 0] == xai_node]

    #8337, 3132, 7688, 1011, 9525, 3946, 7382, 781, 633, 2830, 4049
    df_xai_node_links_plotted_node = df_xai_node_links[(df_xai_node_links.iloc[:, 1].isin(n_nodes))]
    print(f"Plotted {n_nodes} nodes")
    print("Dataframe of plotted nodes")
    print(df_plotted_nodes)
    print()
    print("All nodes for xai node")
    print(df_xai_node_links)
    print("All links between xai node and links cache")
    print(df_xai_node_links_plotted_node)

import copy, math, torch, networkx as nx, matplotlib.pyplot as plt
from torch_geometric.loader import NeighborLoader
from torch_geometric.explain import Explainer, GNNExplainer

'''def make_bounded_subgraph(data, global_idx, num_neighbors=[10,5], directed=True):
    # Single seed ⇒ seed is local index 0
    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([int(global_idx)]),
        num_neighbors=num_neighbors,
        batch_size=1,
        shuffle=False,
        directed=directed
    )
    batch = next(iter(loader))
    sub_x, sub_ei, subset = batch.x, batch.edge_index, batch.n_id  # local→global
    local_idx = 0
    return sub_x, sub_ei, subset, local_idx'''




if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    plot_local_path = config.p_plot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(config.p_torch_data, weights_only=False)
    #data = data.to(device)
    model_path = config.p_torch_model
    #f"{config.p_model}trained_model_edges_{config.n_edges}_epo_{config.n_epo}.ptm"
    model = load_model(model_path, data)
    #model = model.to(device)
    node_i = 8337
    path = plot_local_path + 'subgraph_{}.pdf'.format(node_i)
    '''G = to_networkx(data,
                    node_attrs=['x'],
                    to_undirected=True)'''
    #collect_pred_labels(config)
    #seen_explain_batches = load_local_neighbourhood(node_i, data)
    #local_xai_graph = seen_explain_batches[node_i]
    G = None
    plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer_show_all_conn(model, data, path, node_i, \
                                                                                      explanation_nodes_ratio=1.0, \
                                                                                      masks_for_seed=10, 
                                                                                      num_neighbors=[10], \
                                                                                      expl_epochs=200,
                                                                                      directed=False
                                                                                    )
    #plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer(model, data, path, node_i, explanation_nodes_ratio=1, masks_for_seed=config.exp_epo, G=G, num_pos='all')
    #plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer_khop_GPT(model, data, node_i, path, num_hops=1)
    #plotted_nodes, ranked_nodes = predict_candidate_genes_gnn_explainer_cpu(model, data, node_i, path, num_neighbors=[100], masks_for_seed=config.exp_epo, explanation_nodes_ratio=1.0)
    
    get_node_names_links(plotted_nodes, ranked_nodes, node_i, config)

    '''
    def predict_candidate_genes_gnn_explainer_show_all_conn(
    model,
    dataset,                # PyG Data
    path,                   # output figure path
    xai_node,               # GLOBAL node id to explain
    explanation_nodes_ratio=0.3,  # fraction of nodes to keep in the final plot (0..1]
    masks_for_seed=3,       # average multiple explainer runs
    num_neighbors=(30, 10), # keep small to avoid huge subgraphs
    run_on_cpu=True,        # run explainer/model copy on CPU to avoid CUDA OOM
    expl_epochs=80,         # fewer epochs for speed/robustness
    directed=True
):
    
    '''