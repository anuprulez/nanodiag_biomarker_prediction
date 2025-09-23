import pandas as pd
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
from tqdm import tqdm
from omegaconf.omegaconf import OmegaConf

import gnn_network


def load_model(model_path, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = gnn_network.GPNA(config, data)
    model = model.to(device)
    print(model)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return model


def gnn_explainer(model, data, topk=10):
    print("Running GNN explanation...")
    plot_local_path = config.p_plot
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
        threshold_config=dict(
            threshold_type='topk',
            value=topk,
        )
    )
    data_local_path = config.p_data
    df_test_ids = pd.read_csv(data_local_path + "pred_likely_pos_no_training_genes_probes_bc.csv", sep="\t")
    explore_test_ids = [84]
    plot_local_path += "explainer_plots/" 
    for node_i in explore_test_ids:
        print("Generating subgraph for {}".format(node_i))
        path = plot_local_path + 'subgraph_{}.pdf'.format(node_i)
        node_index = node_i
        explanation = explainer(data.x, data.edge_index, index=node_index)
        print(f'Generated explanations in {explanation.available_explanations}')
        explanation.visualize_graph(path=path, backend="networkx")
        print(f"Subgraph visualization plot has been saved to '{path}'")


def predict_candidate_genes_gnn_explainer(model, dataset, path, xai_node, explanation_nodes_ratio=1, \
                                          masks_for_seed=10, G=None, num_pos='all'):
    x           = dataset.x
    labels      = dataset.y
    edge_index  = dataset.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)
    
    i = 0
    for node in G:
        if labels[i] == 1:
            if node == xai_node:
                print(node, dataset.x[i], dataset.y[i])
            nodes_with_idxs[node] = i
        i += 1
    print('[+]', len(nodes_with_idxs), 'likely positive nodes found in the graph')

    #sub_nodes_with_idxs = dict(itertools.islice(nodes_with_idxs.items(), 1))
    sub_nodes_with_idxs = {k: v for k, v in nodes_with_idxs.items() if v == xai_node}
    print(sub_nodes_with_idxs)
    # Get the subgraphs of a positive nodes
    for node in sub_nodes_with_idxs:
        idx = sub_nodes_with_idxs[node]
        print("Node idx: {}".format(idx))
        subg_nodes, subg_edge_index, _, _ = k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    nodes_explained = 0
    num_pos = len(sub_nodes_with_idxs)
    print(sub_nodes_with_idxs, num_pos)
    for node in tqdm(sub_nodes_with_idxs):
        idx = sub_nodes_with_idxs[node]

        candidates[node] = {}

        mean_mask = torch.zeros(edge_index.shape[1]).to('cpu')

        for i in range(masks_for_seed):
            print(f"seed run: {i}/{masks_for_seed}")
            explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=200), explanation_type='model', node_mask_type='attributes', \
                              edge_mask_type='object', model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs',),)
            
            explanation = explainer(x, edge_index, index=idx)
            mean_mask += explanation.edge_mask.to('cpu')

        mean_mask = torch.div(mean_mask, masks_for_seed)
        print("Shape of mean mask: {}".format(mean_mask.shape))
        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))
        print("Number of nodes: {}".format(num_nodes))
        values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1])
        print("Indices: ", len(values), len(indices))
        print("Number of selected edges: {}".format(len(indices)))

        model.eval()
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1)
        print("Predictions done!")

        seen_genes = set()

        for i in range(len(indices)):
            src     = edge_index[0][indices[i]]
            trgt    = edge_index[1][indices[i]]

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


def collect_pred_labels():
    print("Collecting datasets ...")
    p_test_probe_genes = config.p_base + "test_probe_genes.csv"
    p_nedbit_path = config.p_base + "df_nebit_dnam_features_bc.csv"
    df_nebit_features = pd.read_csv(p_nedbit_path, sep=",")
    df_test_probe_genes = pd.read_csv(p_test_probe_genes, sep=",")
    probe_gene_list = df_test_probe_genes.iloc[:, 1].tolist()
    df_nebit_features_test = df_nebit_features[df_nebit_features["name"].isin(probe_gene_list)]
    df_nebit_features_test.reset_index(drop=True, inplace=True)
    feature_name = df_nebit_features_test.iloc[:, 0]
    labels = df_nebit_features_test.iloc[:, -1]
    df_labels = pd.DataFrame(zip(feature_name.tolist(), labels.tolist()), columns=["feature_name", "labels"])

    true_labels = torch.load(config.p_base + "true_labels.pt", weights_only=False)
    pred_labels = torch.load(config.p_base + "pred_labels.pt", weights_only=False)
    pred_probs = torch.load(config.p_base + "pred_probs.pt", weights_only=False)
    true_labels = [int(item) + 1 for item in true_labels]
    pred_labels = [int(item) + 1 for item in pred_labels]

    df_labels["pred_labels"] = pred_labels
    df_labels["pred_probs"] = pred_probs
    pred_pos = df_labels[(df_labels["labels"].isin([1])) & \
                                    (df_labels["pred_labels"].isin([1]))]
    
    pred_likely_pos = df_labels[(df_labels["labels"].isin([2, 3, 4, 5])) & \
                                    (df_labels["pred_labels"].isin([2]))]
    
    pred_likely_pos.to_csv(config.p_data  + "pred_likely_pos.csv", sep="\t", index=None)
    df_out_genes = pd.read_csv(config.p_data + "out_genes_bc.csv", sep=" ", header=None)

    l_genes = list()
    l_probes = list()
    l_gene_ids = list()
    for i, row in pred_likely_pos.iterrows():
        rvals = row.values[0].split("_")
        l_probes.append(rvals[0])
        l_genes.append(rvals[1])
        match = df_out_genes[df_out_genes.iloc[:, 1] == row.values[0]]
        gene_id = match[0].values[0]
        l_gene_ids.append(gene_id)

    pred_likely_pos["genes"] = l_genes
    pred_likely_pos["probes"] = l_probes
    pred_likely_pos["gene_ids"] = l_gene_ids

    df_tr_probe_genes = pd.read_csv(config.p_data + "training_probe_genes.csv")
    training_node_ids = df_tr_probe_genes["tr_gene_ids"].tolist()
    
    tr_probes_genes = df_out_genes[df_out_genes.iloc[:, 0].isin(training_node_ids)]

    tr_genes = list()
    tr_probes = list()
    for i, row in tr_probes_genes.iterrows():
        rvals = row.values[1].split("_")
        tr_probes.append(rvals[0])
        tr_genes.append(rvals[1])

    tr_probes_genes["genes"] = tr_genes
    tr_probes_genes["probes"] = tr_probes
    tr_probes_genes.to_csv(config.p_data + "tr_probes_genes.csv", sep="\t", index=None)

    pred_likely_pos = pred_likely_pos[~pred_likely_pos["genes"].isin(tr_probes_genes["genes"])]
    pred_likely_pos = pred_likely_pos[~pred_likely_pos["probes"].isin(tr_probes_genes["probes"])]

    print("Pred likely pos with no training genes/probes")
    pred_likely_pos = pred_likely_pos.sort_values(by=["pred_probs"], ascending=False)
    pred_likely_pos.to_csv(config.p_data + "pred_likely_pos_no_training_genes_probes_bc.csv", sep="\t", index=None)


def scale_features(list_feature_names, df_features):
    from sklearn.preprocessing import normalize, RobustScaler
    for feature_name in list_feature_names:
        print("Scaling: {}".format(feature_name))
        feature_val = np.array(df_features[feature_name].tolist())
        feature_val = feature_val.reshape(-1, 1)
        print(len(feature_val), feature_val.shape)
        transformer = RobustScaler().fit(feature_val)
        norm_feature_val = transformer.transform(feature_val)
        df_features[feature_name] = norm_feature_val
    return df_features

if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    plot_local_path = config.p_plot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(config.p_data + 'data.pt', weights_only=False)
    data = data.to(device)
    model_path = f"{config.p_model}trained_model_edges_{config.n_edges}_epo_{config.n_epo}.ptm"
    model = load_model(model_path, data)
    node_i = 1586
    path = plot_local_path + 'subgraph_{}.pdf'.format(node_i)
    G = to_networkx(data,
                    node_attrs=['x'], 
                    to_undirected=True)
    collect_pred_labels()
    predict_candidate_genes_gnn_explainer(model, data, path, node_i, explanation_nodes_ratio=1, masks_for_seed=config.exp_epo, G=G, num_pos='all')