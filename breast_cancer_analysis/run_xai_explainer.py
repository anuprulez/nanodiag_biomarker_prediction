import pandas as pd
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer, CaptumExplainer
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
import itertools
from tqdm import tqdm

import gnn_network

base_path = "data_19_Sept_25/outputs/"

config = {
    "SEED": 32,
    "n_edges": 40000,
    "n_epo": 4,
    "k_folds": 5,
    "batch_size": 128,
    "num_classes": 5,
    "gene_dim": 86,
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "scale_features": "0,1,3",  #"degree,ring,NetShort",
    "out_links": f"{base_path}/out_links_bc.csv",
    "out_genes": f"{base_path}/out_genes_bc.csv",
    "out_gene_rankings": f"{base_path}/out_gene_rankings_bc.csv",
    "merged_signals": f"{base_path}/combined_pos_neg_signals_bc.csv",
    "nedbit_features": f"{base_path}/nedbit_features_bc.csv",
    "dnam_features": f"{base_path}/dnam_features_bc.csv",
    "nedbit_dnam_features": f"{base_path}/df_nebit_dnam_features_bc.csv",
    "nedbit_dnam_features_norm": "data/df_nebit_dnam_features_norm.csv",
    "plot_local_path": f"{base_path}",
    "data_local_path": f"{base_path}",
    "model_local_path": "model/"
}

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
    plot_local_path = config["plot_local_path"]
    print("Running GNN explanation...")
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

    data_local_path = config["data_local_path"]
    df_test_ids = pd.read_csv(data_local_path + "pred_likely_pos_no_training_genes_probes_bc.csv", sep="\t")
    #explore_test_ids = df_test_ids["probe_gene_ids"].tolist()
    #explore_test_ids = [int(item) for item in explore_test_ids]
    explore_test_ids = [84] #explore_test_ids[:2]
    plot_local_path = config["plot_local_path"]
    plot_local_path += "explainer_plots/" 
    for node_i in explore_test_ids:
        print("Generating subgraph for {}".format(node_i))
        path = plot_local_path + 'subgraph_{}.pdf'.format(node_i)
        # predict_candidate_genes_gnn_explainer(model, dataset, predictions, path, explanation_nodes_ratio=1, masks_for_seed=10, G=None, num_pos='all'):
        '''node_index = node_i
        explanation = explainer(data.x, data.edge_index, index=node_index)
        print(f'Generated explanations in {explanation.available_explanations}')
        plt.figure(figsize=(8, 6))
        plt.grid(True)
        
        
        subgraph = explanation.get_explanation_subgraph()
        print(subgraph)
        G = to_networkx(subgraph, to_undirected=True)
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G, seed=7)
        nx.draw(
            G, pos,
            with_labels=True,
        )
        #plt.show()
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        #plot_with_explanation(explanation, path)
        #explanation.visualize_graph(path=path, backend="networkx")
        print(f"Subgraph visualization plot has been saved to '{path}'")
        print("----")'''


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
            print("seed run: {}".format(i))
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
        #model = model.cuda()
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
    #plt.show()

def collect_pred_labels():
    base_path = "data_19_Sept_25/outputs/"
    nedbit_path = "df_nebit_dnam_features_bc.csv"
    df_nebit_features = pd.read_csv(base_path + nedbit_path, sep=",")
    print(df_nebit_features.head())
    df_test_probe_genes = pd.read_csv(base_path + "test_probe_genes.csv", sep=",")
    print(df_test_probe_genes.head())
    probe_gene_list = df_test_probe_genes.iloc[:, 1].tolist()
    df_nebit_features_test = df_nebit_features[df_nebit_features["name"].isin(probe_gene_list)]
    df_nebit_features_test.reset_index(drop=True, inplace=True)
    nebit_features = df_nebit_features_test.iloc[:, 2:-1]
    print(nebit_features.head())
    nebit_features_norm = scale_features(["degree", "ring", "NetShort"], nebit_features)

    feature_name = df_nebit_features_test.iloc[:, 0]
    labels = df_nebit_features_test.iloc[:, -1]

    df_labels = pd.DataFrame(zip(feature_name.tolist(), labels.tolist()), columns=["feature_name", "labels"])
    print(df_labels.head())
    feature_names = df_labels["feature_name"].tolist()
    nebit_dnam_features = nebit_features
    df_nebit_dnam_features = nebit_dnam_features
    df_nebit_dnam_features["labels"] = labels
    df_nebit_dnam_features["feature_names"] = df_labels["feature_name"].tolist()

    true_labels = torch.load(base_path + "true_labels.pt", weights_only=False)
    pred_labels = torch.load(base_path + "pred_labels.pt", weights_only=False)
    true_labels = [int(item) + 1 for item in true_labels]
    pred_labels = [int(item) + 1 for item in pred_labels]

    df_labels["pred_labels"] = pred_labels
    pred_pos = df_labels[(df_labels["labels"].isin([1])) & \
                                    (df_labels["pred_labels"].isin([1]))]
    
    pred_likely_pos = df_labels[(df_labels["labels"].isin([2, 3, 4, 5])) & \
                                    (df_labels["pred_labels"].isin([2]))]
    
    print(pred_likely_pos)
    

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(config["data_local_path"] + 'data.pt', weights_only=False)
    data = data.to(device)
    model = load_model(config["model_local_path"] + "trained_model_edges_40000_epo_4.ptm", data)
    #find_k_hop_subgraph(data, model)
    #gnn_explainer(model, data)

    plot_local_path = config["plot_local_path"]
    plot_local_path += "explainer_plots/"
    node_i = 3306
    path = plot_local_path + 'subgraph_{}.pdf'.format(node_i)
    G = to_networkx(data,
                    node_attrs=['x'], # optional: include node attributes
                    #edge_attrs=['weight'], # optional: include edge attributes
                    to_undirected=True)
    collect_pred_labels()
    predict_candidate_genes_gnn_explainer(model, data, path, node_i, explanation_nodes_ratio=1, masks_for_seed=10, G=G, num_pos='all')