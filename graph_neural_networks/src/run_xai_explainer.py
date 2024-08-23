import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer, CaptumExplainer
from torch_geometric.data import Data
import matplotlib.pyplot as plt

import gnn_network

config = {
    "SEED": 32,
    "n_edges": 2000000,
    "n_epo": 4,
    "k_folds": 5,
    "batch_size": 128,
    "num_classes": 5,
    "gene_dim": 40,
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "scale_features": "degree,ring,NetShort",
    "out_links": "../../pu_label_propagation/data/output/out_links.csv",
    "out_genes": "../../pu_label_propagation/data/output/out_genes.csv",
    "out_gene_rankings": "../../pu_label_propagation/data/output/out_gene_rankings.csv",
    "merged_signals": "../../process_illumina_arrays/data/output/merged_signals.csv",
    "nedbit_features": "../../pu_label_propagation/data/output/nedbit_features.csv",
    "dnam_features": "../../pu_label_propagation/data/output/dnam_features.csv",
    "nedbit_dnam_features": "../data/output/df_nebit_dnam_features.csv",
    "plot_local_path": "../data/output/",
    "data_local_path": "../data/output/",
    "model_local_path": "../model/"
}

def load_model(model_path, data):
    #model = gnn_network.GCN(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = gnn_network.GPNA(config, data)
    model = model.to(device)
    
    print(model)
    
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return model

def gnn_explainer(model, data):
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
            value=10,
        )
    )
    '''explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
        threshold_config=dict(
            threshold_type='topk',
            value=200,
        ),
    )'''
    data_local_path = config["data_local_path"]
    df_test_ids = pd.read_csv(data_local_path + "pos_likely_pos.csv", sep=",")
    explore_test_ids = df_test_ids["test_probe_ids"].tolist()
    explore_test_ids = [int(item) for item in explore_test_ids]
    explore_test_ids.append(10841)
    plot_local_path = config["plot_local_path"]
    plot_local_path += "explainer_plots/" 
    for node_i in explore_test_ids:
        print("Generating subgraph for {}".format(node_i))
        node_index = node_i
        explanation = explainer(data.x, data.edge_index, index=node_index)
        #print(f'Generated explanations in {explanation.available_explanations}')
        plt.figure(figsize=(8, 6))
        plt.grid(True)
        path = plot_local_path + 'subgraph_{}.pdf'.format(node_index)
        explanation.visualize_graph(path=path, backend="networkx")
        print(f"Subgraph visualization plot has been saved to '{path}'")
        print("----")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(config["data_local_path"] + 'data.pt')
    data = data.to(device)
    model = load_model(config["model_local_path"] + "trained_model_edges_2000000_epo_4.ptm", data) 
    gnn_explainer(model, data)