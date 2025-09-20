import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import normalize, RobustScaler
import pandas as pd

import numpy as np
import plot_gnn


def create_test_masks(mapped_node_ids, mask_list, out_genes):
    print("Creating test masks...")
    print(out_genes)
    print("mapped_node_ids:", mapped_node_ids, len(mapped_node_ids))
    print("mask list: ", mask_list, len(mask_list))
    gene_names = out_genes.loc[:, 1]
    gene_ids = out_genes.loc[:, 0]
    updated_mask_list = list()
    probes = dict()
    probe_genes = list()
    probe_genes_ids = list()
    for name in gene_names:
        p_name = name.split("_")[0]
        if p_name not in probes:
            probes[p_name] = 1
        else:
            probes[p_name] += 1
    for m_item in mask_list:
        m_name = out_genes[out_genes.loc[:, 0] == m_item]
        pm_name = m_name.values[0][1].split("_")[0]
        if pm_name in probes and probes[pm_name] == 1:
            updated_mask_list.append(m_item)
            probe_genes.append(m_name.values[0][1])
            probe_genes_ids.append(m_name.values[0][0])
    mask = mapped_node_ids.index.isin(updated_mask_list)
    print("Test mask:", mask, len(mask))
    return torch.tensor(mask, dtype=torch.bool), probe_genes, probe_genes_ids


def scale_features(list_feature_names, df_features):

    for feature_name in list_feature_names:
        print("Scaling: {}".format(feature_name))
        feature_val = np.array(df_features[feature_name].tolist())
        feature_val = feature_val.reshape(-1, 1)
        print(len(feature_val), feature_val.shape)
        transformer = RobustScaler().fit(feature_val)
        norm_feature_val = transformer.transform(feature_val)
        df_features[feature_name] = norm_feature_val
        
    return df_features


def filter_tr_genes(test_probe_ids, out_genes):
    print("Filtering training genes...")
    tr_genes_names = list()
    tr_gene_ids = list()
    for i, item in out_genes.iterrows():
        if item[0] not in test_probe_ids:
            tr_gene_ids.append(item[0])
            tr_genes_names.append(item[1])
    return tr_gene_ids, tr_genes_names
    

def create_gnn_data(features, labels, l_probes, mapped_feature_ids, te_nodes, config):
    print("Creating X and Y")
    print("te_nodes:", te_nodes, len(te_nodes))
    data_local_path = config["data_local_path"]
    sfeatures_ids = config["scale_features"].split(",")
    sfeatures_ids = [int(i) for i in sfeatures_ids]
    out_genes = pd.read_csv(config["out_genes"], sep=" ", header=None)
    print(features)
    labels = np.array(labels)
    
    features_extract = features
    labels_extract = labels
    
    x = features_extract
    y = labels_extract
    # shift labels from 1...5 to 0..4
    y = [int(i) - 1 for i in y]
    y = torch.tensor(y, dtype=torch.long)
    # create data object
    print("Features")
    print(x)
    x = torch.tensor(x.to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(l_probes.to_numpy(), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    data.y = y

    data.test_mask, test_probe_genes, test_probe_ids = create_test_masks(mapped_feature_ids, te_nodes, out_genes)

    print("After creating test masks")

    print(out_genes.head())
    print(len(data.test_mask), len(test_probe_genes), test_probe_genes[:5], len(test_probe_ids), test_probe_ids[:5])

    df_test_probe_genes = pd.DataFrame(zip(test_probe_ids, test_probe_genes), columns=["test_gene_ids", "test_gene_names"])
    df_test_probe_genes.to_csv(data_local_path + "test_probe_genes.csv", index=None)

    tr_gene_ids, tr_gene_names = filter_tr_genes(test_probe_ids, out_genes)
    df_tr_probe_genes = pd.DataFrame(zip(tr_gene_ids, tr_gene_names), columns=["tr_gene_ids", "tr_gene_names"])
    df_tr_probe_genes.to_csv(data_local_path + "training_probe_genes.csv", index=None)

    print(f"Intersection between train and test genes: {set(tr_gene_ids).intersection(set(test_probe_ids))}")

    train_x = data.x[data.test_mask == 0]
    test_x = data.x[data.test_mask == 1]
    print(train_x.shape, test_x.shape)

    cols_normalise = [0, 1] #[0, 1, 2, 3, 4, 5]
    # Apply normalization for train data
    for col_idx in sfeatures_ids:
        print("Scaling column: {}".format(col_idx))
        tr_feature_val = data.x[data.test_mask == 0][:, col_idx]
        tr_feature_val = tr_feature_val.reshape(-1, 1)
        transformer = RobustScaler().fit(tr_feature_val)
        tr_norm_feature_val = transformer.transform(tr_feature_val)
        tr_norm_feature_val = torch.tensor(tr_norm_feature_val, dtype=torch.float)
        tr_norm_feature_val = tr_norm_feature_val.squeeze()
        print("Train feature: ", tr_norm_feature_val.shape)
        print(tr_norm_feature_val)
        tr_mask = data.test_mask == 0
        data.x[tr_mask, col_idx] = tr_norm_feature_val

        te_feature_val = data.x[data.test_mask == 1][:, col_idx]
        te_feature_val = te_feature_val.reshape(-1, 1)
        te_norm_feature_val = transformer.transform(te_feature_val)
        te_norm_feature_val = torch.tensor(te_norm_feature_val, dtype=torch.float)
        te_norm_feature_val = te_norm_feature_val.squeeze()
        print("Test feature: ", te_norm_feature_val.shape)
        te_mask = data.test_mask == 1
        data.x[te_mask, col_idx] = te_norm_feature_val
        
    train_x = data.x[data.test_mask == 0]
    train_y = data.y[data.test_mask == 0]
    
    test_x = data.x[data.test_mask == 1]
    test_y = data.y[data.test_mask == 1]

    print(train_x.shape, train_x)
    print()
    print(test_x.shape, test_x)
    
    torch.save(data, data_local_path + 'data.pt')

    # save normalized data
    preprocessed_data = data.x.detach()
    preprocessed_data_labels = data.y.detach()
    df_preprocessed_data = pd.DataFrame(preprocessed_data.numpy())
    print(df_preprocessed_data)
    df_preprocessed_data["labels"] = preprocessed_data_labels.numpy()
    df_preprocessed_data.to_csv(config["nedbit_dnam_features_norm"], sep=",", index=None)

    print("Plotting UMAP using raw features")
    plot_gnn.plot_features(train_x, train_y, config, "UMAP Visualization of NedBit + DNA Methylation features", "train_before_GNN")
    plot_gnn.plot_features(test_x, test_y, config, "UMAP Visualization of NedBit + DNA Methylation features", "test_before_GNN")