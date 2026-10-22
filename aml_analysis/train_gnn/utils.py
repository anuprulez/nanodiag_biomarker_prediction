import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import to_undirected, is_undirected, coalesce

import pandas as pd
import numpy as np
import json

import plot_gnn
import gnn_network


def read_csv(csv_path, sep=",", engine="c", header=None):
    df = pd.read_csv(csv_path, sep=sep, header=header, engine=engine)
    return df


def save_accuracy_scores(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def detach_from_gpu(tensor):
    return tensor.cpu().detach().numpy()


def choose_model(config, data, chosen_model, training=False):
    name = chosen_model.lower()
    if name == "pna":
        return gnn_network.GPNA(config, data)
    elif name == "gcn":
        return gnn_network.GCN(config)
    elif name == "graphsage":
        return gnn_network.GraphSAGE(config)
    elif name == "gatv2":
        return gnn_network.GATv2(config)
    elif name == "graphtransformer":
        return gnn_network.GraphTransformer(config)
    else:
        return gnn_network.GPNA(config, data, training)


def create_gnn_data(features, labels, l_probes, mapped_feature_ids, config):
    print("Creating data ojbect for GNN...")
    sfeatures_names = config.keep_feature_names.split(",")
    sfeatures_ids = [int(i) for i in range(len(sfeatures_names))]
    out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    labels = np.array(labels)

    print("Final features before normalisation")
    print(features)

    x = features
    y = labels
    # shift labels from 1...5 to 0..4 for ML training
    y = [int(i) - 1 for i in y]
    y = torch.tensor(y, dtype=torch.long)
    # create data object
    x = torch.tensor(x.to_numpy(), dtype=torch.float)
    num_nodes = x.shape[0]
    print(f"Number of nodes: {num_nodes}")
    edge_index = torch.tensor(l_probes.to_numpy(), dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    # create undirected graph
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    data = Data(x=x, edge_index=edge_index)
    print(
        f"Is Graph undirected: {is_undirected(edge_index, num_nodes=num_nodes)}"
    )  # should be True
    print(f"edge_index.shape: {edge_index.shape}")
    assert edge_index.dim() == 2
    assert edge_index.size(0) == 2  # must be 2×E
    assert edge_index.dtype == torch.long
    # set up true labels
    data.y = y
    manifest = pd.read_csv(config.p_split_manifest)
    split_by_name = manifest.set_index("feature_name")["split"]
    node_names = out_genes.iloc[:, 1].tolist()
    missing = set(node_names) - set(split_by_name.index)
    if missing:
        raise ValueError(f"Split manifest is missing {len(missing)} graph nodes")
    for split in ("train", "val", "test"):
        ids = [node_id for node_id, name in zip(mapped_feature_ids, node_names)
               if split_by_name[name] == split]
        names = [name for name in node_names if split_by_name[name] == split]
        setattr(data, f"{split}_mask", torch.tensor(
            [node_id in set(ids) for node_id in mapped_feature_ids], dtype=torch.bool
        ))
        pd.DataFrame({f"{split}_gene_ids": ids, f"{split}_gene_names": names}).to_csv(
            getattr(config, f"p_{split}_probe_genes"), index=False
        )

    assert not torch.any(data.train_mask & data.val_mask)
    assert not torch.any(data.train_mask & data.test_mask)
    assert not torch.any(data.val_mask & data.test_mask)
    assert torch.all(data.train_mask | data.val_mask | data.test_mask)
    split_code = torch.full((num_nodes,), -1, dtype=torch.long)
    split_code[data.train_mask] = 0
    split_code[data.val_mask] = 1
    split_code[data.test_mask] = 2
    if edge_index.numel() and not torch.all(
        split_code[edge_index[0]] == split_code[edge_index[1]]
    ):
        raise ValueError("Merged graph contains a cross-split edge")
    print(
        f"Split masks: train={data.train_mask.sum().item()}, "
        f"val={data.val_mask.sum().item()}, test={data.test_mask.sum().item()}"
    )

    for col_idx in sfeatures_ids:
        # Fit preprocessing on training nodes only, then apply it unchanged to
        # validation and test nodes.
        scaler = MinMaxScaler().fit(data.x[data.train_mask, col_idx].reshape(-1, 1))
        scaled = scaler.transform(data.x[:, col_idx].reshape(-1, 1))
        data.x[:, col_idx] = torch.tensor(scaled[:, 0], dtype=torch.float)

    # save normalized data
    torch.save(data, config.p_torch_data)
    preprocessed_data = data.x.detach()
    preprocessed_data_labels = data.y.detach()
    df_preprocessed_data = pd.DataFrame(preprocessed_data.numpy())
    df_preprocessed_data["labels"] = preprocessed_data_labels.numpy()
    df_preprocessed_data.to_csv(config.p_nedbit_dnam_features_norm, sep=",", index=None)
