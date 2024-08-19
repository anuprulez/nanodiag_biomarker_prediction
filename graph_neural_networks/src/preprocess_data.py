
import plot_gnn

import torch
from torch_geometric.data import Data

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, RobustScaler


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping


def save_mapping_json(lp, mapping_file):
    with open(lp, 'gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping_file))


def replace_name_by_ids(dataframe, col_index, mapper):
    names = dataframe.iloc[:, col_index]
    lst_names = names.tolist()
    ids = [mapper[mapper["name"] == name]["id"].values[0] for name in lst_names]
    dataframe.iloc[:, col_index] = ids
    return dataframe


def merge_features(config):
    print("Reading NedBit features...")
    df_nebit_features = pd.read_csv(config["nedbit_features"], sep=",")
    nebit_features = df_nebit_features.iloc[:, 3:]
    print(df_nebit_features)

    netshort = np.array(nebit_features["NetShort"].tolist())
    netshort = netshort.reshape(-1, 1)
    print(len(netshort), netshort.shape)
    transformer = RobustScaler().fit(netshort)
    norm_netshort = transformer.transform(netshort)
    nebit_features["NetShort"] = norm_netshort
    
    feature_names = df_nebit_features["name"].tolist()
    print("Reading {}".format(config["merged_signals"]))
    df_merged_signals = pd.read_csv(config["merged_signals"], sep="\t", engine="c")
    dnam_signals = df_merged_signals[feature_names]
    dnam_signals_transpose = dnam_signals.transpose()
    dnam_signals_transpose.to_csv(config["dnam_features"])
    dnam_signals_transpose = dnam_signals_transpose.reset_index()
    dnam_features = dnam_signals_transpose.iloc[:, 1:]

    df_nebit_dnam_features = pd.concat([df_nebit_features, dnam_features], axis=1)
    nebit_dnam_features_embeddings = df_nebit_dnam_features.iloc[:, 3:]
    print(nebit_dnam_features_embeddings)
    print("Assigning labels...")
    df_apu_labels = pd.read_csv(config["out_gene_rankings"], sep=" ", header=None)
    print(df_apu_labels)
    l_name = list()
    l_labels = list()
    for i, item in df_nebit_features.iterrows():
        r_val = item.values
        matched_row = df_apu_labels[df_apu_labels.loc[:, 0] == r_val[0]]
        if len(matched_row.index) > 0:
            l_name.append(r_val[0])
            l_labels.append(matched_row.values[0][2])

    df_labels = pd.DataFrame(zip(l_name, l_labels), columns=["feature_name", "labels"])
    labels = df_labels["labels"].tolist()
    df_nebit_dnam_features["labels"] = labels
    df_nebit_dnam_features.to_csv(config["nedbit_dnam_features"], sep=",", index=None)
    print("Plotting UMAP using raw features")
    plot_gnn.plot_features(nebit_dnam_features_embeddings, labels, config)
    return nebit_dnam_features_embeddings, labels


def read_files(config):
    '''
    Read raw data files and create Pytorch dataset
    '''
    naipu_dnam_features, labels = merge_features(config)
    data_local_path = config["data_local_path"]
    n_edges = config["n_edges"]
    print("Probe genes relations")
    relations_probe_ids = pd.read_csv(config["out_links"], sep=" ", header=None)
    print(relations_probe_ids)
    print("Edges created")
    print("NAIPU and DNAM features and labels")
    print(naipu_dnam_features)
    print()
    print("Labels")
    print(len(labels))
    print()
    out_genes = pd.read_csv(config["out_genes"], sep=" ", header=None)
    print("Out genes NIAPU")
    print(out_genes)
    print("Feature names")
    feature_names = out_genes.iloc[:, 1]
    print(feature_names)
    print()
    print("Mapped feature names to ids")
    mapped_feature_names = out_genes.loc[:, 0]
    print(mapped_feature_names)
    print()
    print("Mapped links before sampling")
    #links_relation_probes = relations_probe_ids[:n_edges]
    print(relations_probe_ids[:n_edges])
    print("Mapped links after sampling")
    links_relation_probes = relations_probe_ids.sample(n_edges)
    print(links_relation_probes)
    print("Add cg01550473_HSPA6 to links")
    cg01550473_HSPA6 = relations_probe_ids[(relations_probe_ids.loc[:, 0] == 10841) | (relations_probe_ids.loc[:, 1] == 10841)]
    cg01550473_HSPA6.reset_index(drop=True, inplace=True)
    links_relation_probes.reset_index(drop=True, inplace=True)
    print(cg01550473_HSPA6)
    links_relation_probes = pd.concat([links_relation_probes, cg01550473_HSPA6], axis=0, ignore_index=True)
    links_relation_probes = links_relation_probes.drop_duplicates()
    print(links_relation_probes)
    print()
    print("Creating X and Y")
    x = naipu_dnam_features
    y = labels
    # shift labels from 1...5 to 0..4
    y = [int(i) - 1 for i in y]
    y = torch.tensor(y, dtype=torch.long)
    # create data object
    print("Features")
    print(x)
    x = torch.tensor(x.to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(links_relation_probes.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y
    return compact_data, feature_names, mapped_feature_names, out_genes