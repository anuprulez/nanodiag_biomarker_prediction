import sys

import plot_gnn
import utils

import torch
from torch_geometric.data import Data

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


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
    #df_nebit_features = pd.read_csv(config["nedbit_features"], sep=",")
    df_nebit_features = pd.read_csv(config.p_nedbit_features, sep=",", header='infer', engine=None)
    print(df_nebit_features)
    feature_names = df_nebit_features["name"].tolist()
    print("Reading {}".format(config.p_combined_pos_neg_signals))
    #df_merged_signals = pd.read_csv(config["merged_signals"], sep="\t", engine="c")
    df_merged_signals = pd.read_csv(config.p_combined_pos_neg_signals, sep="\t", engine="c", header='infer')
    #fake_merged_signals = np.zeros((34, 11756)) #pd.read_csv(config["merged_signals"], sep="\t", engine="c")
    #df_fake_merged_signals = pd.DataFrame(fake_merged_signals, columns=feature_names)
    dnam_signals = df_merged_signals[feature_names]
    dnam_signals_transpose = dnam_signals.transpose()
    dnam_signals_transpose.to_csv(config.p_dnam_features)
    dnam_signals_transpose = dnam_signals_transpose.reset_index()
    dnam_features = dnam_signals_transpose.iloc[:, 1:]

    df_nebit_dnam_features = pd.concat([df_nebit_features, dnam_features], axis=1)
    print(df_nebit_dnam_features)
    nebit_dnam_features_embeddings = df_nebit_dnam_features.iloc[:, 2:]
    print(nebit_dnam_features_embeddings)

    print("Assigning labels...")
    df_apu_labels = pd.read_csv(config.p_out_gene_rankings, sep=" ", header=None)
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
    df_nebit_dnam_features.to_csv(config.p_nedbit_dnam_features, sep=",", index=None)
    return nebit_dnam_features_embeddings, labels


def read_files(config):
    '''
    Read raw data files and create Pytorch dataset
    '''
    #sfeatures = config["scale_features"].split(",")
    naipu_dnam_features, labels = merge_features(config)
    #data_local_path = config["data_local_path"]
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

    #print("df_features")
    #print(df_features)

    # separate train and test nodes
    lst_mapped_f_name = np.array(feature_names.index)
    print(lst_mapped_f_name)
    complete_rand_index = [item for item in range(len(feature_names.index))]
    tr_index, te_index = train_test_split(complete_rand_index, shuffle=True, test_size=0.33, random_state=42)
    tr_nodes = lst_mapped_f_name[tr_index]
    te_nodes = lst_mapped_f_name[te_index]
    print("tr_nodes: ", len(tr_nodes), tr_index[:5], tr_nodes[:5])
    print("te_nodes: ", len(te_nodes), te_index[:5], te_nodes[:5])
    print("intersection: ", list(set(tr_nodes).intersection(set(te_nodes))))

    df_tr_nodes = pd.DataFrame(tr_nodes, columns=["training_node_ids"])
    #df_tr_nodes.to_csv(data_local_path + "training_node_ids.csv", index=None)

    df_te_nodes = pd.DataFrame(te_nodes, columns=["test_node_ids"])
    #df_te_nodes.to_csv(data_local_path + "test_node_ids.csv", index=None)
    utils.create_gnn_data(naipu_dnam_features, labels, links_relation_probes, mapped_feature_names, te_index, te_nodes, config)