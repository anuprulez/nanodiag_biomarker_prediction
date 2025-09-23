import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils


def merge_features(config):
    """
    Merge Nedbit and DNAm features and assign labels
    """
    print("Reading Nedbit and signals features...")
    df_nebit_features = utils.read_csv(config.p_nedbit_features, sep=",", header='infer', engine=None)
    feature_names = df_nebit_features["name"].tolist()
    df_merged_signals = utils.read_csv(config.p_combined_pos_neg_signals, sep="\t", engine="c", header='infer')
    dnam_signals = df_merged_signals[feature_names]
    dnam_signals_transpose = dnam_signals.transpose()
    dnam_signals_transpose.to_csv(config.p_dnam_features)
    dnam_signals_transpose = dnam_signals_transpose.reset_index()
    dnam_features = dnam_signals_transpose.iloc[:, 1:]
    df_nebit_dnam_features = pd.concat([df_nebit_features, dnam_features], axis=1)
    nebit_dnam_features_embeddings = df_nebit_dnam_features.iloc[:, 2:]
    print("Assigning labels...")
    df_apu_labels = utils.read_csv(config.p_out_gene_rankings, sep=" ", header=None)
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
    """
    Read raw data files and create Pytorch dataset
    """
    naipu_dnam_features, labels = merge_features(config)
    print("Reading out links and genes...")
    relations_probe_ids = utils.read_csv(config.p_out_links, sep=" ", header=None)
    out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    feature_names = out_genes.iloc[:, 1]
    mapped_feature_ids = out_genes.loc[:, 0]
    links_relation_probes = relations_probe_ids.sample(config.n_edges)
    links_relation_probes.reset_index(drop=True, inplace=True)
    links_relation_probes = links_relation_probes.drop_duplicates()
    lst_mapped_f_name = np.array(feature_names.index)
    complete_rand_index = [item for item in range(len(feature_names.index))]
    print("Splitting test and train nodes...")
    _, te_index = train_test_split(complete_rand_index, shuffle=True, test_size=config.test_size, random_state=42)
    te_nodes = lst_mapped_f_name[te_index]
    utils.create_gnn_data(naipu_dnam_features, labels, links_relation_probes, mapped_feature_ids, te_nodes, config)