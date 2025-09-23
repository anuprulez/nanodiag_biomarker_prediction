import argparse
import os
import sys
import time

import pandas as pd
import numpy as np


def load_clean_arrays(arrays_path, ):
    df_merged_signals = pd.read_csv(arrays_path, sep="\t")
    print(df_merged_signals.head())
    
    features = df_merged_signals
    feature_names = features.columns.tolist()

    df_feature_names = pd.DataFrame(feature_names, columns=["feature_names"])
    df_feature_names.to_csv("data/output/" + "final_feature_names.tsv", index=None, sep="\t")
    return df_merged_signals

def extract_positives_negatives(cpgs_path, genes_path, clean_signals, size_negative=10000):
    non_rand_cpgs = pd.read_csv(cpgs_path, sep="\t")
    print(non_rand_cpgs.head())
    all_cols = non_rand_cpgs.columns
    all_cols = [item.strip() for item in all_cols]
    non_rand_cpgs.columns = all_cols
    print(non_rand_cpgs.columns)

    genes_symbols_diff_exp_meth = pd.read_csv(genes_path, sep="\t")
    print(genes_symbols_diff_exp_meth.head())

    positive_probes_genes = list()
    for index, col in enumerate(clean_signals.columns):
        cpg, gene = col.split("_")[0], col.split("_")[1]
        if gene in genes_symbols_diff_exp_meth["Gene symbol"].tolist() or cpg in non_rand_cpgs["CpG"].tolist():
            positive_probes_genes.append(col)
    print(len(positive_probes_genes))

    positive_signals = clean_signals[positive_probes_genes]
    print(positive_signals.head())

    all_features_names = clean_signals.columns.tolist()
    negative_cols = [x for x in all_features_names if x not in positive_probes_genes]
    print(len(all_features_names), len(negative_cols))
    negative_signals = clean_signals[negative_cols]
    print(negative_signals.head())

    #size_negative = 10000 #len(positive_probes_genes)
    balanced_negative_signals = clean_signals[negative_signals.columns[:size_negative]]
    print(balanced_negative_signals.head())

    negative_probes_genes = balanced_negative_signals.columns.tolist()

    df_neg = pd.DataFrame(negative_probes_genes, columns=["negative_probes_genes"])
    df_neg.to_csv("data/output/" + "negative_probes_genes_large.tsv", index=None)
    print(df_neg.head())

    positive_probes_genes = positive_probes_genes
    df_pos = pd.DataFrame(positive_probes_genes, columns=["positive_probes_genes"])
    df_pos.to_csv("data/output/" + "positive_probes_genes.tsv", index=None)
    print(df_pos.head())

    probe_genes_mapping = clean_signals.columns.tolist()
    id_range = np.arange(0, len(probe_genes_mapping))
    df_probe_genes_mapping = pd.DataFrame(zip(probe_genes_mapping, id_range), columns=["name", "id"])
    print(df_probe_genes_mapping.head())

    df_probe_genes_mapping.to_csv("data/output/" + "probe_genes_mapping_id.tsv", sep="\t", index=None)

    balanced_negative_signals.to_csv("data/output/" + "balanced_negative_signals.tsv", sep="\t", index=None)
    positive_signals.to_csv("data/output/" + "positive_signals.tsv", sep="\t", index=None)

    combined_pos_neg_signals = pd.concat([positive_signals, balanced_negative_signals], axis=1)
    print(combined_pos_neg_signals.head())

    transposed_matrix = np.transpose(combined_pos_neg_signals)
    print(transposed_matrix.head())

    return transposed_matrix

def compute_correlation(extracted_features, size_negative=10000, relation_threshold=0.5):
    correlation_matrix = np.corrcoef(extracted_features)
    print(correlation_matrix, correlation_matrix.shape)

    df_corr = pd.DataFrame(correlation_matrix)
    df_corr.index = extracted_features.index
    df_corr.columns = extracted_features.index
    print(df_corr.head())

    #relation_threshold = 0.5
    probe_gene_names = df_corr.columns
    corr_mat = df_corr
    corr_mat_filtered = corr_mat > relation_threshold
    print(corr_mat_filtered.head())

    in_probe_relation = list()
    out_probe_relation = list()
    print("Creating probe-gene relations...")
    for index, col in enumerate(corr_mat_filtered.columns):
        for item_idx, item in enumerate(corr_mat_filtered[col]):
            if item == True and col != probe_gene_names[item_idx]:
                in_probe_relation.append(col)
                out_probe_relation.append(probe_gene_names[item_idx])
    df_significant_gene_relation = pd.DataFrame(zip(in_probe_relation, out_probe_relation), columns=["In", "Out"])
    print(df_significant_gene_relation.head())

    file_name = "data/output/" + "significant_gene_relation_{}_{}.tsv".format("only_positive_corr", size_negative)
    df_significant_gene_relation.to_csv(file_name, sep="\t", header=None, index=False)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("-ca", "--clean_arrays_path", required=True, help="clean Illumina arrays path")
    arg_parser.add_argument("-de", "--non_rand_dem", required=True, help="demethylated CpGs")
    arg_parser.add_argument("-diex", "--diff_exp", required=True, help="differentially expressed genes")

    args = vars(arg_parser.parse_args())

    clean_arrays_path = args["clean_arrays_path"]
    non_rand_dem_cpgs = args["non_rand_dem"]
    diff_exp_genes = args["diff_exp"]

    clean_arrays = load_clean_arrays(clean_arrays_path)
    features = extract_positives_negatives(non_rand_dem_cpgs, diff_exp_genes, clean_arrays)
    compute_correlation(features)