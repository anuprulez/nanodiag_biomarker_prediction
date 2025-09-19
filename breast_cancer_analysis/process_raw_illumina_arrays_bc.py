
import argparse
import os
import sys
import time
import random

import pandas as pd
import numpy as np
import scanpy as sc


i_base_path = "data_19_Sept_25/inputs/"
o_base_path = "data_19_Sept_25/outputs/"

def process_arrays(arrays_path, mapper_path, snp_probes_path, platform_path, seeds_path):
    
    print("Loading Illumina arrays...")
    df_GSE237036_matrix = pd.read_csv(arrays_path, sep="\t")

    #df_GSE175758_GEO_processed = df_GSE175758_GEO_processed[:10000]

    print(df_GSE237036_matrix.head())

    # Cell 16
    column_names = df_GSE237036_matrix.columns
    print(column_names)


    # load mapper
    #probe_mapper_full = pd.read_csv(base_path + "GPL13534-11288-mapper-HMBC450.txt", sep="\t")
    #probe_mapper_full
    print("Loading gene mapper...")
    probe_mapper = pd.read_csv(mapper_path, sep="\t")
    print(probe_mapper.head())

    # Cell 19
    # filter probes
    ## https://github.com/mwsill/mnp_training/blob/master/preprocessing.R
    ### https://github.com/lizhiqi49/SAGCN/tree/main/code/preprocessing/filter
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8235477/


    # Cell 21
    probe_mapper = probe_mapper.dropna()
    print(probe_mapper.head())

    # Cell 22
    expanded_probe_ids = []
    gene_names = []
    gene_group_names = []
    for i, row in probe_mapper.iterrows():
        genes = row["UCSC_RefGene_Name"].split(";")
        probe_ids = np.repeat(row["probeID"], len(genes))
        expanded_probe_ids.extend(probe_ids)
        gene_names.extend(genes)

    df_probe_gene_mapper = pd.DataFrame(zip(expanded_probe_ids, gene_names), columns=["ID_REF", "Genes"])

    # Cell 23
    df_probe_gene_mapper = df_probe_gene_mapper.dropna()
    df_probe_gene_mapper = df_probe_gene_mapper.drop_duplicates(["ID_REF", "Genes"])
    df_probe_gene_mapper = df_probe_gene_mapper.reset_index(drop=True)
    print(df_probe_gene_mapper.head())

    print("Mapping genes and probes...")
    df_probe_signals_merged = pd.merge(df_probe_gene_mapper, df_GSE237036_matrix, how="inner", on=["ID_REF"])
    print(df_probe_signals_merged.head())

    df_platform = pd.read_csv(platform_path, sep=",")
    df_platform_CHR = df_platform[["ID", "CHR"]]
    df_platform_CHR_X_Y = df_platform_CHR[df_platform_CHR["CHR"].isin(["X","Y"])]
    df_platform_CHR_X_Y_IDS = df_platform_CHR_X_Y["ID"].tolist()

    df_probe_signals_merged = pd.merge(df_probe_gene_mapper, df_GSE237036_matrix, how="inner", on=["ID_REF"])

    print("Filtering SNPs...")
    snp_probes = pd.read_csv(snp_probes_path, sep=",", header=None)
    print(len(snp_probes[0].tolist()))
    probes_snp = snp_probes[0].tolist()
    print(len(probes_snp))
    excluded_probes = probes_snp
    excluded_probes.extend(df_platform_CHR_X_Y_IDS)
    print(len(excluded_probes))

    df_probe_signals_merged_no_snp_x_y = df_probe_signals_merged[~df_probe_signals_merged["ID_REF"].isin(excluded_probes)]

    info_cols = df_probe_signals_merged_no_snp_x_y[["ID_REF", "Genes"]]

    feature_names = df_probe_signals_merged_no_snp_x_y["ID_REF"] + "_" + df_probe_signals_merged_no_snp_x_y["Genes"]
    feature_names_list = feature_names.tolist()
    print(len(feature_names_list))

    df_probe_signals_merged_no_snp_x_y_data = df_probe_signals_merged_no_snp_x_y.iloc[:, 2:]
    df_probe_signals_merged_no_snp_x_y_data = df_probe_signals_merged_no_snp_x_y_data.reset_index(drop=True)

    df_probe_signals_merged_no_snp_x_y_data_T = df_probe_signals_merged_no_snp_x_y_data.transpose()

    df_probe_signals_merged_no_snp_x_y_data_T.columns = feature_names_list

    print("Saving merged signals...")
    df_probe_signals_merged_no_snp_x_y_data_T.to_csv(o_base_path + "merged_signals_bc.csv", sep="\t", index=None)

    df_merged_signals = df_probe_signals_merged_no_snp_x_y_data_T

    df_seeds = pd.read_csv(seeds_path, sep="\t")

    df_seeds = df_seeds[df_seeds["P.Value"] <= 0.01]
    #df_seeds = df_seeds[df_seeds["deltaBeta"].abs() >= 0.08]
    df_seeds = df_seeds[df_seeds["CHR"] != "X"]

    # find POS probe ids from original matrix using gene symbols and CpGs mentioned in https://www.nature.com/articles/s41375-023-01876-2
    positive_probes_genes = list()
    all_seed_probes = df_seeds["probeID"].tolist()
    for index, col in enumerate(df_merged_signals.columns):
        cpg, gene = col.split("_")[0], col.split("_")[1]
        #if gene in genes_symbols_diff_exp_meth["Gene symbol"].tolist() or 
        if cpg in all_seed_probes:
            positive_probes_genes.append(col)
    print(len(positive_probes_genes))

    positive_signals = df_merged_signals[positive_probes_genes]

    all_features_names = df_merged_signals.columns.tolist()
    negative_cols = [x for x in all_features_names if x not in positive_probes_genes]
    print(len(all_features_names), len(negative_cols))
    negative_signals = df_merged_signals[negative_cols]

    col_names = list(negative_signals.columns)
    print(col_names[:5])
    random.shuffle(col_names)
    print(col_names[:5])

    size_negative = 10000

    gene_expression_data = negative_signals #pd.read_csv('gene_expression_data.csv', index_col=0)

    # Convert the DataFrame to an AnnData object for Scanpy processing
    adata = sc.AnnData(gene_expression_data)

    # Calculate highly variable genes using Scanpy's built-in function
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=size_negative)

    # Extract the highly variable genes
    hvg_df = adata.var[adata.var['highly_variable']]

    print("Highly variable genes selected:")

    print(hvg_df.head())

    negative_col_names = list(hvg_df.index)

    balanced_negative_signals = df_merged_signals[negative_col_names]

    negative_probes_genes = balanced_negative_signals.columns.tolist()
    df_neg = pd.DataFrame(negative_probes_genes, columns=["negative_probes_genes"])
    df_neg.to_csv(o_base_path + "negative_probes_genes_large.tsv", index=None)
    print("Neg genes selected:")
    print(df_neg.head())

    df_pos = pd.DataFrame(positive_probes_genes, columns=["positive_probes_genes"])
    df_pos.to_csv(o_base_path + "positive_probes_genes.tsv", index=None)

    probe_genes_mapping = df_merged_signals.columns.tolist()

    balanced_negative_signals.to_csv(o_base_path + "balanced_negative_signals.tsv", sep="\t", index=None)
    positive_signals.to_csv(o_base_path + "positive_signals.tsv", sep="\t", index=None)

    combined_pos_neg_signals = pd.concat([positive_signals, balanced_negative_signals], axis=1)

    transposed_matrix = np.transpose(combined_pos_neg_signals)

    combined_pos_neg_signals.to_csv(o_base_path + "combined_pos_neg_signals_bc.csv", sep="\t", index=None)

    correlation_matrix = np.corrcoef(transposed_matrix)

    df_corr = pd.DataFrame(correlation_matrix)
    df_corr.index = transposed_matrix.index
    df_corr.columns = transposed_matrix.index

    relation_threshold = 0.33
    probe_gene_names = df_corr.columns
    corr_mat = df_corr
    corr_mat_filtered = corr_mat > relation_threshold

    in_probe_relation = list()
    out_probe_relation = list()
    print("Creating probe-gene relations...")
    for index, col in enumerate(corr_mat_filtered.columns):
        for item_idx, item in enumerate(corr_mat_filtered[col]):
            if item == True and col != probe_gene_names[item_idx]:
                in_probe_relation.append(col)
                out_probe_relation.append(probe_gene_names[item_idx])
    df_significant_gene_relation = pd.DataFrame(zip(in_probe_relation, out_probe_relation), columns=["In", "Out"])
    print("Gene relations PPI")
    print(df_significant_gene_relation.head())

    file_name = o_base_path + "significant_gene_relation_{}.tsv".format("breast_cancer", size_negative)
    df_significant_gene_relation.to_csv(file_name, sep="\t", header=None, index=False)

    probe_id = list()
    probe_importance = list()
    for i, item in df_pos.iterrows():
        values = item.values
        probe_name = values[0].split("_")[0]
        match = df_seeds[df_seeds["probeID"] == probe_name]
        if (len(match)) > 0:
            probe_id.append(values[0])
            probe_importance.append(np.abs(match["deltaBeta"].values[0]))
    df_seed_features = pd.DataFrame(zip(probe_id, probe_importance))
    print("Creating seed features")
    print(df_seed_features.head())

    df_seed_features.to_csv(o_base_path + "seed_features_bc.csv", sep="\t", index=None, header=None)
    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    #arg_parser.add_argument("-ap", "--arrays_path", required=True, help="Illumina arrays path")
    #arg_parser.add_argument("-mp", "--mapper_path", required=True, help="Illumina arrays mapper path")
    #arg_parser.add_argument("-snp", "--snp_probes_path", required=True, help="snp probes path")
    #arg_parser.add_argument("-pp", "--platform_path", required=True, help="platform path")
    #arg_parser.add_argument("-sp", "--seeds_path", required=True, help="seeds path")

    #args = vars(arg_parser.parse_args())

    arrays_path = i_base_path + "GSE237036_matrix_processed.txt"
    mapper_path = i_base_path + "probe_genes.csv"
    snp_probes_path = i_base_path + "snp_7998probes.vh20151030.txt"
    platform_path = i_base_path + "GPL21145-48548.txt"
    seeds_path = i_base_path + "seed_probes.csv"

    process_arrays(arrays_path, mapper_path, snp_probes_path, platform_path, seeds_path)
    