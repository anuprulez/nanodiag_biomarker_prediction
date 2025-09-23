#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script preprocesses the Human Methylation 450 dataset.
Converted from Jupyter Notebook to Python script.
"""

# Cell 12
import argparse
import os
import time
import sys
import subprocess

import pandas as pd
import numpy as np

from omegaconf.omegaconf import OmegaConf

def load_illumina_arrays(config):
    # Cell 13
    #base_path = "../nanodiag_datasets/GSE175758/"
    #GSE175758_GEO_processed = base_path + "GSE175758_GEO_processed.txt"
    # config.p_arrays, config.p_mapper
    df_GSE175758_GEO_processed = pd.read_csv(config.p_arrays, sep="\t")

    df_GSE175758_GEO_processed = df_GSE175758_GEO_processed[:10000]

    print(df_GSE175758_GEO_processed.head())

    # Cell 16
    column_names = df_GSE175758_GEO_processed.columns
    print(column_names)

    # Cell 17
    cutoff_pval = 5e-2
    for i in range(len(column_names)):
        if i > 0 and i % 2 == 0:
            print(column_names[i])
            col_name = column_names[i]
            df_GSE175758_GEO_processed[column_names[i-1]] = df_GSE175758_GEO_processed.apply(lambda x: x[column_names[i-1]] if x[col_name] < cutoff_pval else 0.0, axis=1)
    print(df_GSE175758_GEO_processed.head())


    # load mapper
    #probe_mapper_full = pd.read_csv(base_path + "GPL13534-11288-mapper-HMBC450.txt", sep="\t")
    #probe_mapper_full
    probe_mapper_full = pd.read_csv(config.p_mapper, sep="\t")
    print(probe_mapper_full.head())

    # Cell 19
    # filter probes
    ## https://github.com/mwsill/mnp_training/blob/master/preprocessing.R
    ### https://github.com/lizhiqi49/SAGCN/tree/main/code/preprocessing/filter
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8235477/


    # Cell 21
    probe_mapper = probe_mapper_full[["ID", "UCSC_RefGene_Name", "UCSC_RefGene_Group"]]
    probe_mapper = probe_mapper.dropna()
    print(probe_mapper)

    # Cell 22
    expanded_probe_ids = []
    gene_names = []
    gene_group_names = []
    for i, row in probe_mapper.iterrows():
        genes = row["UCSC_RefGene_Name"].split(";")
        genes_groups = row["UCSC_RefGene_Group"].split(";")
        probe_ids = np.repeat(row["ID"], len(genes))
        expanded_probe_ids.extend(probe_ids)
        gene_names.extend(genes)
        gene_group_names.extend(genes_groups)

    df_probe_gene_mapper = pd.DataFrame(zip(expanded_probe_ids, gene_names, gene_group_names), columns=["ID_REF", "Genes", "Gene_Groups"])
    print(df_probe_gene_mapper.head())

    # Cell 23
    df_probe_gene_mapper_uniq = df_probe_gene_mapper.drop_duplicates(["ID_REF", "Genes", "Gene_Groups"])
    print(df_probe_gene_mapper_uniq.head())

    # Cell 24
    # filter probe 

    # Cell 25
    #df_GSE175758_GEO_processed[df_GSE175758_GEO_processed["ID_REF"] == "cg00035864"]

    # Cell 26
    print(df_GSE175758_GEO_processed.head())

    # Cell 27
    #df_GSE175758_GEO_processed[df_GSE175758_GEO_processed["ID_REF"] == "ch.22.909671F"]

    # Cell 28
    df_probe_signals_merged = pd.merge(df_probe_gene_mapper_uniq, df_GSE175758_GEO_processed, how="inner", on=["ID_REF"])
    print(df_probe_signals_merged.head())

    # Cell 30
    #df_probe_signals_merged[df_probe_signals_merged["Genes"] == "RIC3"]
    return df_probe_signals_merged, probe_mapper_full
    

def filter_probes(snp_probes_path, processed_arrays, probe_mapper_full):
    ### Filter probes based on X,Y CHR and SNP probes

    ## probes_snp, probe_x_y
    # list of probes for SNP
    #snp_probes = pd.read_csv(base_path + "snp_7998probes.vh20151030.txt", sep=",", header=None)
    snp_probes = pd.read_csv(snp_probes_path, sep=",", header=None)
    len(snp_probes[0].tolist())
    probes_snp = snp_probes[0].tolist()

    # Cell 51
    # list of probes for CHR X and Y
    probe_mapper_x_y = probe_mapper_full[probe_mapper_full["Chromosome_36"].isin(["X", "Y"])]
    probe_x_y = probe_mapper_x_y["ID"].tolist()
    len(probe_x_y)
    
    excluded_probes = probes_snp
    excluded_probes.extend(probe_x_y)
    print(len(excluded_probes))

    df_probe_signals_merged_no_snp_x_y = processed_arrays[~processed_arrays["ID_REF"].isin(excluded_probes)]
    df_probe_signals_merged_no_snp_x_y

    # Cell 47
    df_probe_signals_merged_no_snp_x_y[df_probe_signals_merged_no_snp_x_y["ID_REF"] == "cg00050873"]

    return df_probe_signals_merged_no_snp_x_y


def merge_patients(clean_probes, config):
    # Cell 52
    d15_features = list()
    d8_features = list()
    d0_features = list()

    for col in clean_probes.columns:
        if "c1.blasts.d0" in col and "Detection.Pval" not in col:
            d0_features.append(col)
        if "c1.blasts.d8" in col and "Detection.Pval" not in col:
            d8_features.append(col)
        if "c1.blasts.d15" in col and "Detection.Pval" not in col:
            d15_features.append(col)
    print(len(d0_features), len(d8_features), len(d15_features))

    # Cell 53
    df_d0 = clean_probes[d0_features]
    print(df_d0.head())

    # Cell 54
    df_d8 = clean_probes[d8_features]
    print(df_d8.head())

    # Cell 55
    df_d15 = clean_probes[d15_features]
    print(df_d15.head())

    # Cell 36
    #df_d0.columns, df_d8.columns, df_d15.columns

    # Cell 58
    merged_do_d8 = pd.concat([df_d0, df_d8], axis=1)
    print(merged_do_d8.head())

    # Cell 60
    merged_do_d8_transpose = merged_do_d8.transpose()
    print(merged_do_d8_transpose.head())

    # Cell 61
    info_cols = clean_probes[["ID_REF", "Genes", "Gene_Groups"]]
    print(info_cols)

    # Cell 64
    feature_names = clean_probes["ID_REF"] + "_" + clean_probes["Genes"]
    feature_names_list = feature_names.tolist()
    print(len(feature_names_list))

    # Cell 65
    print(len(merged_do_d8_transpose.columns))
    merged_do_d8_transpose.columns = feature_names_list
    print(merged_do_d8_transpose.head())

    # Cell 67
    # Merge D15 data

    df_d15_transpose = df_d15.transpose()
    df_d15_transpose.columns = feature_names_list
    print(df_d15_transpose.head())

    # Cell 69
    df_d15_transpose.to_csv(config.p_base + "df_d15.csv" , sep="\t")

    print(merged_do_d8_transpose.head())

    rownames = merged_do_d8_transpose.index.tolist()
    df_row_names = pd.DataFrame(rownames, columns=["PatientIDs"])
    df_row_names.to_csv(config.p_base + "final_patient_names.csv", sep="\t", index=None)
    print(df_row_names)

    # Cell 76
    # PatientIDS for Demethylation
    # https://www.nature.com/articles/s41375-023-01876-2/figures/1

    patiendIds_demethylation = ["S16005", "S01033", "S01013", \
                            "S01015", "S01016", "S1007", "S03005", \
                            "S26004", "S01039", "S01027", "S01004", \
                            "S01012", "S1888", "S14008", "S31001", "S01010", \
                            "S01025", "S01006", "S01999", "S25005", "S14007" \
                            "S14001", "S01001", "S01032", "S26002", "S04012", \
                            "S11011", "S01777" ]

    patiendIds_demethylation_full_names = list()
    for item in df_row_names["PatientIDs"].tolist():
        if item.split(".")[0] in patiendIds_demethylation:
            patiendIds_demethylation_full_names.append(item)

    df_patiendIds_demethylation_full_names = pd.DataFrame(patiendIds_demethylation_full_names, columns=["PatientIDs_Demethylation"])
    df_patiendIds_demethylation_full_names.to_csv(config.p_base + "patiendIds_demethylation.csv", sep="\t", index=None)
    print(df_patiendIds_demethylation_full_names)

    # Cell 85
    # PatientIDS for RNA expression
    # https://www.nature.com/articles/s41375-023-01876-2/figures/5

    patiendIds_rna_expr = ["S26004", "S14001", "S03005", "S26002", "S01006", \
                       "S01032", "S01004", "S16005", "S14007", \
                       "S01007", "S01016", "S11011", "S01033", "S01999", \
                       "S01038", "S31001", "S14008", "S01015", "S01888", \
                       "S04012", "S01777", "S01030", "S01039"
                      ]

    patiendIds_rna_expr_full_names = list()

    for item in df_row_names["PatientIDs"].tolist():
        if item.split(".")[0] in patiendIds_rna_expr:
            patiendIds_rna_expr_full_names.append(item)

    df_patiendIds_rna_expr_full_names = pd.DataFrame(patiendIds_rna_expr_full_names, columns=["PatientIDs_RNA_Expr"])
    df_patiendIds_rna_expr_full_names.to_csv(config.p_base + "patiendIds_rna_expr.csv", sep="\t", index=None)
    print(df_patiendIds_rna_expr_full_names)

    # Cell 101
    dnam_patients = df_patiendIds_demethylation_full_names["PatientIDs_Demethylation"].tolist()
    rna_expr_patients = df_patiendIds_rna_expr_full_names["PatientIDs_RNA_Expr"].tolist()

    commmon_patients = list(set(rna_expr_patients).intersection(set(dnam_patients)))
    commmon_patients = sorted(commmon_patients, reverse=False)
    sorted_tuples = sorted([(s, s[-1]) for s in commmon_patients], key=lambda x: x[1])
    commmon_patients = list(map(lambda x: x[0], sorted_tuples))

    df_commmon_patients = pd.DataFrame(commmon_patients, columns=["final_patient_names"])
    df_commmon_patients.to_csv(config.p_base + "final_dnam_rna_patients.csv", sep="\t")
    print(df_commmon_patients.head())

    # Cell 98
    signals_do_d8_res_no_res = merged_do_d8_transpose
    signals_do_d8_res_no_res = signals_do_d8_res_no_res.loc[commmon_patients]
    print(signals_do_d8_res_no_res.head())

    df_reset_signals_do_d8_res_no_res = signals_do_d8_res_no_res.reset_index(drop=True)
    print(df_reset_signals_do_d8_res_no_res)

    df_reset_signals_do_d8_res_no_res.to_csv(config.p_merged_signals, sep="\t", index=None)
    print(df_reset_signals_do_d8_res_no_res.head())


def load_clean_arrays(config):
    df_merged_signals = pd.read_csv(config.p_merged_signals, sep="\t")
    print(df_merged_signals.head())
    
    features = df_merged_signals
    feature_names = features.columns.tolist()

    df_feature_names = pd.DataFrame(feature_names, columns=["feature_names"])
    df_feature_names.to_csv(config.p_base + "final_feature_names.tsv", index=None, sep="\t")
    return df_merged_signals


def extract_positives_negatives(clean_signals, config): 
    # cpgs_path, genes_path, config.p_seeds_methylated_cpgs, config.p_seeds_gene_methylation_expr
    non_rand_cpgs = pd.read_csv(config.p_seeds_methylated_cpgs, sep="\t")
    print(non_rand_cpgs.head())
    all_cols = non_rand_cpgs.columns
    all_cols = [item.strip() for item in all_cols]
    non_rand_cpgs.columns = all_cols
    print(non_rand_cpgs.columns)

    genes_symbols_diff_exp_meth = pd.read_csv(config.p_seeds_gene_methylation_expr, sep="\t")
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
    balanced_negative_signals = clean_signals[negative_signals.columns[:config.size_negative]]
    print(balanced_negative_signals.head())

    negative_probes_genes = balanced_negative_signals.columns.tolist()

    df_neg = pd.DataFrame(negative_probes_genes, columns=["negative_probes_genes"])
    df_neg.to_csv(config.p_base + "negative_probes_genes_large.tsv", index=None)
    print(df_neg.head())

    positive_probes_genes = positive_probes_genes
    df_pos = pd.DataFrame(positive_probes_genes, columns=["positive_probes_genes"])
    df_pos.to_csv(config.p_base + "positive_probes_genes.tsv", index=None)
    print(df_pos.head())

    probe_genes_mapping = clean_signals.columns.tolist()
    id_range = np.arange(0, len(probe_genes_mapping))
    df_probe_genes_mapping = pd.DataFrame(zip(probe_genes_mapping, id_range), columns=["name", "id"])
    print(df_probe_genes_mapping.head())

    df_probe_genes_mapping.to_csv(config.p_base + "probe_genes_mapping_id.tsv", sep="\t", index=None)

    balanced_negative_signals.to_csv(config.p_base + "balanced_negative_signals.tsv", sep="\t", index=None)
    positive_signals.to_csv(config.p_base + "positive_signals.tsv", sep="\t", index=None)

    combined_pos_neg_signals = pd.concat([positive_signals, balanced_negative_signals], axis=1)
    combined_pos_neg_signals.to_csv(config.p_combined_pos_neg_signals, sep="\t", index=False)
    print(combined_pos_neg_signals.head())

    transposed_matrix = np.transpose(combined_pos_neg_signals)
    print(transposed_matrix.head())

    return transposed_matrix


def compute_correlation(extracted_features, config):
    correlation_matrix = np.corrcoef(extracted_features)
    print(correlation_matrix, correlation_matrix.shape)

    df_corr = pd.DataFrame(correlation_matrix)
    df_corr.index = extracted_features.index
    df_corr.columns = extracted_features.index
    print(df_corr.head())

    #relation_threshold = 0.5
    probe_gene_names = df_corr.columns
    corr_mat = df_corr
    corr_mat_filtered = corr_mat > config.corr_threshold
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

    #file_name = config.p_base + "significant_gene_relation_{}_{}.tsv".format("only_positive_corr", config.size_negative)
    df_significant_gene_relation = df_significant_gene_relation[:10000]
    df_significant_gene_relation.to_csv(config.p_significant_edges, sep="\t", header=None, index=False)


def create_network_gene_ids(ppi_path, links_path):
    gene = {}
    ngene = 0
    mat = {}
    with open(ppi_path, 'r') as flink, open(links_path, 'w') as fout_link:
        for line in flink:
            node1, node2 = line.strip().split()
            if node1 != node2:
                if node1 not in gene:
                    gene[node1] = ngene
                    ngene += 1
                if node2 not in gene:
                    gene[node2] = ngene
                    ngene += 1
            
                id1 = gene[node1]
                id2 = gene[node2]
            
                if (id1, id2) not in mat and (id2, id1) not in mat:
                    mat[(id1, id2)] = mat[(id2, id1)] = 1
                    fout_link.write(f"{id1} {id2}\n")
    return gene


def mark_seed_genes(seed_genes_path, genes_path, gene):
    
    score_seed_gene = {}
    max_score = 0
    nseedgene = 0
    notfoundseedgene = 0

    with open(seed_genes_path, 'r') as fin:
        for line in fin:
            name_gene, score = line.strip().split()
            score = float(score)
            if name_gene in gene:
                score_seed_gene[name_gene] = score
                nseedgene += 1
            else:
                print(f"Error, not found seed gene {name_gene}")
                notfoundseedgene += 1
            if score > max_score:
                max_score = score

    print(f"{notfoundseedgene} seed genes not found")
    print(f"{nseedgene} seed genes present")
    print(f"Maximum score {max_score}")
    
    #out_gene = "data/output/out_gene"
    with open(genes_path, 'w') as fout_gene:
        for name_gene, gene_id in gene.items():
            if name_gene in score_seed_gene:
                adapt_score = max_score - score_seed_gene[name_gene]
                fout_gene.write(f"{gene_id} {name_gene} {score_seed_gene[name_gene]}\n")
            else:
                fout_gene.write(f"{gene_id} {name_gene} 0.0\n")


def calculate_features(links_data_path, genes_data_path, nedbit_path):
    # nedbit_features_calculator out_links out_genes nedbit_features
    # nedbit-features-calculator
    
    print("calculating nedbit features ...")
    print(links_data_path, genes_data_path, nedbit_path)
    result = subprocess.run(['nedbit-features-calculator', links_data_path, genes_data_path, nedbit_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


def assign_initial_labels(nedbit_path, header, output_gene_ranking_path, q1=0.05, q2=0.2):
    # apu_label_propagation nedbit_features HEADER_PRESENCE output_gene_ranking 0.05 0.2
    # apu-label-propagation
    print("propagating labels ...")
    result = subprocess.run(['apu-label-propagation', nedbit_path, header, output_gene_ranking_path, q1, q2], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)



if __name__ == "__main__":
    '''arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("-ap", "--arrays_path", required=True, help="Illumina arrays path")
    arg_parser.add_argument("-mp", "--mapper_path", required=True, help="Illumina arrays mapper path")
    arg_parser.add_argument("-snp", "--snp_probes_path", required=True, help="snp probes path")

    args = vars(arg_parser.parse_args())

    arrays_path = args["arrays_path"]
    mapper_path = args["mapper_path"]
    snp_probes_path = args["snp_probes_path"]'''

    config = OmegaConf.load("../config/config.yaml")

    ## Step 1
    print("======== Step 1: loading datasets =============")
    processed_arrays, probe_mapper_full = load_illumina_arrays(config)
    clean_probes = filter_probes(config.p_snp_probes, processed_arrays, probe_mapper_full)
    merge_patients(clean_probes, config)

    ## Step 2
    print("======== Step 2: cleaning datasets and computing correlation =============")
    clean_arrays = load_clean_arrays(config)
    features = extract_positives_negatives(clean_arrays, config)
    compute_correlation(features, config)

    ## Step 3

    print("======== Step 3: Label propagation =============")
    genes = create_network_gene_ids(config.p_significant_edges, config.p_out_links)
    mark_seed_genes(config.p_seed_features, config.p_out_genes, genes)
    calculate_features(config.p_out_links, config.p_out_genes, config.p_nedbit_features)
    assign_initial_labels(config.p_nedbit_features, str(config.nedbit_header), config.p_out_gene_rankings, str(config.quantile_1), str(config.quantile_2))

    # Step 3 python src/assign_pre_labels.py 
    # -ppi ../process_illumina_arrays/data/output/significant_gene_relation_only_positive_corr_10000.tsv 
    # -sg data/input/seed_features.tsv 
    # -ol data/output/out_links.csv 
    # -og data/output/out_genes.csv 
    # -nf data/output/nedbit_features.csv 
    # -nh 1 -gr data/output/out_gene_rankings.csv 
    # -qt1 0.05 -qt2 0.2


    ##########################
    # python src/preprocess_HumanMethylation450_dataset.py 
    # -ap data/input/GSE175758_GEO_processed.txt 
    # -mp data/input/GPL13534-11288-mapper-HMBC450.txt 
    # -snp data/input/snp_7998probes.vh20151030.txt

    ##########################

    '''clean_arrays_path = args["clean_arrays_path"]
    non_rand_dem_cpgs = args["non_rand_dem"]
    diff_exp_genes = args["diff_exp"]

    clean_arrays = load_clean_arrays(clean_arrays_path)
    features = extract_positives_negatives(non_rand_dem_cpgs, diff_exp_genes, clean_arrays)
    compute_correlation(features)'''

    # Step 2python src/create_probe_gene_network.py 
    # -ca data/output/merged_signals.csv 
    # -de data/input/non-randomly-demethylated-CpGs.tsv 
    # -diex data/input/genes_symbol_methylation_expression.tsv

    ##########################

    # Step 3 python src/assign_pre_labels.py 
    # -ppi ../process_illumina_arrays/data/output/significant_gene_relation_only_positive_corr_10000.tsv 
    # -sg data/input/seed_features.tsv 
    # -ol data/output/out_links.csv 
    # -og data/output/out_genes.csv 
    # -nf data/output/nedbit_features.csv 
    # -nh 1 -gr data/output/out_gene_rankings.csv 
    # -qt1 0.05 -qt2 0.2


    


    
