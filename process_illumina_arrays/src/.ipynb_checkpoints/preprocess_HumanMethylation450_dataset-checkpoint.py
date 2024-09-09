#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script preprocesses the Human Methylation 450 dataset.
Converted from Jupyter Notebook to Python script.
"""

# Cell 12
import argparse
import os
import sys
import time

import pandas as pd
import numpy as np


def load_illumina_arrays(arrays_path, mapper_path):
    # Cell 13
    #base_path = "../nanodiag_datasets/GSE175758/"
    #GSE175758_GEO_processed = base_path + "GSE175758_GEO_processed.txt"
    df_GSE175758_GEO_processed = pd.read_csv(arrays_path, sep="\t")

    #df_GSE175758_GEO_processed = df_GSE175758_GEO_processed[:10000]

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
    probe_mapper_full = pd.read_csv(mapper_path, sep="\t")
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


def merge_patients(clean_probes):
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
    df_d15_transpose.to_csv("data/output/" + "df_d15.csv" , sep="\t")

    print(merged_do_d8_transpose.head())

    rownames = merged_do_d8_transpose.index.tolist()
    df_row_names = pd.DataFrame(rownames, columns=["PatientIDs"])
    df_row_names.to_csv("data/output/" + "final_patient_names.csv", sep="\t", index=None)
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
    df_patiendIds_demethylation_full_names.to_csv("data/output/" + "patiendIds_demethylation.csv", sep="\t", index=None)
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
    df_patiendIds_rna_expr_full_names.to_csv("data/output/" + "patiendIds_rna_expr.csv", sep="\t", index=None)
    print(df_patiendIds_rna_expr_full_names)

    # Cell 101
    dnam_patients = df_patiendIds_demethylation_full_names["PatientIDs_Demethylation"].tolist()
    rna_expr_patients = df_patiendIds_rna_expr_full_names["PatientIDs_RNA_Expr"].tolist()

    commmon_patients = list(set(rna_expr_patients).intersection(set(dnam_patients)))
    commmon_patients = sorted(commmon_patients, reverse=False)
    sorted_tuples = sorted([(s, s[-1]) for s in commmon_patients], key=lambda x: x[1])
    commmon_patients = list(map(lambda x: x[0], sorted_tuples))

    df_commmon_patients = pd.DataFrame(commmon_patients, columns=["final_patient_names"])
    df_commmon_patients.to_csv("data/output/" + "final_dnam_rna_patients.csv", sep="\t")
    print(df_commmon_patients.head())

    # Cell 98
    signals_do_d8_res_no_res = merged_do_d8_transpose
    signals_do_d8_res_no_res = signals_do_d8_res_no_res.loc[commmon_patients]
    print(signals_do_d8_res_no_res.head())

    df_reset_signals_do_d8_res_no_res = signals_do_d8_res_no_res.reset_index(drop=True)
    print(df_reset_signals_do_d8_res_no_res)

    df_reset_signals_do_d8_res_no_res.to_csv("data/output/" + "merged_signals.csv", sep="\t", index=None)
    print(df_reset_signals_do_d8_res_no_res.head())



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("-ap", "--arrays_path", required=True, help="Illumina arrays path")
    arg_parser.add_argument("-mp", "--mapper_path", required=True, help="Illumina arrays mapper path")
    arg_parser.add_argument("-snp", "--snp_probes_path", required=True, help="snp probes path")

    args = vars(arg_parser.parse_args())

    arrays_path = args["arrays_path"]
    mapper_path = args["mapper_path"]
    snp_probes_path = args["snp_probes_path"]

    processed_arrays, probe_mapper_full = load_illumina_arrays(arrays_path, mapper_path)
    clean_probes = filter_probes(snp_probes_path, processed_arrays, probe_mapper_full)
    merge_patients(clean_probes)
    
