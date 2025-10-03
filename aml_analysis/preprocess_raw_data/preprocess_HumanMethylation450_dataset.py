"""
This script preprocesses the Human Methylation 450 dataset for AML analysis.
It includes loading raw data, filtering probes, merging patient data
"""

import argparse
import os
import time
import sys
import subprocess
import random
import requests
import zipfile

import pandas as pd
import numpy as np
import scanpy as sc
import polars as pl

from omegaconf.omegaconf import OmegaConf


def extract_preprocessed_data(config):
    # Your Zenodo link
    url = config.p_raw_data
    output_dir = config.p_base
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "download.zip")

    # Download ZIP
    print(f"[+] Downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract into output_dir (flatten top-level folder)
    print(f"[+] Extracting into {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            # Get only the filename (ignoring any subfolders in the ZIP)
            filename = os.path.basename(member.filename)
            if not filename:
                continue  # skip directories
            target_path = os.path.join(output_dir, filename)

            # Extract file
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    # Optionally remove the zip after extraction
    os.remove(zip_path)


def load_illumina_arrays(config):
    df_GSE175758_GEO_processed = pd.read_csv(config.p_arrays, sep="\t")
    print("Raw Illumina arrays for AML patients")
    print(df_GSE175758_GEO_processed.head())

    column_names = df_GSE175758_GEO_processed.columns

    print(f"Filtering based on p-value: {config.cutoff_pval}")

    df_filtered = df_GSE175758_GEO_processed.copy()

    # Iterate over all columns ending with ".Detection.Pval"
    for col in column_names:
        if (
            col.endswith(".Detection.Pval")
            and "d15" not in col
            and "T_cells" not in col
        ):
            # Don't process d15 and T_cells columns; Only take d0 and d8 from blasts
            sample = col.replace(".Detection.Pval", "")
            methyl_col = sample
            pval_col = col

            df_filtered[methyl_col] = df_GSE175758_GEO_processed.apply(
                lambda row: row[methyl_col] if row[pval_col] < 0.05 else 0, axis=1
            )

            print(f"Methylation col: {methyl_col}, P.Val col: {pval_col}")

    # Finally drop all p-value columns
    print("Dropping p-value columns and d15, T_cells columns ...")
    df_filtered = df_filtered.drop(
        columns=[c for c in column_names if c.endswith(".Detection.Pval")]
    )

    for c in df_filtered.columns:
        if "d15" in c or "T_cells" in c:
            print("Removing column: ", c)
            df_filtered = df_filtered.drop(columns=[c])

    print(df_filtered.head())

    probe_mapper_full = pd.read_csv(config.p_mapper, sep="\t")
    print("Gene probe mapper")
    print(probe_mapper_full.head())

    ## https://github.com/mwsill/mnp_training/blob/master/preprocessing.R
    ### https://github.com/lizhiqi49/SAGCN/tree/main/code/preprocessing/filter
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8235477/

    probe_mapper = probe_mapper_full[["ID", "UCSC_RefGene_Name", "UCSC_RefGene_Group"]]
    probe_mapper = probe_mapper.dropna()
    print(probe_mapper)

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

    df_probe_gene_mapper = pd.DataFrame(
        zip(expanded_probe_ids, gene_names, gene_group_names),
        columns=["ID_REF", "Genes", "Gene_Groups"],
    )
    print(df_probe_gene_mapper.head())

    df_probe_gene_mapper_uniq = df_probe_gene_mapper.drop_duplicates(
        ["ID_REF", "Genes", "Gene_Groups"]
    )
    print("Probe gene mapper uniq: %d", len(df_probe_gene_mapper_uniq))
    print(df_probe_gene_mapper_uniq.head())

    print(df_filtered.head())

    df_probe_signals_merged = pd.merge(
        df_probe_gene_mapper_uniq, df_filtered, how="inner", on=["ID_REF"]
    )
    print(df_probe_signals_merged.head())

    return df_probe_signals_merged, probe_mapper_full


def filter_probes(snp_probes_path, processed_arrays, probe_mapper_full):
    ### Filter probes based on X,Y CHR and SNP probes
    # list of probes for SNP
    snp_probes = pd.read_csv(snp_probes_path, sep=",", header=None)
    len(snp_probes[0].tolist())
    probes_snp = snp_probes[0].tolist()

    # list of probes for CHR X and Y
    probe_mapper_x_y = probe_mapper_full[
        probe_mapper_full["Chromosome_36"].isin(["X", "Y"])
    ]
    probe_x_y = probe_mapper_x_y["ID"].tolist()
    len(probe_x_y)

    excluded_probes = probes_snp
    excluded_probes.extend(probe_x_y)
    print(len(excluded_probes))

    df_probe_signals_merged_no_snp_x_y = processed_arrays[
        ~processed_arrays["ID_REF"].isin(excluded_probes)
    ]
    df_probe_signals_merged_no_snp_x_y

    return df_probe_signals_merged_no_snp_x_y


def merge_patients(clean_probes, config):
    print(f"Checking if T_cells in clean_probes...")
    if any("T_cells" in col for col in clean_probes.columns):
        print("T_cells found, removing T_cells columns ...")
        clean_probes = clean_probes[
            [col for col in clean_probes.columns if "T_cells" not in col]
        ]

    merged_do_d8 = clean_probes
    print(merged_do_d8.head())

    merged_do_d8_transpose = merged_do_d8.transpose()
    print(merged_do_d8_transpose.head())

    info_cols = clean_probes[["ID_REF", "Genes", "Gene_Groups"]]
    print(info_cols)

    feature_names = clean_probes["ID_REF"] + "_" + clean_probes["Genes"]
    feature_names_list = feature_names.tolist()
    print(len(feature_names_list))

    print(len(merged_do_d8_transpose.columns))
    merged_do_d8_transpose.columns = feature_names_list
    print(merged_do_d8_transpose.head())

    rownames = merged_do_d8_transpose.index.tolist()
    df_row_names = pd.DataFrame(rownames, columns=["PatientIDs"])
    df_row_names.to_csv(config.p_base + "final_patient_names.csv", sep="\t", index=None)
    print(df_row_names)

    # PatientIDS for Demethylation
    # https://www.nature.com/articles/s41375-023-01876-2/figures/1

    patiendIds_demethylation = [
        "S16005",
        "S01033",
        "S01013",
        "S01015",
        "S01016",
        "S1007",
        "S03005",
        "S26004",
        "S01039",
        "S01027",
        "S01004",
        "S01012",
        "S1888",
        "S14008",
        "S31001",
        "S01010",
        "S01025",
        "S01006",
        "S01999",
        "S25005",
        "S14007S14001",
        "S01001",
        "S01032",
        "S26002",
        "S04012",
        "S11011",
        "S01777",
    ]

    patiendIds_demethylation_full_names = list()
    for item in df_row_names["PatientIDs"].tolist():
        if item.split(".")[0] in patiendIds_demethylation:
            patiendIds_demethylation_full_names.append(item)

    df_patiendIds_demethylation_full_names = pd.DataFrame(
        patiendIds_demethylation_full_names, columns=["PatientIDs_Demethylation"]
    )
    df_patiendIds_demethylation_full_names.to_csv(
        config.p_base + "patiendIds_demethylation.csv", sep="\t", index=None
    )
    print(df_patiendIds_demethylation_full_names)

    # PatientIDS for RNA expression
    # https://www.nature.com/articles/s41375-023-01876-2/figures/5

    patiendIds_rna_expr = [
        "S26004",
        "S14001",
        "S03005",
        "S26002",
        "S01006",
        "S01032",
        "S01004",
        "S16005",
        "S14007",
        "S01007",
        "S01016",
        "S11011",
        "S01033",
        "S01999",
        "S01038",
        "S31001",
        "S14008",
        "S01015",
        "S01888",
        "S04012",
        "S01777",
        "S01030",
        "S01039",
    ]

    patiendIds_rna_expr_full_names = list()

    for item in df_row_names["PatientIDs"].tolist():
        if item.split(".")[0] in patiendIds_rna_expr:
            patiendIds_rna_expr_full_names.append(item)

    df_patiendIds_rna_expr_full_names = pd.DataFrame(
        patiendIds_rna_expr_full_names, columns=["PatientIDs_RNA_Expr"]
    )
    df_patiendIds_rna_expr_full_names.to_csv(
        config.p_base + "patiendIds_rna_expr.csv", sep="\t", index=None
    )
    print(df_patiendIds_rna_expr_full_names)

    dnam_patients = df_patiendIds_demethylation_full_names[
        "PatientIDs_Demethylation"
    ].tolist()
    rna_expr_patients = df_patiendIds_rna_expr_full_names[
        "PatientIDs_RNA_Expr"
    ].tolist()

    commmon_patients = list(set(rna_expr_patients).intersection(set(dnam_patients)))
    commmon_patients = sorted(commmon_patients, reverse=False)
    sorted_tuples = sorted([(s, s[-1]) for s in commmon_patients], key=lambda x: x[1])
    commmon_patients = list(map(lambda x: x[0], sorted_tuples))

    df_commmon_patients = pd.DataFrame(
        commmon_patients, columns=["final_patient_names"]
    )
    df_commmon_patients.to_csv(config.p_base + "final_dnam_rna_patients.csv", sep="\t")
    print(df_commmon_patients.head())

    signals_do_d8_res_no_res = merged_do_d8_transpose
    signals_do_d8_res_no_res = signals_do_d8_res_no_res.loc[commmon_patients]
    print(signals_do_d8_res_no_res.head())

    df_reset_signals_do_d8_res_no_res = signals_do_d8_res_no_res.reset_index(drop=True)
    print(df_reset_signals_do_d8_res_no_res)

    df_reset_signals_do_d8_res_no_res.to_csv(
        config.p_merged_signals, sep="\t", index=None
    )
    print(df_reset_signals_do_d8_res_no_res.head())


def load_clean_arrays(config):
    df_merged_signals = pl.read_csv(config.p_merged_signals, separator="\t")
    print(df_merged_signals.head())

    features = df_merged_signals
    feature_names = features.columns

    df_feature_names = pd.DataFrame(feature_names, columns=["feature_names"])
    df_feature_names.to_csv(
        config.p_base + "final_feature_names.tsv", index=None, sep="\t"
    )
    df_merged_signals = df_merged_signals.to_pandas()
    return df_merged_signals


def select_hv_features(negative_df: pd.DataFrame, n_top: int) -> pd.DataFrame:
    """
    Select highly variable 'genes' (here: probe_gene features) using Scanpy's
    Seurat flavor. negative_df is shape (n_samples, n_features).
    """
    # Scanpy expects observations (cells/samples) x variables (features)
    adata = sc.AnnData(negative_df)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top)
    hv_idx = adata.var["highly_variable"].fillna(False)
    hv_names = adata.var.index[hv_idx].tolist()
    return negative_df[hv_names]


def extract_positives_negatives(clean_signals, config):
    rng = random.Random(config.SEED)
    np.random.seed(config.SEED)
    non_rand_cpgs = pd.read_csv(config.p_seeds_methylated_cpgs, sep="\t")
    print("Non randomly demethylated cpgs (n=%d):", len(non_rand_cpgs))
    print(non_rand_cpgs.head())
    all_cols = non_rand_cpgs.columns
    all_cols = [item.strip() for item in all_cols]
    non_rand_cpgs.columns = all_cols

    genes_symbols_diff_exp_meth = pd.read_csv(
        config.p_seeds_gene_methylation_expr, sep="\t"
    )
    print("Differentially expressed genes (n=%d):", len(genes_symbols_diff_exp_meth))
    print(genes_symbols_diff_exp_meth.head())

    positive_probes_genes = list()
    for index, col in enumerate(clean_signals.columns):
        cpg, gene = col.split("_")[0], col.split("_")[1]
        if (
            gene in genes_symbols_diff_exp_meth["Gene symbol"].tolist()
            or cpg in non_rand_cpgs["CpG"].tolist()
        ):
            positive_probes_genes.append(col)
    print(f"Number of positive genes: {len(positive_probes_genes)}")

    positive_signals = clean_signals[positive_probes_genes]
    print(positive_signals.head())

    all_features_names = clean_signals.columns
    negative_cols = [x for x in all_features_names if x not in positive_probes_genes]
    print("All negative features (target n=%d)...", len(negative_cols))
    rng.shuffle(negative_cols)
    df_neg_pool = clean_signals[negative_cols]
    print(df_neg_pool.head())

    print(
        "Selecting highly variable negative features (target n=%d)...",
        config.size_negative,
    )
    df_neg_hv = select_hv_features(df_neg_pool, n_top=config.size_negative)
    print("Selected %d negative features.", df_neg_hv.shape[1])

    combined_pos_neg_signals = pd.concat([positive_signals, df_neg_hv], axis=1)
    combined_pos_neg_signals.to_csv(
        config.p_combined_pos_neg_signals, sep="\t", index=False
    )
    print(combined_pos_neg_signals.head())

    edges = build_correlation_edges(
        combined_pos_neg_signals, threshold=config.corr_threshold
    )

    print("Correlation edges found: %d", len(edges))

    # The original code wrote no header; keep that behavior:
    edges.to_csv(config.p_significant_edges, sep="\t", header=False, index=False)


def build_correlation_edges(
    features_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Compute feature-feature correlation and return edges (In, Out)
    where corr > threshold (excluding self-edges).
    features_df: samples x features
    """
    feature_names = features_df.columns.tolist()
    # corrcoef expects rows as variables when input is 2D array; transpose:
    corr = np.corrcoef(features_df.T)
    corr_df = pd.DataFrame(corr, index=feature_names, columns=feature_names)

    mask = corr_df > threshold
    np.fill_diagonal(mask.values, False)  # remove self-relations

    in_nodes, out_nodes = [], []
    for col in mask.columns:
        hits = mask.index[mask[col]].tolist()
        if hits:
            in_nodes.extend([col] * len(hits))
            out_nodes.extend(hits)

    edges = pd.DataFrame({"In": in_nodes, "Out": out_nodes})
    return edges


def create_network_gene_ids(ppi_path, links_path):
    gene = {}
    ngene = 0
    mat = {}
    with open(ppi_path, "r") as flink, open(links_path, "w") as fout_link:
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

    with open(seed_genes_path, "r") as fin:
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

    # out_gene = "data/output/out_gene"
    with open(genes_path, "w") as fout_gene:
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
    result = subprocess.run(
        ["nedbit-features-calculator", links_data_path, genes_data_path, nedbit_path],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


def assign_initial_labels(
    nedbit_path, header, output_gene_ranking_path, q1=0.05, q2=0.2
):
    # apu_label_propagation nedbit_features HEADER_PRESENCE output_gene_ranking 0.05 0.2
    # apu-label-propagation
    print("propagating labels ...")
    result = subprocess.run(
        [
            "apu-label-propagation",
            nedbit_path,
            header,
            output_gene_ranking_path,
            q1,
            q2,
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")

    ## Step 1
    print("======== Step 1: loading datasets =============")
    extract_preprocessed_data(config) if config.download_raw_data else None
    processed_arrays, probe_mapper_full = load_illumina_arrays(config)
    clean_probes = filter_probes(
        config.p_snp_probes, processed_arrays, probe_mapper_full
    )
    merge_patients(clean_probes, config)

    ## Step 2
    print("======== Step 2: cleaning datasets and computing correlation =============")
    clean_arrays = load_clean_arrays(config)
    features = extract_positives_negatives(clean_arrays, config)

    ## Step 3
    print("======== Step 3: Label propagation =============")
    genes = create_network_gene_ids(config.p_significant_edges, config.p_out_links)
    mark_seed_genes(config.p_seed_features, config.p_out_genes, genes)
    calculate_features(config.p_out_links, config.p_out_genes, config.p_nedbit_features)
    assign_initial_labels(
        config.p_nedbit_features,
        str(config.nedbit_header),
        config.p_out_gene_rankings,
        str(config.quantile_1),
        str(config.quantile_2),
    )
