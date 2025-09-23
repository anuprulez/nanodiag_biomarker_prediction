#!/usr/bin/env python3
"""
Process Illumina methylation arrays to build positive/negative probe-gene
feature matrices, filter SNP/X/Y probes, select highly variable negatives,
and derive probe–probe relations via correlation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import List, Tuple
import argparse
import sys
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc

from omegaconf.omegaconf import OmegaConf

# ----------------------------- Helper Functions ------------------------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )

def expand_probe_gene_map(probe_mapper: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a mapper with columns: ["probeID", "UCSC_RefGene_Name"] where names
    can be semicolon-separated into a long form DataFrame:

    Returns DataFrame with columns ["ID_REF", "Genes"].
    """
    if not {"probeID", "UCSC_RefGene_Name"} <= set(probe_mapper.columns):
        raise ValueError("probe_mapper must contain 'probeID' and 'UCSC_RefGene_Name' columns.")

    probe_mapper = probe_mapper.dropna(subset=["probeID", "UCSC_RefGene_Name"])
    expanded_probe_ids: List[str] = []
    gene_names: List[str] = []

    for _, row in probe_mapper.iterrows():
        genes = str(row["UCSC_RefGene_Name"]).split(";")
        genes = [g for g in genes if g]  # drop empty fragments
        expanded_probe_ids.extend([row["probeID"]] * len(genes))
        gene_names.extend(genes)

    df = pd.DataFrame({"ID_REF": expanded_probe_ids, "Genes": gene_names})
    df = df.dropna().drop_duplicates(["ID_REF", "Genes"]).reset_index(drop=True)
    return df


def filter_snps_xy(
    merged: pd.DataFrame,
    snp_probes: List[str],
    xy_probe_ids: List[str],
) -> pd.DataFrame:
    """
    Remove rows where ID_REF is in SNP probe list or X/Y chromosome probe IDs.
    """
    excluded = set(snp_probes) | set(xy_probe_ids)
    keep_mask = ~merged["ID_REF"].isin(excluded)
    return merged.loc[keep_mask].copy()


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

    mask = (corr_df > threshold)
    np.fill_diagonal(mask.values, False)  # remove self-relations

    in_nodes, out_nodes = [], []
    for col in mask.columns:
        hits = mask.index[mask[col]].tolist()
        if hits:
            in_nodes.extend([col] * len(hits))
            out_nodes.extend(hits)

    edges = pd.DataFrame({"In": in_nodes, "Out": out_nodes})
    return edges


def extract_seed_features(
    pos_feature_names: List[str],
    seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    From positive feature names like 'cg00000029_TP73', match 'cg...' to seeds_df['probeID']
    and record |deltaBeta| as importance.
    Returns DataFrame with two columns: [feature_id, importance]
    """
    if not {"probeID", "deltaBeta"} <= set(seeds_df.columns):
        raise ValueError("seeds_df must contain 'probeID' and 'deltaBeta' columns.")

    probe_to_delta = dict(zip(seeds_df["probeID"], seeds_df["deltaBeta"]))
    probe_ids: List[str] = []
    importances: List[float] = []

    for feat in pos_feature_names:
        probe = feat.split("_", 1)[0]
        if probe in probe_to_delta:
            probe_ids.append(feat)
            importances.append(abs(float(probe_to_delta[probe])))

    return pd.DataFrame({"feature": probe_ids, "importance": importances})


# --------------------------------- Pipeline ----------------------------------

def process_arrays(
    config,
) -> None:
    """
    Full processing pipeline. See module docstring for outputs.
    """
    rng = random.Random(config.SEED)
    np.random.seed(config.SEED)

    print("Loading Illumina arrays: %s", config.p_arrays)
    df_arrays = pd.read_csv(config.p_arrays, sep="\t")
    print("Arrays head:\n%s", df_arrays.head())

    # ---------------------- Gene mapper & probe expansion ----------------------
    print("Loading gene mapper: %s", config.p_mapper)
    probe_mapper = pd.read_csv(config.p_mapper, sep="\t")
    df_probe_gene_mapper = expand_probe_gene_map(probe_mapper)
    print("Expanded mapper head:\n%s", df_probe_gene_mapper.head())

    # Merge probe-gene map with array signals
    print("Mapping genes and probes...")
    df_probe_signals = df_probe_gene_mapper.merge(df_arrays, how="inner", left_on="ID_REF", right_on="ID_REF")
    print("Probe signals head:\n%s", df_probe_signals.head())

    # ------------------------- Platform / X,Y filtering ------------------------
    print("Loading platform (for X/Y probes): %s", config.p_platform)
    df_platform = pd.read_csv(config.p_platform, sep=",")
    if not {"ID", "CHR"} <= set(df_platform.columns):
        raise ValueError("Platform file must contain 'ID' and 'CHR' columns.")
    xy_ids = df_platform.loc[df_platform["CHR"].isin(["X", "Y"]), "ID"].astype(str).tolist()

    # ------------------------------ SNP filtering ------------------------------
    print("Filtering SNP probes: %s", config.p_snp_probes)
    snp_probes = pd.read_csv(config.p_snp_probes, sep=",", header=None)[0].astype(str).tolist()
    print("Loaded %d SNP probes", len(snp_probes))

    df_no_snp_xy = filter_snps_xy(df_probe_signals, snp_probes, xy_ids)

    # ----------------------------- Feature building ----------------------------
    info_cols = df_no_snp_xy[["ID_REF", "Genes"]].copy()
    feature_names = (df_no_snp_xy["ID_REF"].astype(str) + "_" + df_no_snp_xy["Genes"].astype(str)).tolist()

    # Signals-only matrix (samples x features)
    data_only = df_no_snp_xy.iloc[:, 2:].reset_index(drop=True)
    data_T = data_only.T  # transpose so rows=samples, cols=features
    data_T.columns = feature_names

    print("Saving merged signals to: %s", config.p_merged_signals)
    data_T.to_csv(config.p_merged_signals, sep="\t", index=False)

    df_merged = data_T  # samples x features

    # ------------------------------- Seeds filter -------------------------------
    print("Loading seeds: %s", config.p_seeds)
    df_seeds = pd.read_csv(config.p_seeds, sep="\t")
    # P-value and chromosome filters
    df_seeds = df_seeds.loc[(df_seeds["P.Value"] <= 0.01) & (df_seeds["CHR"] != "X")].copy()

    # Positive features: any feature whose probe part is in seeds
    all_seed_probes = set(df_seeds["probeID"].astype(str).tolist())
    positive_features = [c for c in df_merged.columns if c.split("_", 1)[0] in all_seed_probes]
    print("Positive features selected: %d", len(positive_features))
    df_pos = df_merged[positive_features]

    # Negative pool: everything else
    all_features = df_merged.columns.tolist()
    negative_pool = [c for c in all_features if c not in positive_features]

    # Shuffle negative pool (deterministic) before HVG selection
    rng.shuffle(negative_pool)
    df_neg_pool = df_merged[negative_pool]

    # Select highly variable negatives (up to size_negative)
    print("Selecting highly variable negative features (target n=%d)...", config.size_negative)
    df_neg_hv = select_hv_features(df_neg_pool, n_top=config.size_negative)
    print("Selected %d negative features.", df_neg_hv.shape[1])

    # Save positive/negative feature name lists
    #pd.DataFrame({"negative_probes_genes": df_neg_hv.columns}).to_csv(
    #    out_dir / "negative_probes_genes_large.tsv", sep="\t", index=False
    #)
    #pd.DataFrame({"positive_probes_genes": positive_features}).to_csv(
    #    out_dir / "positive_probes_genes.tsv", sep="\t", index=False
    #)

    # Save matrices
    #df_neg_hv.to_csv(out_dir / "balanced_negative_signals.tsv", sep="\t", index=False)
    #df_pos.to_csv(out_dir / "positive_signals.tsv", sep="\t", index=False)

    # ----------------------- Combine and build correlations ---------------------
    combined = pd.concat([df_pos, df_neg_hv], axis=1)  # samples x features
    combined.to_csv(config.p_combined_pos_neg_signals, sep="\t", index=False)

    print("Computing feature–feature correlations (threshold=%.2f)...", config.corr_threshold)
    edges = build_correlation_edges(combined, threshold=config.corr_threshold)
    
    print("Correlation edges found: %d", len(edges))

    #edges_path = out_dir / f"significant_gene_relation_{cohort_name}.tsv"
    # The original code wrote no header; keep that behavior:
    edges = edges[:10000]
    edges.to_csv(config.p_significant_edges, sep="\t", header=False, index=False)

    # --------------------------- Seed feature importances -----------------------
    print("Extracting seed feature importances...")
    seed_feats = extract_seed_features(positive_features, df_seeds)
    seed_feats.to_csv(config.p_seed_features, sep="\t", index=False, header=False)

    print("Done.")


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

# ----------------------------------  Main  -----------------------------------

def main() -> None:
    
    config = OmegaConf.load("../config/config.yaml")
    process_arrays(config)
    genes = create_network_gene_ids(config.p_significant_edges, config.p_out_links)
    mark_seed_genes(config.p_seed_features, config.p_out_genes, genes)
    calculate_features(config.p_out_links, config.p_out_genes, config.p_nedbit_features)
    
    assign_initial_labels(config.p_nedbit_features, str(config.nedbit_header), config.p_out_gene_rankings, str(config.quantile_1), str(config.quantile_2))



if __name__ == "__main__":
    main()
