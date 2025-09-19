#!/usr/bin/env python3
"""
Process Illumina methylation arrays to build positive/negative probe-gene
feature matrices, filter SNP/X/Y probes, select highly variable negatives,
and derive probe–probe relations via correlation.

Outputs (TSV/CSV in output folder):
- merged_signals_bc.csv
- negative_probes_genes_large.tsv
- positive_probes_genes.tsv
- balanced_negative_signals.tsv
- positive_signals.tsv
- combined_pos_neg_signals_bc.csv
- significant_gene_relation_<cohort>.tsv
- seed_features_bc.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


# ----------------------------- Config / Constants -----------------------------

DEFAULT_INPUT_BASE = Path("data_19_Sept_25/inputs")
DEFAULT_OUTPUT_BASE = Path("data_19_Sept_25/outputs")

# ----------------------------- Helper Functions ------------------------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    arrays_path: Path,
    mapper_path: Path,
    snp_probes_path: Path,
    platform_path: Path,
    seeds_path: Path,
    out_dir: Path,
    size_negative: int = 10_000,
    corr_threshold: float = 0.33,
    cohort_name: str = "breast_cancer",
    seed: int = 42,
) -> None:
    """
    Full processing pipeline. See module docstring for outputs.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    ensure_outdir(out_dir)

    logging.info("Loading Illumina arrays: %s", arrays_path)
    df_arrays = pd.read_csv(arrays_path, sep="\t")
    logging.debug("Arrays head:\n%s", df_arrays.head())

    # ---------------------- Gene mapper & probe expansion ----------------------
    logging.info("Loading gene mapper: %s", mapper_path)
    probe_mapper = pd.read_csv(mapper_path, sep="\t")
    df_probe_gene_mapper = expand_probe_gene_map(probe_mapper)
    logging.debug("Expanded mapper head:\n%s", df_probe_gene_mapper.head())

    # Merge probe-gene map with array signals
    logging.info("Mapping genes and probes...")
    df_probe_signals = df_probe_gene_mapper.merge(df_arrays, how="inner", left_on="ID_REF", right_on="ID_REF")
    logging.debug("Probe signals head:\n%s", df_probe_signals.head())

    # ------------------------- Platform / X,Y filtering ------------------------
    logging.info("Loading platform (for X/Y probes): %s", platform_path)
    df_platform = pd.read_csv(platform_path, sep=",")
    if not {"ID", "CHR"} <= set(df_platform.columns):
        raise ValueError("Platform file must contain 'ID' and 'CHR' columns.")
    xy_ids = df_platform.loc[df_platform["CHR"].isin(["X", "Y"]), "ID"].astype(str).tolist()

    # ------------------------------ SNP filtering ------------------------------
    logging.info("Filtering SNP probes: %s", snp_probes_path)
    snp_probes = pd.read_csv(snp_probes_path, sep=",", header=None)[0].astype(str).tolist()
    logging.debug("Loaded %d SNP probes", len(snp_probes))

    df_no_snp_xy = filter_snps_xy(df_probe_signals, snp_probes, xy_ids)

    # ----------------------------- Feature building ----------------------------
    info_cols = df_no_snp_xy[["ID_REF", "Genes"]].copy()
    feature_names = (df_no_snp_xy["ID_REF"].astype(str) + "_" + df_no_snp_xy["Genes"].astype(str)).tolist()

    # Signals-only matrix (samples x features)
    data_only = df_no_snp_xy.iloc[:, 2:].reset_index(drop=True)
    data_T = data_only.T  # transpose so rows=samples, cols=features
    data_T.columns = feature_names

    logging.info("Saving merged signals to: %s", out_dir / "merged_signals_bc.csv")
    data_T.to_csv(out_dir / "merged_signals_bc.csv", sep="\t", index=False)

    df_merged = data_T  # samples x features

    # ------------------------------- Seeds filter -------------------------------
    logging.info("Loading seeds: %s", seeds_path)
    df_seeds = pd.read_csv(seeds_path, sep="\t")
    # P-value and chromosome filters
    df_seeds = df_seeds.loc[(df_seeds["P.Value"] <= 0.01) & (df_seeds["CHR"] != "X")].copy()

    # Positive features: any feature whose probe part is in seeds
    all_seed_probes = set(df_seeds["probeID"].astype(str).tolist())
    positive_features = [c for c in df_merged.columns if c.split("_", 1)[0] in all_seed_probes]
    logging.info("Positive features selected: %d", len(positive_features))
    df_pos = df_merged[positive_features]

    # Negative pool: everything else
    all_features = df_merged.columns.tolist()
    negative_pool = [c for c in all_features if c not in positive_features]

    # Shuffle negative pool (deterministic) before HVG selection
    #rng.shuffle(negative_pool)
    df_neg_pool = df_merged[negative_pool]

    # Select highly variable negatives (up to size_negative)
    logging.info("Selecting highly variable negative features (target n=%d)...", size_negative)
    df_neg_hv = select_hv_features(df_neg_pool, n_top=size_negative)
    logging.info("Selected %d negative features.", df_neg_hv.shape[1])

    # Save positive/negative feature name lists
    pd.DataFrame({"negative_probes_genes": df_neg_hv.columns}).to_csv(
        out_dir / "negative_probes_genes_large.tsv", sep="\t", index=False
    )
    pd.DataFrame({"positive_probes_genes": positive_features}).to_csv(
        out_dir / "positive_probes_genes.tsv", sep="\t", index=False
    )

    # Save matrices
    df_neg_hv.to_csv(out_dir / "balanced_negative_signals.tsv", sep="\t", index=False)
    df_pos.to_csv(out_dir / "positive_signals.tsv", sep="\t", index=False)

    # ----------------------- Combine and build correlations ---------------------
    combined = pd.concat([df_pos, df_neg_hv], axis=1)  # samples x features
    combined.to_csv(out_dir / "combined_pos_neg_signals_bc.csv", sep="\t", index=False)

    logging.info("Computing feature–feature correlations (threshold=%.2f)...", corr_threshold)
    edges = build_correlation_edges(combined, threshold=corr_threshold)
    
    logging.info("Correlation edges found: %d", len(edges))

    edges_path = out_dir / f"significant_gene_relation_{cohort_name}.tsv"
    # The original code wrote no header; keep that behavior:
    filtered_edges = edges[:50000]
    filtered_edges.to_csv(edges_path, sep="\t", header=False, index=False)

    # --------------------------- Seed feature importances -----------------------
    logging.info("Extracting seed feature importances...")
    seed_feats = extract_seed_features(positive_features, df_seeds)
    seed_feats.to_csv(out_dir / "seed_features_bc.csv", sep="\t", index=False, header=False)

    logging.info("Done.")


# ----------------------------------  Main  -----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Illumina array processing pipeline.")
    p.add_argument("--arrays_path", "-ap", type=Path,
                   default=DEFAULT_INPUT_BASE / "GSE237036_matrix_processed.txt",
                   help="Path to arrays matrix (TSV).")
    p.add_argument("--mapper_path", "-mp", type=Path,
                   default=DEFAULT_INPUT_BASE / "probe_genes.csv",
                   help="Path to probe-gene mapper (TSV with columns: probeID, UCSC_RefGene_Name).")
    p.add_argument("--snp_probes_path", "--snp", type=Path,
                   default=DEFAULT_INPUT_BASE / "snp_7998probes.vh20151030.txt",
                   help="Path to SNP probe IDs (CSV single column).")
    p.add_argument("--platform_path", "-pp", type=Path,
                   default=DEFAULT_INPUT_BASE / "GPL21145-48548.txt",
                   help="Path to platform file with columns ID, CHR (CSV).")
    p.add_argument("--seeds_path", "-sp", type=Path,
                   default=DEFAULT_INPUT_BASE / "seed_probes.csv",
                   help="Path to seed probes (TSV with columns: probeID, deltaBeta, P.Value, CHR).")
    p.add_argument("--out_dir", "-o", type=Path, default=DEFAULT_OUTPUT_BASE, help="Output directory.")
    p.add_argument("--size_negative", type=int, default=10_000, help="Number of HV negative features to select.")
    p.add_argument("--corr_threshold", type=float, default=0.33, help="Correlation threshold for edges.")
    p.add_argument("--cohort_name", type=str, default="breast_cancer", help="Name inserted into edges filename.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Basic sanity checks
    for pth in [args.arrays_path, args.mapper_path, args.snp_probes_path, args.platform_path, args.seeds_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Input file not found: {pth}")

    process_arrays(
        arrays_path=args.arrays_path,
        mapper_path=args.mapper_path,
        snp_probes_path=args.snp_probes_path,
        platform_path=args.platform_path,
        seeds_path=args.seeds_path,
        out_dir=args.out_dir,
        size_negative=args.size_negative,
        corr_threshold=args.corr_threshold,
        cohort_name=args.cohort_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
