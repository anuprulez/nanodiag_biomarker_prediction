#!/usr/bin/env python3
"""
Process Illumina methylation arrays to build positive/negative probe-gene
feature matrices, filter SNP/X/Y probes, select highly variable negatives,
and derive probe–probe relations via correlation.
"""

from __future__ import annotations

import os
import random
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scanpy as sc

from omegaconf.omegaconf import OmegaConf

# ----------------------------- Helper Functions ------------------------------


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


def expand_probe_gene_map(probe_mapper: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a mapper with columns: ["probeID", "UCSC_RefGene_Name"] where names
    can be semicolon-separated into a long form DataFrame:

    Returns DataFrame with columns ["ID_REF", "Genes"].
    """
    if not {"probeID", "UCSC_RefGene_Name"} <= set(probe_mapper.columns):
        raise ValueError(
            "probe_mapper must contain 'probeID' and 'UCSC_RefGene_Name' columns."
        )

    probe_mapper = probe_mapper.dropna(subset=["probeID", "UCSC_RefGene_Name"])
    expanded_probe_ids: list[str] = []
    gene_names: list[str] = []

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
    snp_probes: list[str],
    cr_probes: list[str],
    xy_probe_ids: list[str],
) -> pd.DataFrame:
    """
    Remove rows where ID_REF is in SNP probe list or X/Y chromosome probe IDs.
    """
    excluded = set(snp_probes) | set(xy_probe_ids) | set(cr_probes)
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
    # Subset to HVGs
    hvg = adata.var[adata.var["highly_variable"]]
    # Lowest normalized dispersion among selected HVGs
    lowest_disp = hvg["dispersions_norm"].min()
    # Also, which gene has that lowest dispersion
    lowest_gene = hvg["dispersions_norm"].idxmin()
    print("Lowest dispersion among HVGs:", lowest_disp)
    print("Gene with lowest dispersion:", lowest_gene)
    hv_idx = adata.var["highly_variable"].fillna(False)
    hv_names = adata.var.index[hv_idx].tolist()
    return negative_df[hv_names]

def build_knn_correlation_edges(
    features_df: pd.DataFrame,
    k: int,
    corr_threshold: float = 0.5,
    mutual: bool = False,
) -> pd.DataFrame:
    """
    Build a kNN correlation graph between features and return edges (In, Out, Corr)
    where corr > corr_threshold.

    Parameters
    ----------
    features_df : pd.DataFrame
        Input matrix (samples x features)
    k : int
        Number of neighbors per feature (excluding self)
    corr_threshold : float
        Minimum correlation required to keep an edge
    mutual : bool
        If True, keep only edges that are mutual in both directions

    Returns
    -------
    edges : pd.DataFrame
        Columns: ["In", "Out", "Corr"]
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    feature_names = features_df.columns.to_list()
    n_feat = len(feature_names)
    if n_feat < 2:
        return pd.DataFrame(columns=["In", "Out", "Corr"])

    # Compute feature-feature correlation matrix
    corr = np.corrcoef(features_df.T)
    np.fill_diagonal(corr, -np.inf)  # exclude self-correlation
    corr = np.nan_to_num(corr, nan=-np.inf)

    k_eff = min(k, n_feat - 1)
    in_nodes, out_nodes, corrs = [], [], []
    neighbors_of = [None] * n_feat

    # Build kNN edges for each feature
    for j in range(n_feat):
        # Get top-k most correlated features
        idx = np.argpartition(corr[:, j], -k_eff)[-k_eff:]
        idx = idx[np.argsort(corr[idx, j])[::-1]]  # sort descending
        neighbors_of[j] = set(idx)

        # Keep only those with correlation above threshold
        for i in idx:
            cval = corr[i, j]
            if cval > corr_threshold:
                in_nodes.append(feature_names[j])
                out_nodes.append(feature_names[i])
                corrs.append(cval)

    if mutual:
        # Keep only mutual edges
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        keep_in, keep_out, keep_corr = [], [], []
        for u, v, c in zip(in_nodes, out_nodes, corrs):
            iu, iv = name_to_idx[u], name_to_idx[v]
            if iu in neighbors_of[iv] and iv in neighbors_of[iu]:
                keep_in.append(u)
                keep_out.append(v)
                keep_corr.append(c)
        in_nodes, out_nodes, corrs = keep_in, keep_out, keep_corr

    edges = pd.DataFrame({"In": in_nodes, "Out": out_nodes})
    edges_corr = pd.DataFrame({"In": in_nodes, "Out": out_nodes, "Corr": corrs})
    return edges, edges_corr


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
    return edges, corr_df


def extract_seed_features(
    pos_feature_names: list[str],
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
    probe_ids: list[str] = []
    importances: list[float] = []

    for feat in pos_feature_names:
        probe = feat.split("_", 1)[0]
        if probe in probe_to_delta:
            probe_ids.append(feat)
            importances.append(abs(float(probe_to_delta[probe])))

    return pd.DataFrame({"feature": probe_ids, "importance": importances})


SPLIT_NAMES = ("train", "val", "test")


def split_path(path: str, split: str) -> str:
    """Return a sibling path with ``_<split>`` before its extension."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{split}{p.suffix}"))


def partition_probe_features(
    feature_names: Sequence[str], ratios: Sequence[float], rng: random.Random
) -> dict[str, list[str]]:
    """Split nodes by CpG ID, keeping every mapping of a CpG in one partition."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split_ratios must contain three values that sum to 1")

    by_probe: dict[str, list[str]] = {}
    for feature in feature_names:
        by_probe.setdefault(feature.split("_", 1)[0], []).append(feature)
    probes = sorted(by_probe)
    rng.shuffle(probes)
    n_train = int(len(probes) * ratios[0])
    n_val = int(len(probes) * ratios[1])
    probe_splits = {
        "train": probes[:n_train],
        "val": probes[n_train : n_train + n_val],
        "test": probes[n_train + n_val :],
    }
    return {
        split: [feature for probe in selected for feature in by_probe[probe]]
        for split, selected in probe_splits.items()
    }


# --------------------------------- Pipeline ----------------------------------


def process_arrays(
    config,
) -> None:
    """
    Full processing pipeline. See module docstring for outputs.
    """
    rng = random.Random(config.SEED)
    np.random.seed(config.SEED)

    output_dir = config.p_base
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Illumina arrays: %s", config.p_arrays)
    df_arrays = pd.read_csv(config.p_arrays, sep="\t")
    print("Arrays head:\n%s", df_arrays.head())

    # ---------------------- Gene mapper & probe expansion ----------------------
    print("Loading gene mapper: %s", config.p_mapper)
    probe_mapper = pd.read_csv(config.p_mapper, sep="\t")
    probe_mapper = probe_mapper[probe_mapper["P.Value"] <= config.cutoff_pval]
    print("Mapper after P.val filtering:\n%s", probe_mapper)
    df_probe_gene_mapper = expand_probe_gene_map(probe_mapper)
    print("Expanded mapper head:\n%s", df_probe_gene_mapper.head())

    # Merge probe-gene map with array signals
    print("Mapping genes and probes...")
    df_probe_signals = df_probe_gene_mapper.merge(
        df_arrays, how="inner", left_on="ID_REF", right_on="ID_REF"
    )
    print("Probe signals head:\n%s", df_probe_signals.head())

    # ------------------------- Platform / X,Y filtering ------------------------
    print("Loading platform (for X/Y probes): %s", config.p_platform)
    df_platform = pd.read_csv(config.p_platform, sep=",")
    if not {"ID", "CHR"} <= set(df_platform.columns):
        raise ValueError("Platform file must contain 'ID' and 'CHR' columns.")
    xy_ids = (
        df_platform.loc[df_platform["CHR"].isin(["X", "Y"]), "ID"].astype(str).tolist()
    )

    # ------------------------------ SNP filtering ------------------------------
    print("Filtering SNP probes: %s", config.p_snp_probes)
    snp_probes = (
        pd.read_csv(config.p_snp_probes, sep=",", header=None)[0].astype(str).tolist()
    )
    print("Loaded %d SNP probes", len(snp_probes))

    cr_probes = pd.read_csv(config.p_cross_reactive_probes, sep=",", header=None)
    print("Loaded %d Cross-reactive probes", len(cr_probes))
    cross_reactive_probes = cr_probes[0].tolist()
    print(f"Number of cross-reactive probes: {len(cross_reactive_probes)}")

    df_no_snp_xy = filter_snps_xy(df_probe_signals, snp_probes, cross_reactive_probes, xy_ids)

    # ----------------------------- Feature building ----------------------------
    feature_names = (
        df_no_snp_xy["ID_REF"].astype(str) + "_" + df_no_snp_xy["Genes"].astype(str)
    ).tolist()

    # Signals-only matrix (samples x features)
    data_only = df_no_snp_xy.iloc[:, 2:].reset_index(drop=True)
    data_T = data_only.T  # transpose so rows=samples, cols=features
    data_T.columns = feature_names

    df_merged = data_T  # samples x features

    # ------------------------------- Seeds filter -------------------------------
    print("Loading seeds: %s", config.p_seeds)
    df_seeds = pd.read_csv(config.p_seeds, sep="\t")
    # P-value and chromosome filters
    df_seeds = df_seeds.loc[
        (df_seeds["P.Value"] <= config.cutoff_pval) & (df_seeds["CHR"] != "X")
    ].copy()
    print("Seeds after P.val filtering:\n%s", df_seeds)

    # Positive features: any feature whose probe part is in seeds
    all_seed_probes = set(df_seeds["probeID"].astype(str).tolist())
    positive_features = [
        c for c in df_merged.columns if c.split("_", 1)[0] in all_seed_probes
    ]
    print("Positive features selected: %d", len(positive_features))
    # Negative pool: everything else
    all_features = df_merged.columns.tolist()
    negative_pool = [c for c in all_features if c not in positive_features]

    # Shuffle negative pool (deterministic) before HVG selection
    rng.shuffle(negative_pool)
    df_neg_pool = df_merged[negative_pool]

    # Select highly variable negatives (up to size_negative)
    print(
        "Selecting highly variable negative features (target n=%d)...",
        config.size_negative,
    )
    df_neg_hv = select_hv_features(df_neg_pool, n_top=config.size_negative)
    print("Selected %d negative features.", df_neg_hv.shape[1])

    # Split before graph/NeDBIT computation. Split by probe rather than mapped
    # feature so one CpG with several gene mappings cannot cross partitions.
    ratios = list(config.split_ratios)
    positive_splits = partition_probe_features(positive_features, ratios, rng)
    negative_splits = partition_probe_features(df_neg_hv.columns.tolist(), ratios, rng)
    manifest_rows = []
    combined_parts = []

    for split in SPLIT_NAMES:
        split_features = positive_splits[split] + negative_splits[split]
        combined = df_merged[split_features]
        combined_parts.append(combined)
        combined.to_csv(
            split_path(config.p_combined_pos_neg_signals, split),
            sep="\t",
            index=False,
        )
        print(
            f"Computing {split} correlations for {len(split_features)} nodes "
            f"(threshold={config.corr_threshold:.2f})..."
        )
        edges, edges_corr = build_knn_correlation_edges(
            combined,
            k=config.knn_neighbours,
            corr_threshold=config.corr_threshold,
            mutual=True,
        )
        edges.to_csv(
            split_path(config.p_significant_edges, split),
            sep="\t", header=False, index=False,
        )
        edges_corr.to_csv(
            split_path(config.p_significant_edges_with_corr, split),
            sep="\t", header=False, index=False,
        )
        seed_feats = extract_seed_features(positive_splits[split], df_seeds)
        seed_feats.to_csv(
            split_path(config.p_seed_features, split),
            sep="\t", index=False, header=False,
        )
        positive_set = set(positive_splits[split])
        manifest_rows.extend(
            {"feature_name": name, "split": split,
             "source": "seed" if name in positive_set else "highly_variable"}
            for name in split_features
        )
        print(
            f"{split}: {len(positive_splits[split])} seed mappings, "
            f"{len(negative_splits[split])} highly-variable mappings, "
            f"{len(edges)} edges"
        )

    # Canonical signal file and manifest describe the disconnected union used by
    # training. No cross-split correlations are ever calculated.
    pd.concat(combined_parts, axis=1).to_csv(
        config.p_combined_pos_neg_signals, sep="\t", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(config.p_split_manifest, index=False)

    print("Done.")


def create_network_gene_ids(ppi_path, links_path, all_nodes=None):
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
        # Correlation filtering can isolate nodes. Retain them in the feature and
        # label files even though they have no edges.
        for node in all_nodes or []:
            if node not in gene:
                gene[node] = ngene
                ngene += 1
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
    # nedbit-features-calculator out_links out_genes nedbit_features
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
    result.check_returncode()


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
    result.check_returncode()


def merge_split_graphs(config) -> None:
    """Merge three independently processed graphs as a disconnected union."""
    merged_genes = []
    merged_edges = []
    merged_nedbit = []
    merged_rankings = []
    offset = 0
    for split in SPLIT_NAMES:
        genes = pd.read_csv(split_path(config.p_out_genes, split), sep=" ", header=None)
        links = pd.read_csv(split_path(config.p_out_links, split), sep=" ", header=None)
        genes[0] = genes[0].astype(int) + offset
        if not links.empty:
            links = links.astype(int) + offset
        nedbit = pd.read_csv(split_path(config.p_nedbit_features, split))
        rankings = pd.read_csv(
            split_path(config.p_out_gene_rankings, split), sep=r"\s+", header=None
        )
        merged_genes.append(genes)
        merged_edges.append(links)
        merged_nedbit.append(nedbit)
        merged_rankings.append(rankings)
        offset += len(genes)

    pd.concat(merged_genes, ignore_index=True).to_csv(
        config.p_out_genes, sep=" ", header=False, index=False
    )
    pd.concat(merged_edges, ignore_index=True).to_csv(
        config.p_out_links, sep=" ", header=False, index=False
    )
    pd.concat(merged_nedbit, ignore_index=True).to_csv(
        config.p_nedbit_features, index=False
    )
    pd.concat(merged_rankings, ignore_index=True).to_csv(
        config.p_out_gene_rankings, sep=" ", header=False, index=False
    )


# ----------------------------------  Main  -----------------------------------


def main() -> None:
    config = OmegaConf.load("../config/config.yaml")
    extract_preprocessed_data(config) if config.download_raw_data else None
    process_arrays(config)
    manifest = pd.read_csv(config.p_split_manifest)
    for split in SPLIT_NAMES:
        selected = manifest.loc[manifest["split"] == split, "feature_name"].tolist()
        p_edges = split_path(config.p_significant_edges, split)
        p_links = split_path(config.p_out_links, split)
        p_genes = split_path(config.p_out_genes, split)
        p_nedbit = split_path(config.p_nedbit_features, split)
        p_rankings = split_path(config.p_out_gene_rankings, split)
        genes = create_network_gene_ids(p_edges, p_links, selected)
        mark_seed_genes(split_path(config.p_seed_features, split), p_genes, genes)
        calculate_features(p_links, p_genes, p_nedbit)
        assign_initial_labels(
            p_nedbit, str(config.nedbit_header), p_rankings,
            str(config.quantile_1), str(config.quantile_2),
        )
    merge_split_graphs(config)


if __name__ == "__main__":
    main()
