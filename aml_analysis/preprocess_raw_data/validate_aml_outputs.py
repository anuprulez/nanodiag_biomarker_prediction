#!/usr/bin/env python3
"""Check fixed CpG partitions, graph integrity, labels and AML signal dimensions."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from preprocessing_grid import preprocessing_configs, add_parameter_arguments, load_preprocessing_config


def validate(config_path, args=None):
    config_path = Path(config_path).resolve()
    os.chdir(HERE / 'train_gnn')
    for config in preprocessing_configs(load_preprocessing_config(config_path, args)):
        validate_config(config)


def validate_config(config):
    nodes = pd.read_csv(config.p_out_genes, sep=r'\s+', header=None)
    manifest = pd.read_csv(config.p_split_manifest)
    labels = pd.read_csv(config.p_out_gene_rankings, sep=r'\s+', header=None)
    nedbit = pd.read_csv(config.p_nedbit_features)
    signals = pd.read_csv(config.p_combined_pos_neg_signals, sep='\t')
    assert nodes[0].tolist() == list(range(len(nodes))), 'Noncontiguous graph IDs'
    assert nodes[1].is_unique and manifest['feature_name'].is_unique
    assert labels[0].is_unique and nedbit['name'].is_unique
    assert set(nodes[1]) == set(labels[0]) == set(nedbit['name'])
    assert set(nodes[1]) <= set(signals.columns)
    assert signals.shape[0] == 526
    assert np.isfinite(signals.to_numpy()).all()
    assert ((signals.to_numpy() >= 0) & (signals.to_numpy() <= 1)).all()
    assert np.isfinite(nedbit.iloc[:, 2:].to_numpy(dtype=float)).all()
    assert set(labels[2]) == {1, 2, 3, 4, 5}
    assert set(nedbit['class']) == {0, 1}
    indexed = manifest.set_index('feature_name')
    assigned = indexed.loc[nodes[1], 'split'].to_numpy()
    manifest['CpG'] = manifest['feature_name'].str.split('_', n=1).str[0]
    assert manifest.groupby('CpG')['split'].nunique().eq(1).all(), 'CpG leakage'
    counts = {}
    label_by_name = labels.set_index(0)[2]
    for split in ['train', 'val', 'test']:
        names = nodes.loc[assigned == split, 1]
        distribution = label_by_name.loc[names].value_counts().sort_index().to_dict()
        assert set(distribution) == {1, 2, 3, 4, 5}, f'Missing class in {split}'
        counts[split] = {'nodes': len(names), 'labels': {str(k): int(v) for k, v in distribution.items()}}
    edges = 0
    for chunk in pd.read_csv(config.p_out_links, sep=r'\s+', header=None, chunksize=200000):
        a, b = chunk[0].to_numpy(), chunk[1].to_numpy()
        assert a.min() >= 0 and b.min() >= 0 and a.max() < len(nodes) and b.max() < len(nodes)
        assert np.equal(assigned[a], assigned[b]).all(), 'Cross-split edge'
        assert np.not_equal(a, b).all(), 'Self loop'
        edges += len(chunk)
    report = dict(nodes=len(nodes), edges=edges, samples=signals.shape[0],
                  feature_dimension=len(config.keep_feature_names.split(',')) + signals.shape[0],
                  selected_features=signals.shape[1], splits=counts, cpg_leakage=False,
                  cross_split_edges=False, finite_beta_values=True)
    assert report['feature_dimension'] == config.gene_dim
    path = Path(config.p_base_files) / 'preprocessing_validation.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config/config.yaml')
    add_parameter_arguments(parser)
    args = parser.parse_args()
    validate(args.config.resolve(), args)
