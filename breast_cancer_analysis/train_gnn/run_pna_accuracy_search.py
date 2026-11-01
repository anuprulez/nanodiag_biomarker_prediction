#!/usr/bin/env python3
"""Search PNA validation accuracy without installing settings or evaluating test data."""
import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf
from tune_models import (HERE, tuning_source, preprocessing_configs,
                         optimize_model, validate_seed_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE.parent /
                        'data_corr_0.33_knn_n_3000_n_neg_10000/preprocessing_config.yaml')
    parser.add_argument('--search-space', type=Path, default=HERE.parent / 'config/optuna_pna_accuracy.yaml')
    parser.add_argument('--trials', type=int, default=120, help='Additional trials')
    parser.add_argument('--timeout', type=float)
    parser.add_argument('--study-tag', default='pna_accuracy_architecture_20261028')
    args = parser.parse_args()
    if args.trials < 1 or (args.timeout is not None and args.timeout <= 0):
        parser.error('trials and timeout must be positive')
    config_path = args.config.resolve()
    search = OmegaConf.load(args.search_space.resolve())
    if search.metric != 'accuracy' or list(search.models) != ['PNA']:
        parser.error('Expected a PNA-only accuracy search')
    os.chdir(HERE)
    base = tuning_source(config_path, search)
    for current in preprocessing_configs(base):
        validate_seed_manifest(current)
        directory = optimize_model(current, search, 'PNA', args.trials,
                                   timeout=args.timeout, study_tag=args.study_tag)
        print(f'Validation winner exported to {directory}; dataset config unchanged.')


if __name__ == '__main__':
    main()
