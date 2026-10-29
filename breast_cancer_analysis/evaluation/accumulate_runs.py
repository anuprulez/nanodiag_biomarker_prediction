#!/usr/bin/env python3
"""Create cumulative training and evaluation plots from a dataset's runs/run<number> archives."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_configs import downstream_configs
from preprocessing_grid import add_parameter_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/config.yaml')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    add_parameter_arguments(parser)
    args = parser.parse_args()
    config_path = args.config.resolve()
    os.chdir(ROOT/'train_gnn')
    for config in downstream_configs(config_path, args):
        if args.dry_run:
            print(f"Accumulate {config.p_base}runs/run*/generated + evaluation/run* -> {config.p_base}accumulated_results")
            continue
        from plot_degree_per_soft_label import plot_degree_per_soft_label
        plot_degree_per_soft_label(config)
        from train_gnn.plot_aggregations import plot_cumulative_runs
        plot_cumulative_runs(config, args.models)
        from accumulate_evaluation import accumulate_evaluation
        accumulate_evaluation(config, args.models)


if __name__ == '__main__':
    main()
