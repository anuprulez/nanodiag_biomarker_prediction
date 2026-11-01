#!/usr/bin/env python3
"""Create cumulative training and evaluation plots from a dataset's runs/run<number> archives."""
import argparse
import os
import subprocess
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_configs import downstream_configs
from preprocessing_grid import add_parameter_arguments


def ensure_archived_evaluations(config, models=None):
    """Build missing evaluation tables from each immutable training archive."""
    from training_runs import completed_runs, comparison_run, archived_generated_root
    from train_gnn.model_settings import apply_model_settings
    folder = Path(config.p_base).resolve()
    runs = completed_runs(folder)
    if not runs:
        raise FileNotFoundError(f'{folder}: no completed training archives; run --stage train first')
    selected = list(models or config.model_type)
    suffix_tail = folder.name.removeprefix('data_')
    rebuilt = 0
    for run in runs:
        _, archived, available = comparison_run(folder, run.name)
        for model in selected:
            if model not in available:
                continue
            output = folder/'evaluation'/run.name
            required = [output/f'{stem}_{model}_{suffix_tail}.tsv' for stem in
                        ('summary', 'criterion_sensitivity',
                         'likely_positive_common_unique_cpgs', 'negative_common_unique_cpgs')]
            if all(path.is_file() for path in required):
                continue
            current = apply_model_settings(archived, model)
            generated = archived_generated_root(folder, run, current, model)
            command = [sys.executable, str(ROOT/'evaluation/compare_predictions.py'),
                       '--data', str(folder), '--config', str(run/'config.yaml'),
                       '--matrix', str(Path(current.p_arrays).resolve()),
                       '--metadata', str(Path(current.p_sample_metadata).resolve()),
                       '--generated-dir', str(generated), '--output', str(output),
                       '--model', model, '--tables-only']
            print(f'{run.name}/{model}: rebuilding missing archived evaluation tables', flush=True)
            subprocess.run(command, check=True)
            if not all(path.is_file() for path in required):
                raise RuntimeError(f'{run.name}/{model}: comparison did not produce required tables in {output}')
            rebuilt += 1
    print(f'Archived evaluation tables ready ({len(runs)} runs; {rebuilt} rebuilt)', flush=True)
    return rebuilt


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
        ensure_archived_evaluations(config, args.models)
        from plot_degree_per_soft_label import plot_degree_per_soft_label
        plot_degree_per_soft_label(config)
        from train_gnn.plot_aggregations import plot_cumulative_runs
        plot_cumulative_runs(config, args.models)
        from accumulate_evaluation import accumulate_evaluation
        accumulate_evaluation(config, args.models)


if __name__ == '__main__':
    main()
