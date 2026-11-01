#!/usr/bin/env python3
"""Compare supported families using validation accuracy without installing model settings."""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback

from omegaconf import OmegaConf
from tune_models import (HERE, MODELS, tuning_source, preprocessing_configs,
                         optimize_model, validate_seed_manifest, write_json, input_fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE.parent /
                        'data_corr_0.33_knn_n_3000_n_neg_10000/preprocessing_config.yaml')
    parser.add_argument('--search-space', type=Path, default=HERE.parent / 'config/optuna_accuracy_all_models.yaml')
    parser.add_argument('--models', choices=MODELS, nargs='+')
    parser.add_argument('--trials', type=int, help='Additional trials per selected model')
    parser.add_argument('--timeout', type=float, help='Seconds per model, checked between trials')
    parser.add_argument('--study-tag', default='accuracy_all_models_20261028')
    args = parser.parse_args()
    config_path, search_path = args.config.resolve(), args.search_space.resolve()
    search = OmegaConf.load(search_path)
    trials = args.trials if args.trials is not None else int(search.n_trials)
    if search.metric != 'accuracy' or trials < 1 or (args.timeout is not None and args.timeout <= 0):
        parser.error('Use accuracy with positive trials and timeout')
    models = list(dict.fromkeys(args.models or search.models))
    if not models or any(model not in MODELS for model in models):
        parser.error('Choose supported models')
    os.chdir(HERE)
    base = tuning_source(config_path, search)
    for current in preprocessing_configs(base):
        validate_seed_manifest(current)
        root = Path(current.p_generated_files) / 'optuna' / args.study_tag
        # Validate before root can be written, including in failure handling.
        if not args.study_tag or Path(args.study_tag).name != args.study_tag or args.study_tag in ('.', '..'):
            parser.error('study-tag must be a single directory name')
        root.mkdir(parents=True, exist_ok=True)
        context = copy.deepcopy(current)
        context.model_type = 'model_selection'
        fingerprint = input_fingerprint(context, search)
        context_path = root / 'selection_inputs.json'
        if context_path.exists():
            previous = json.loads(context_path.read_text())
            if previous['input_fingerprint'] != fingerprint:
                raise ValueError('Selection inputs/search changed; use a new study tag')
        else:
            # Refuse to adopt pre-existing studies with unknown provenance.
            if list(root.glob('*/best_params.json')):
                raise ValueError('Existing selection has no input manifest; use a new study tag')
            write_json(context_path, dict(input_fingerprint=fingerprint,
                                          search_space=str(search_path),
                                          config=str(config_path)))
            OmegaConf.save(search, root / 'search_space.yaml')
        leaderboard, failures = [], []
        for model in models:
            try:
                optimize_model(current, search, model, trials, args.timeout, args.study_tag)
            except Exception as error:
                failures.append(dict(model=model, error=str(error), traceback=traceback.format_exc()))
                print(f'{model}: FAILED: {error}', flush=True)
            if input_fingerprint(context, search) != fingerprint:
                raise RuntimeError('Selection inputs changed during the run; discard this comparison')
            # Include all completed families under this tag, including resumed runs.
            leaderboard = []
            for candidate in root.glob('*/best_params.json'):
                row = json.loads(candidate.read_text())
                if row.get('validation_metric') == 'accuracy':
                    leaderboard.append(dict(row, study=str(candidate.parent.resolve())))
            leaderboard.sort(key=lambda row: (-row['validation_score'], row['model']))
            write_json(root / 'validation_leaderboard.json', leaderboard)
            write_json(root / 'failures.json', failures)
        if leaderboard:
            write_json(root / 'validation_selected_candidate.json',
                       dict(leaderboard[0], selection='Highest validation accuracy; no test selection',
                            test_evaluated=False, model_settings_installed=False))
        print(json.dumps(dict(leaderboard=leaderboard, failures=failures), indent=2), flush=True)
        if not leaderboard:
            raise RuntimeError('No completed model; inspect failures.json')


if __name__ == '__main__':
    main()
