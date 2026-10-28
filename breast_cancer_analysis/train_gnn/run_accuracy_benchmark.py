#!/usr/bin/env python3
"""Run a fixed trial budget per model, then compare frozen validation winners on test."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import optuna
from omegaconf import OmegaConf

from tune_models import (HERE, MODELS, tuning_source, preprocessing_configs,
                         validate_seed_manifest, input_fingerprint, optimize_model, write_json)
from evaluate_accuracy_winner import evaluate_candidate


def now():
    return datetime.now(timezone.utc).isoformat()


def trials_for(root, name):
    path = root / name / 'study.sqlite3'
    if not path.exists():
        return None, []
    study = optuna.load_study(study_name=name, storage='sqlite:///' + str(path))
    return study, study.get_trials(deepcopy=False)


def benchmark(config, search, models, budget, tag):
    validate_seed_manifest(config)
    root = (Path(config.p_generated_files) / 'optuna' / tag).resolve()
    root.mkdir(parents=True, exist_ok=True)
    context = copy.deepcopy(config)
    context.model_type = 'model_selection'
    fingerprint = input_fingerprint(context, search)
    manifest = root / 'benchmark_inputs.json'
    metadata = dict(input_fingerprint=fingerprint, models=models, trials_per_model=budget,
                    selection='Validation accuracy only; all tuning finishes before test evaluation')
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError('Benchmark inputs, model list or trial budget changed; use a new tag')
    else:
        if list(root.glob('*/study.sqlite3')):
            raise ValueError('Existing studies lack a benchmark manifest; use a new tag')
        write_json(manifest, metadata)
        OmegaConf.save(search, root / 'search_space.yaml')
    errors = []
    status = dict(state='tuning', started_utc=now(), models=models, trials_per_model=budget)
    write_json(root / 'status.json', status)
    leaderboard = []
    for name in models:
        status.update(current_model=name, phase='tuning', updated_utc=now())
        write_json(root / 'status.json', status)
        while True:
            study, trials = trials_for(root, name)
            if any(t.state == optuna.trial.TrialState.RUNNING for t in trials):
                raise RuntimeError(f'{name}: another worker or interrupted trial is RUNNING; inspect before resuming')
            finished = sum(t.state.is_finished() for t in trials)
            if finished > budget:
                raise ValueError(f'{name} already has more than {budget} finished trials')
            if finished == budget:
                break
            print(f'{now()} {name}: {finished}/{budget} trials finished; running {budget-finished}', flush=True)
            try:
                optimize_model(config, search, name, budget - finished, study_tag=tag)
            except Exception as error:
                errors.append(dict(model=name, phase='tuning', error=str(error), traceback=traceback.format_exc()))
                write_json(root / 'errors.json', errors)
                _, after = trials_for(root, name)
                after_finished = sum(t.state.is_finished() for t in after)
                if after_finished <= finished:
                    print(f'{name}: failed before making trial progress; continuing other models', flush=True)
                    break
                print(f'{name}: trial failed; recording it and continuing the remaining fixed budget', flush=True)
        study, trials = trials_for(root, name)
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            params_path = root / name / 'best_params.json'
            exported = json.loads(params_path.read_text()) if params_path.exists() else {}
            if exported.get('trial') != study.best_trial.number or exported.get('validation_score') != study.best_value:
                # Export an already completed winner without adding a trial.
                optimize_model(config, search, name, 0, study_tag=tag)
            row = json.loads(params_path.read_text())
            row.update(study=str(params_path.parent), trials_finished=sum(t.state.is_finished() for t in trials),
                       trials_completed=len(completed),
                       trials_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in trials),
                       trials_failed=sum(t.state == optuna.trial.TrialState.FAIL for t in trials))
            leaderboard.append(row)
        leaderboard.sort(key=lambda row: (-row['validation_score'], row['model']))
        write_json(root / 'validation_leaderboard.json', leaderboard)
        if input_fingerprint(context, search) != fingerprint:
            raise RuntimeError('Inputs changed during tuning; do not compare these studies')
    # The entire fixed search is over before any test prediction begins.
    write_json(root / 'validation_selection_frozen.json', dict(created_utc=now(), candidates=leaderboard,
               input_fingerprint=fingerprint, selection='Each model winner frozen on validation accuracy'))
    status.update(state='evaluating', phase='test', updated_utc=now())
    write_json(root / 'status.json', status)
    results = []
    for row in leaderboard:
        name = row['model']
        status.update(current_model=name, updated_utc=now())
        write_json(root / 'status.json', status)
        candidate = root / name / 'validation_candidate.json'
        write_json(candidate, row)
        output = root / name / 'test_evaluation'
        try:
            if (output / 'result.json').exists():
                result = json.loads((output / 'result.json').read_text())
                if result['trial'] != row['trial'] or result['input_fingerprint'] != row['input_fingerprint']:
                    raise ValueError('Existing test evaluation is for a different winner')
            else:
                result = evaluate_candidate(candidate, output)
            results.append(dict(model=name, trial=row['trial'], validation_accuracy=row['validation_score'],
                                reproduced_validation_accuracy=result['validation']['accuracy'],
                                test_accuracy=result['test']['accuracy'], validation_macro_f1=result['validation']['macro_f1'],
                                test_macro_f1=result['test']['macro_f1'], **{k:row[k] for k in
                                ['trials_finished','trials_completed','trials_pruned','trials_failed']}))
        except Exception as error:
            errors.append(dict(model=name, phase='test', error=str(error), traceback=traceback.format_exc()))
            write_json(root / 'errors.json', errors)
    import pandas as pd
    frame = pd.DataFrame(results)
    if results:
        frame = frame.sort_values(['validation_accuracy','model'], ascending=[False,True])
        frame.to_csv(root / 'accuracy_comparison.csv', index=False)
        lines = ['# Model accuracy comparison', '',
                 'Sorted by validation accuracy. Each checkpoint was selected before test prediction.', '',
                 '| Model | Finished trials | Complete / pruned / failed | Validation accuracy | Test accuracy |',
                 '|---|---:|---|---:|---:|']
        for row in frame.to_dict('records'):
            lines.append(f"| {row['model']} | {row['trials_finished']} | {row['trials_completed']} / {row['trials_pruned']} / {row['trials_failed']} | {row['validation_accuracy']:.2%} | {row['test_accuracy']:.2%} |")
        for row in frame.to_dict('records'):
            if abs(row['validation_accuracy'] - row['reproduced_validation_accuracy']) > 1e-10:
                lines += ['', f"{row['model']}: selected validation accuracy {row['validation_accuracy']:.2%}; reloaded validation accuracy {row['reproduced_validation_accuracy']:.2%}. The estimator declares nondeterministic inference. The selected trial and checkpoint were retained."]
        lines += ['', 'Single training seed and fixed node split. Test outcomes were not used for model/hyperparameter selection.',
                  'This existing test split has been inspected historically; these scores are not a new independent holdout.',
                  'Selected production model settings were not replaced.']
        (root / 'ACCURACY_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    complete = len(results) == len(models) and all(row['trials_finished']==budget for row in results)
    status.update(state='complete' if complete else 'incomplete', phase='finished', updated_utc=now(),
                  comparison=str(root/'accuracy_comparison.csv'), results=results, errors=errors)
    write_json(root / 'status.json', status)
    write_json(root / 'errors.json', errors)
    print(json.dumps(status, indent=2), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--search-space', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=25, help='Total finished Optuna trials per model, including failed/pruned trials')
    parser.add_argument('--study-tag', required=True)
    args = parser.parse_args()
    if args.trials < 1 or not args.study_tag or Path(args.study_tag).name != args.study_tag or args.study_tag in ('.','..'):
        parser.error('Use a positive trial budget and a single-component study tag')
    config_path, search_path = args.config.resolve(), args.search_space.resolve()
    search = OmegaConf.load(search_path)
    if search.metric != 'accuracy' or any(name not in MODELS for name in search.models):
        parser.error('Expected an accuracy search with supported models')
    os.chdir(HERE)
    source = tuning_source(config_path, search)
    okay = True
    for config in preprocessing_configs(source):
        okay = benchmark(config, search, list(search.models), args.trials, args.study_tag) and okay
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
