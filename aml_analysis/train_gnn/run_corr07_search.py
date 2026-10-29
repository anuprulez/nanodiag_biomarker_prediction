"""Equal-budget validation-only model comparison for the selected AML dataset."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data_corr_0.7_knn_n_3000_n_neg_10000'
TAG = 'corr07_optunal_20261026'
MODELS = ['PNA', 'GCN', 'GraphSAGE', 'GATv2', 'GraphTransformer', 'XGBoost', 'TabICL', 'TabPFN']


def main():
    import optuna
    output = DATASET / 'logs' / TAG
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    rows = []
    for name in MODELS:
        directory = DATASET / 'generated' / 'optuna' / TAG / name
        attempted = 0
        if (directory / 'study.sqlite3').exists():
            study = optuna.load_study(study_name=name, storage='sqlite:///' + str(directory / 'study.sqlite3'))
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            attempted = len(study.trials)
        remaining = max(0, 25 - attempted)
        command = [sys.executable, '-u', str(ROOT / 'train_gnn/tune_models.py'),
                   '--config', str(DATASET / 'preprocessing_config.yaml'),
                   '--search-space', str(ROOT / 'config/optunal.yaml'),
                   '--models', name, '--study-tag', TAG]
        command += ['--trials', str(remaining)] if remaining else ['--export-existing']
        (output / 'status.json').write_text(json.dumps({'state': 'running', 'model': name, 'completed': rows}, indent=2) + '\n')
        print(f'Starting {name}: {remaining} additional trials, 25 total budget', flush=True)
        with (output / f'{name}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'model': name, 'returncode': result.returncode, 'trial_budget': 25}
        if result.returncode == 0:
            best = json.loads((directory / 'best_params.json').read_text())
            row.update(validation_macro_f1=best['validation_score'], trial=best['trial'],
                       parameters=json.dumps(best['parameters'], sort_keys=True))
        rows.append(row)
        ranked = sorted(rows, key=lambda r: r.get('validation_macro_f1', -1), reverse=True)
        with (output / 'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['model', 'returncode', 'trial_budget', 'validation_macro_f1', 'trial', 'parameters'])
            writer.writeheader()
            writer.writerows(ranked)
        print(f'Finished {name}: {row}', flush=True)
    complete = all(row['returncode'] == 0 for row in rows)
    (output / 'status.json').write_text(json.dumps({'state': 'complete' if complete else 'failed', 'results': ranked, 'test_evaluated': False}, indent=2) + '\n')
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
