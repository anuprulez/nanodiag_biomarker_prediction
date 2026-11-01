"""Run validation-only PNA tuning for the four selected breast cancer datasets."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TAG = 'pna_selected_20261026'
FOLDERS = [
    'data_corr_0.33_knn_n_4500_n_neg_15000',
    'data_corr_0.33_knn_n_3000_n_neg_10000',
    'data_corr_0.7_knn_n_3000_n_neg_10000',
    'data_corr_0.5_knn_n_3000_n_neg_10000',
]


def main():
    import optuna
    output = ROOT / 'logs' / TAG
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    results = []
    for folder in FOLDERS:
        dataset = ROOT / folder
        study_dir = dataset / 'generated' / 'optuna' / TAG / 'PNA'
        attempted = 0
        if (study_dir / 'study.sqlite3').exists():
            study = optuna.load_study(study_name='PNA', storage='sqlite:///' + str(study_dir / 'study.sqlite3'))
            attempted = sum(t.state.is_finished() for t in study.trials)
            if any(not t.state.is_finished() for t in study.trials):
                raise RuntimeError(f'{folder}: unfinished trials; inspect the study before resuming')
        remaining = max(0, 50 - attempted)
        command = [sys.executable, '-u', str(ROOT / 'train_gnn/tune_models.py'),
                   '--config', str(dataset / 'preprocessing_config.yaml'),
                   '--search-space', str(ROOT / 'config/optuna_pna_selected.yaml'),
                   '--models', 'PNA', '--study-tag', TAG]
        command += ['--trials', str(remaining)] if remaining else ['--export-existing']
        status = {'folder': folder, 'state': 'running', 'remaining_trials': remaining,
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command}
        (output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(f'Starting {folder}: {remaining} additional trials', flush=True)
        with (output / f'{folder}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'folder': folder, 'returncode': result.returncode}
        if result.returncode == 0:
            best = json.loads((study_dir / 'best_params.json').read_text())
            row.update(validation_macro_f1=best['validation_score'], trial=best['trial'],
                       parameters=json.dumps(best['parameters'], sort_keys=True))
        results.append(row)
        with (output / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['folder', 'returncode', 'validation_macro_f1', 'trial', 'parameters'])
            writer.writeheader()
            writer.writerows(results)
        print(f'Finished {folder}: exit={result.returncode}', flush=True)
    state = 'complete' if all(r['returncode'] == 0 for r in results) else 'failed'
    (output / 'status.json').write_text(json.dumps({'state': state, 'results': results}, indent=2) + '\n')
    return 0 if state == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
