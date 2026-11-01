"""Fresh validation-only comparison of all configured models on the selected split."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data_corr_0.33_knn_n_3000_n_neg_10000'


def main():
    tag = 'all_models_fresh_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = DATASET / 'logs' / tag
    output.mkdir(parents=True, exist_ok=False)
    search_path = output / 'search_space.yaml'
    shutil.copy2(ROOT / 'config/optuna.yaml', search_path)
    search = OmegaConf.load(search_path)
    assert search.n_trials == 25
    shutil.copy2(DATASET / 'config.yaml', output / 'config.before_fresh.yaml')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    results = []
    for name in search.models:
        command = [sys.executable, '-u', str(ROOT / 'train_gnn/tune_models.py'),
                   '--config', str(DATASET / 'preprocessing_config.yaml'),
                   '--search-space', str(search_path), '--models', name,
                   '--study-tag', tag, '--trials', '25']
        (output / 'status.json').write_text(json.dumps({'state': 'running', 'model': name, 'tag': tag, 'completed': results}, indent=2) + '\n')
        print(f'Starting {name}: 25 fresh trials; logs: {output}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(model=name, returncode=completed.returncode, trial_budget=25)
        if completed.returncode == 0:
            path = DATASET / 'generated/optuna' / tag / name / 'best_params.json'
            best = json.loads(path.read_text())
            row.update(validation_macro_f1=best['validation_score'], trial=best['trial'])
        results.append(row)
        ranked = sorted(results, key=lambda item: item.get('validation_macro_f1', -1), reverse=True)
        with (output / 'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['model', 'returncode', 'trial_budget', 'validation_macro_f1', 'trial'])
            writer.writeheader()
            writer.writerows(ranked)
        print(f'Finished {name}: {row}', flush=True)
    success = all(item['returncode'] == 0 for item in results)
    (output / 'status.json').write_text(json.dumps({'state': 'complete' if success else 'failed', 'results': ranked, 'test_evaluated': False}, indent=2) + '\n')
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
