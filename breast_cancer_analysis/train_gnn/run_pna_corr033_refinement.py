"""Refine PNA on the selected breast-cancer split and retain the better winner."""
import json
import os
from pathlib import Path
import subprocess
import sys
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data_corr_0.33_knn_n_3000_n_neg_10000'
TAG = 'pna_corr033_refine_20261026'


def main():
    output = DATASET / 'logs' / TAG
    output.mkdir(parents=True, exist_ok=True)
    before = OmegaConf.load(DATASET / 'config.yaml')
    OmegaConf.save(before, output / 'config.before_refinement.yaml')
    study = DATASET / 'generated/optuna' / TAG / 'PNA'
    if (study / 'study.sqlite3').exists():
        raise RuntimeError('Refinement study already exists; inspect before resuming')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    command = [sys.executable, '-u', str(ROOT / 'train_gnn/tune_models.py'),
               '--config', str(DATASET / 'preprocessing_config.yaml'),
               '--search-space', str(ROOT / 'config/optuna_data_corr_0.33_knn_n_3000_n_neg_10000_search_space.yaml'),
               '--models', 'PNA', '--study-tag', TAG, '--trials', '25']
    with (output / 'tune.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        return result.returncode
    best = json.loads((study / 'best_params.json').read_text())
    previous = float(before.tuning_results.PNA.validation_score)
    improved = best['validation_score'] > previous
    if not improved:
        current = OmegaConf.load(DATASET / 'config.yaml')
        current.model_overrides.PNA = before.model_overrides.PNA
        current.tuning_results.PNA = before.tuning_results.PNA
        temporary = DATASET / 'config.yaml.part'
        OmegaConf.save(current, temporary)
        temporary.replace(DATASET / 'config.yaml')
    summary = dict(previous_validation_macro_f1=previous,
                   refinement_validation_macro_f1=best['validation_score'],
                   improved=improved, trial=best['trial'], parameters=best['parameters'],
                   test_evaluated=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
