#!/usr/bin/env python3
"""Run the AML preprocessing and/or model training with per-stage logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'preprocess', 'train'], default='all')
    parser.add_argument('--config', type=Path, default=HERE / 'config/config.yaml')
    args = parser.parse_args()
    stages = [('preprocess', HERE / 'preprocess_raw_data/process_raw_illumina_arrays_aml.py'),
              ('train', HERE / 'train_gnn/main.py')]
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '4'
    env['MPLBACKEND'] = 'Agg'
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    for stage, script in stages:
        if args.stage not in ('all', stage):
            continue
        log = logs / (stage + '.log')
        print(f'Running {stage}; log: {log}', flush=True)
        with log.open('w') as handle:
            completed = subprocess.run([sys.executable, '-u', str(script), '--config', str(args.config.resolve())],
                                       cwd=HERE, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise SystemExit(f'{stage} failed (exit {completed.returncode}); see {log}')
        if stage == 'preprocess':
            subprocess.run([sys.executable, str(HERE / 'preprocess_raw_data/validate_aml_outputs.py'),
                            '--config', str(args.config.resolve())], cwd=HERE, env=env, check=True)
        print(f'{stage} complete', flush=True)


if __name__ == '__main__':
    main()
