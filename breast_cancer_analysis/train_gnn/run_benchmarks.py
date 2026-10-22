"""Run all nine model families sequentially and write a shared comparison.

Optional separate Python interpreters keep foundation model dependencies separate
from the project's PyTorch/PyG environment. See BENCHMARKS.md for installation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from benchmark_common import fresh_output
from compare_benchmarks import compare
from pretrained_checkpoints import default_checkpoint

TABULAR = ['TabPFN', 'TabICLv2', 'XGBoost', 'CatBoost']
GNNS = ['PNA', 'GCN', 'GraphSAGE', 'GATv2', 'GraphTransformer']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=TABULAR + GNNS, default=TABULAR + GNNS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[32])
    parser.add_argument('--variants', nargs='+', choices=['original', 'ranked_regularized'], default=['original', 'ranked_regularized'])
    parser.add_argument('--gnn-python', default=sys.executable)
    parser.add_argument('--tabular-python', default=sys.executable)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--tree-iterations', type=int, default=500)
    parser.add_argument('--foundation-estimators', type=int, default=4)
    parser.add_argument('--tabpfn-model-path', default=str(default_checkpoint('TabPFN')))
    parser.add_argument('--tabicl-model-path', default=str(default_checkpoint('TabICLv2')))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error('Seeds and model names must be unique')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('Seeds must be in [0, 2**32)')
    fresh_output(args.output)
    here = Path(__file__).resolve().parent
    summaries, commands, failures = [], [], []
    for seed in args.seeds:
        for group, names, interpreter, script in [
            ('gnn', [m for m in args.models if m in GNNS], args.gnn_python, 'investigate_gnn.py'),
            ('tabular', [m for m in args.models if m in TABULAR], args.tabular_python, 'benchmark_tabular.py')]:
            if not names:
                continue
            out = args.output.resolve() / f'seed_{seed}' / group
            command = [interpreter, str(here / script), '--data', str(args.data.resolve()),
                       '--output', str(out), '--seed', str(seed), '--device', args.device,
                       '--models', *names, '--variants', *args.variants]
            if group == 'gnn':
                command += ['--epochs', str(args.epochs)]
            else:
                command += ['--tree-iterations', str(args.tree_iterations),
                            '--foundation-estimators', str(args.foundation_estimators),
                            '--tabpfn-model-path', args.tabpfn_model_path]
                if args.tabicl_model_path:
                    command += ['--tabicl-model-path', args.tabicl_model_path]
            out.parent.mkdir(parents=True, exist_ok=True)
            log = out.parent / f'{group}.log'
            commands.append(command)
            print(f'Running {group}, seed {seed}: {names}; log={log}', flush=True)
            try:
                with log.open('w') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    failures.append(dict(seed=seed, group=group, models=names, exit_code=result.returncode, log=str(log)))
            except OSError as error:
                failures.append(dict(seed=seed, group=group, models=names, error=str(error)))
            summary = out / 'summary.json'
            if summary.exists():
                summaries.append(summary)
    manifest = dict(commands=commands, failures=failures, summaries=list(map(str, summaries)))
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if summaries:
        try:
            compare(summaries, args.output / 'comparison')
        except ValueError as error:
            print(f'Comparison unavailable: {error}', file=sys.stderr)
            return 1
    else:
        print('No completed summaries; see run_manifest.json and logs', file=sys.stderr)
        return 1
    if failures:
        report = args.output / 'comparison' / 'COMPARISON.md'
        with report.open('a') as stream:
            stream.write('\n**Incomplete benchmark:** one or more runner processes failed; see run_manifest.json and per-run logs.\n')
    print(f'Comparison saved under {args.output / "comparison"}', flush=True)
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(main())
