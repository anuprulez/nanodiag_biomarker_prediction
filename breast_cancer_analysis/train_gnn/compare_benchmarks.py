"""Merge tabular/GNN summaries only when dataset hashes and seeds agree.

Rows are sorted by validation performance. Test scores never determine ordering.
Legacy GNN summaries need their original adjacent input_hashes.json file.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def compare(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    expected = None
    pairs = set()
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text())
        hashes = data.get('input_hashes')
        if hashes is None:
            sidecar = path.parent / 'input_hashes.json'
            if not sidecar.exists():
                raise ValueError(f'Cannot verify dataset identity for {path}; rerun with the current runner')
            hashes = json.loads(sidecar.read_text())
        required = {'df_nebit_dnam_features_bc_PNA.csv', 'node_split_manifest_bc.csv', 'out_genes_bc.csv', 'data_bc_PNA.pt'}
        if not required <= hashes.keys():
            raise ValueError(f'Incomplete input hashes in {path}')
        hashes = {name: hashes[name] for name in sorted(required)}
        if expected is None:
            expected = hashes
        elif expected != hashes:
            raise ValueError(f'Dataset snapshot mismatch: {path}. Do not compare different data/graphs/splits.')
        seed = data['config']['seed']
        for failure in data.get('failures', []):
            failures.append(dict(failure, seed=seed, source=str(path)))
        for result in data['results']:
            pair = (result['model'], seed)
            if pair in pairs:
                raise ValueError(f'Duplicate model/seed {pair}; do not count repeat runs as independent seeds')
            pairs.add(pair)
            rows.append(dict(model=result['model'], seed=seed, variant=result['variant'],
                             val_accuracy=result['validation']['accuracy'],
                             val_macro_f1=result['validation']['macro_f1'],
                             test_accuracy=result['test']['accuracy'],
                             test_macro_f1=result['test']['macro_f1'],
                             test_balanced_accuracy=result['test']['balanced_accuracy'],
                             seconds=result.get('seconds'), source=str(path)))
    frame = pd.DataFrame(rows)
    if not rows:
        raise ValueError('No completed models to compare; inspect per-model error logs')
    # Comparing across exactly the same seeds prevents selective reporting.
    seed_sets = frame.groupby('model')['seed'].apply(lambda values: tuple(sorted(values)))
    if len(set(seed_sets)) > 1:
        raise ValueError(f'Models have different completed seed sets: {seed_sets.to_dict()}. Resolve failures first.')
    frame = frame.sort_values(['seed', 'val_accuracy', 'val_macro_f1'], ascending=[True, False, False])
    aggregated = []
    for model, group in frame.groupby('model'):
        row = {'model': model, 'runs': len(group), 'seeds': ','.join(map(str, sorted(group.seed)))}
        for metric in ('val_accuracy', 'val_macro_f1', 'test_accuracy', 'test_macro_f1', 'test_balanced_accuracy'):
            row[metric + '_mean'] = float(group[metric].mean())
            row[metric + '_std'] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.
        aggregated.append(row)
    table = pd.DataFrame(aggregated).sort_values(['val_accuracy_mean', 'val_macro_f1_mean'], ascending=False)
    frame.to_csv(output / 'runs.csv', index=False)
    table.to_csv(output / 'comparison.csv', index=False)
    (output / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    text = ['# Fixed-split model comparison', '', 'Sorted by validation accuracy, then validation macro F1. Values are mean ± sample SD across the same seeds; SD is shown as zero for single-seed runs.', '',
            '| Model | Runs | Validation accuracy | Test accuracy | Test macro F1 |',
            '|---|---:|---:|---:|---:|']
    for _, row in table.iterrows():
        text.append(f"| {row['model']} | {row['runs']} | {row['val_accuracy_mean']:.2%} ± {row['val_accuracy_std']:.2%} | {row['test_accuracy_mean']:.2%} ± {row['test_accuracy_std']:.2%} | {row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_std']:.4f} |")
    if failures:
        text += ['', '**Incomplete benchmark:** failed models are listed in failures.json; they have no performance score.']
    text += ['', 'All inputs have identical SHA256 hashes. Models use the same node partitions and available features; GNNs additionally consume graph edges. Ranked features use the entire unlabeled graph cohort in each partition. Labels are seed-informed propagated targets, not independent clinical labels.']
    (output / 'COMPARISON.md').write_text('\n'.join(text) + '\n')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.summaries, args.output).to_string(index=False))

if __name__ == '__main__':
    main()
