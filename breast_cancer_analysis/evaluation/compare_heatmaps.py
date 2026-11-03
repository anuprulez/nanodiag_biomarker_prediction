#!/usr/bin/env python3
"""Compare exported model accuracies across preprocessing folders."""
import argparse
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'^data_corr_([0-9.eE+-]+)_knn_n_(\d+)_n_neg_(\d+)$')
MODEL_ORDER = ('PNA', 'GCN', 'GraphSAGE', 'GATv2', 'GraphTransformer', 'TabPFN', 'TabICL', 'XGBoost', 'CatBoost')
ALIASES = {'GraphSage': 'GraphSAGE', 'GATV2': 'GATv2', 'TabICLv2': 'TabICL'}


def read_folder(folder):
    """Read run1 CSVs and fill omitted cells from run1 model metrics only."""
    folder = Path(folder).resolve()
    match = PATTERN.fullmatch(folder.name)
    if not match:
        raise ValueError(f'Cannot infer preprocessing parameters from {folder.name}')
    candidates = []
    for relative in ('runs/run1/generated/data', 'runs/run1/generated'):
        for path in (folder/relative).glob('Heatmap_avg_edges_*.csv'):
            table = pd.read_csv(path)
            if not {'model', 'best_val_acc', 'te_acc'}.issubset(table):
                continue
            table = table[['model', 'best_val_acc', 'te_acc']].copy()
            table['model'] = table.model.replace(ALIASES)
            if table.model.duplicated().any():
                raise ValueError(f'{path}: duplicate model rows')
            coverage = int(table[['best_val_acc', 'te_acc']].notna().any(axis=1).sum())
            candidates.append((coverage, path.stat().st_mtime_ns, str(path), table))
    rows = {}

    def add(model, values, source):
        row = rows.setdefault(model, dict(model=model, best_val_acc=np.nan, te_acc=np.nan,
                                         validation_source_file='', test_source_file=''))
        for key, source_key in (('best_val_acc', 'validation_source_file'), ('te_acc', 'test_source_file')):
            value = values.get(key)
            if value is None or pd.isna(value):
                continue
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{source}: {key} must lie in [0, 1]')
            if pd.isna(row[key]):
                row[key] = value
                row[source_key] = str(source)

    for _, _, source, table in sorted(candidates, key=lambda item: item[:3], reverse=True):
        for row in table.to_dict('records'):
            add(row['model'], row, source)
    metric_files = sorted((folder/'runs/run1/generated').rglob('all_metrics_*.json'))
    for path in metric_files:
        model = path.stem.removeprefix('all_metrics_')
        model = ALIASES.get(model, model)
        metrics = json.loads(path.read_text())
        validation = metrics.get('best_val_acc')
        if validation is None:
            validation = metrics.get('val_acc')
            if isinstance(validation, list):
                finite = [float(value) for value in validation if value is not None and np.isfinite(float(value))]
                validation = max(finite) if finite else None
        test = metrics.get('te_acc')
        if isinstance(test, list):
            test = test[-1] if test else None
        add(model, {'best_val_acc': validation, 'te_acc': test}, path)
    if not rows:
        raise FileNotFoundError(f'{folder}: no run1 heatmap CSV or model accuracy metrics')
    table = pd.DataFrame(rows.values()).rename(columns={'best_val_acc': 'validation_accuracy', 'te_acc': 'test_accuracy'})
    table['source_file'] = [';'.join(dict.fromkeys(path for path in values if path))
                            for values in zip(table.validation_source_file, table.test_source_file)]
    table.insert(0, 'folder', folder.name)
    table['corr'] = float(match[1])
    table['num_neighbours'] = int(match[2])
    table['num_unknown'] = int(match[3])
    table['source_scope'] = 'run1'
    return table


def order_configurations(configurations):
    """Show correlation sweeps, then neighbour sweeps, then unknown-count sweeps."""
    configs = pd.DataFrame(configurations).copy()
    def group(row):
        if row.num_neighbours == 3000 and row.num_unknown == 10000:
            return 0
        if row.corr == .33 and row.num_unknown == 10000:
            return 1
        if row.corr == .33:
            return 2
        return 3
    configs['configuration_group'] = [group(row) for row in configs.itertuples()]
    configs['_sweep_value'] = [row.corr if row.configuration_group == 0
                               else row.num_neighbours if row.configuration_group == 1
                               else row.num_unknown for row in configs.itertuples()]
    return configs.sort_values(['configuration_group', '_sweep_value', 'corr',
                                'num_neighbours', 'num_unknown']).drop(columns='_sweep_value')


def compare_heatmaps(folders, output, models=None):
    output = Path(output)
    frames, skipped = [], []
    folders = list(dict.fromkeys(Path(folder).resolve() for folder in folders))
    configurations = []
    for folder in folders:
        match = PATTERN.fullmatch(folder.name)
        if not match:
            raise ValueError(f'Cannot infer preprocessing parameters from {folder.name}')
        configurations.append(dict(folder=folder.name, corr=float(match[1]),
                                   num_neighbours=int(match[2]), num_unknown=int(match[3])))
        try:
            frames.append(read_folder(folder))
        except FileNotFoundError as error:
            skipped.append(str(error))
            print(f'Missing run1 data (shown as N/A): {error}', flush=True)
    if not frames:
        raise FileNotFoundError('No usable Heatmap_avg_edges CSV files in the selected folders')
    table = pd.concat(frames, ignore_index=True)
    if models:
        table = table.loc[table.model.isin([ALIASES.get(model, model) for model in models])]
    if table.empty:
        raise ValueError('No selected models occur in the heatmap exports')
    table = table.sort_values(['corr', 'num_neighbours', 'num_unknown', 'model'])
    configs = order_configurations(configurations)
    order = [model for model in MODEL_ORDER if model in set(table.model)]
    order += sorted(set(table.model)-set(order))
    # Keep selected folders without run1 visible, with explicitly missing scores.
    index = pd.MultiIndex.from_product([configs.folder, order], names=['folder', 'model'])
    table = table.set_index(['folder', 'model']).reindex(index).reset_index()
    for key in ('corr', 'num_neighbours', 'num_unknown', 'configuration_group'):
        table[key] = table.folder.map(configs.set_index('folder')[key])
    available_folders = {frame.folder.iloc[0] for frame in frames}
    for index in table.index[table.source_scope.isna()]:
        table.at[index, 'source_scope'] = ('missing_model_in_run1' if table.at[index, 'folder'] in available_folders else 'missing_run1')
    table['missing_reason'] = [
        'No run1 accuracy export for this model' if pd.isna(val) and pd.isna(test)
        else 'No run1 validation accuracy' if pd.isna(val)
        else 'No run1 test accuracy' if pd.isna(test) else ''
        for val, test in zip(table.validation_accuracy, table.test_accuracy)]
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/'validation_test_accuracy_by_preprocessing.tsv', sep='\t', index=False)
    scopes = set(table.source_scope)
    provenance = dict(source_selection='Only run1: merge heatmap CSVs by model coverage/newest, fill missing cells from archived all_metrics JSON; never use other runs or current/accumulated data',
                      skipped_folders=skipped,
                      sources=table[['folder', 'source_file', 'source_scope', 'validation_source_file', 'test_source_file']].drop_duplicates().to_dict('records'))
    (output/'heatmap_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titleweight': 'normal', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, 2, figsize=(max(13, len(order)*1.65), max(5, len(configs)*.58+2.8)),
                                 layout='constrained', sharey=True)
        cmap = plt.colormaps['YlGnBu'].copy();cmap.set_bad('#dddddd')
        try:
            for ax, key, title in zip(axes, ('validation_accuracy', 'test_accuracy'),
                                      ('A. Validation accuracy', 'B. Test accuracy')):
                matrix = table.pivot(index='folder', columns='model', values=key).reindex(index=configs.folder, columns=order)
                matrix.to_csv(output/f'{key}_heatmap.tsv', sep='\t', index_label='folder')
                values = matrix.to_numpy(float)
                image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap, vmin=0, vmax=1)
                ax.set_title(title, loc='center', pad=14, fontsize=16, fontweight='normal')
                ax.set_xticks(range(len(order)), order, rotation=40, ha='right')
                labels = [f'corr={row.corr:g} | neighbours={row.num_neighbours} | unknown={row.num_unknown}' for row in configs.itertuples()]
                ax.set_yticks(range(len(configs)), labels)
                groups = configs.configuration_group.to_numpy()
                for boundary in np.flatnonzero(groups[1:] != groups[:-1]) + .5:
                    ax.axhline(boundary, color='white', linewidth=2)
                ax.set_xlabel('Model')
                for i in range(len(configs)):
                    for j in range(len(order)):
                        value = values[i,j]
                        color = 'white' if np.isfinite(value) and value>.65 else '#222222'
                        ax.text(j, i, f'{value:.3f}' if np.isfinite(value) else 'N/A', ha='center', va='center', color=color, fontsize=10)
                ax.grid(False)
            axes[0].set_ylabel('Preprocessing parameters')
            fig.colorbar(image, ax=axes, label='Accuracy', shrink=.75, ticks=np.linspace(0,1,6))
            cohort = 'AML' if ROOT.name=='aml_analysis' else 'Breast Cancer'
            fig.suptitle(f'{cohort}: Model accuracy across proprocessing configurations', fontsize=18, fontweight='normal')
            for ext in ('pdf', 'svg', 'png'):
                fig.savefig(output/f'validation_test_accuracy_by_preprocessing.{ext}', dpi=300, facecolor='white')
        finally:
            plt.close(fig)
    print(f'Compared {len(configs)} preprocessing folders and {len(order)} models -> {output}', flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders', nargs='+', type=Path)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/preprocessing_comparison')
    args = parser.parse_args()
    folders = args.folders or sorted(ROOT.glob('data_corr_*_knn_n_*_n_neg_*'))
    folders = [folder if folder.is_absolute() else ROOT/folder for folder in folders]
    compare_heatmaps(folders, args.output, args.models)


if __name__ == '__main__':
    main()
