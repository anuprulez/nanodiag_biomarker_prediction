"""Create separate model metric figures from archived runs; preserve existing plots."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'data_corr_0.33_knn_n_3000_n_neg_10000'
METRICS = {
    'best_val_acc': 'Validation accuracy', 'te_acc': 'Test accuracy',
    'te_f1_macro': 'Macro F1', 'te_f1_weighted': 'Weighted F1',
    'te_f1_micro': 'Micro F1', 'te_precision': 'Precision (weighted)',
    'te_recall': 'Recall (weighted)',
}


def create(disease, title):
    base = ROOT / disease / CONFIG
    old = base / 'accumulated_results'
    models = json.loads((base / 'runs/run1/run.json').read_text())['models']
    output = old / 'models_vs_metrics_mean_std'
    (output / 'data').mkdir(parents=True, exist_ok=True)
    (output / 'plots').mkdir(exist_ok=True)
    rows, sources = [], []
    for run_id in range(1, 6):
        run = base / 'runs' / f'run{run_id}'
        manifest = json.loads((run / 'run.json').read_text())
        for model in models:
            assert model in manifest['models'], (run, model)
            paths = list((run / 'generated').rglob(f'all_metrics_{model}.json'))
            assert len(paths) == 1, (run, model, paths)
            path = paths[0]
            values = json.loads(path.read_text())
            sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for metric in METRICS:
                value = values[metric]
                if isinstance(value, list):
                    value = value[-1]
                value = float(value)
                assert np.isfinite(value) and 0 <= value <= 1, (path, metric, value)
                rows.append(dict(model=model, run=f'run{run_id}', metric=metric, value=value))
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['model', 'run', 'metric']).any()
    summary = frame.groupby(['model', 'metric'], sort=False).value.agg(mean='mean', std='std', n_runs='count').reset_index()
    assert summary.n_runs.eq(5).all()
    # Verify that the new means reproduce the existing heatmap's run-level data.
    existing = pd.read_csv(old / 'data/accuracy_metrics_by_run.csv')
    check = frame.merge(existing, on=['model', 'run', 'metric'], suffixes=('_new', '_old'), validate='one_to_one')
    assert len(check) == len(existing)
    assert len(frame) == len(models) * 5 * len(METRICS)
    np.testing.assert_allclose(check.value_new, check.value_old, atol=1e-14, rtol=0)
    frame.to_csv(output / 'data/metrics_by_run.tsv', sep='\t', index=False)
    summary.to_csv(output / 'data/metrics_mean_sample_std.tsv', sep='\t', index=False)
    means = summary.pivot(index='model', columns='metric', values='mean').loc[models, list(METRICS)].to_numpy()
    stds = summary.pivot(index='model', columns='metric', values='std').loc[models, list(METRICS)].to_numpy()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 17, 'axes.titlesize': 21, 'axes.labelsize': 18, 'xtick.labelsize': 17, 'ytick.labelsize': 17})
    fig, ax = plt.subplots(figsize=(20, 10))
    cmap = plt.get_cmap('YlGnBu')
    im = ax.imshow(means, vmin=0, vmax=1, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(METRICS)), list(METRICS.values()), rotation=25, ha='right')
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(METRICS)):
            r, g, b, _ = cmap(means[i, j])
            color = 'black' if .299*r + .587*g + .114*b > .55 else 'white'
            ax.text(j, i, f'{means[i,j]:.3f} ± {stds[i,j]:.3f}', ha='center', va='center', color=color, fontsize=17)
    ax.set_xticks(np.arange(-.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title(f'{title}: Models vs metrics\nMean ± SD across 5 runs', fontsize=27, pad=22)
    fig.colorbar(im, ax=ax, label='Mean metric score', fraction=.035, pad=.025)
    fig.tight_layout()
    save(fig, output, 'Models_vs_metrics_mean_std')
    fig, axes = plt.subplots(2, 4, figsize=(25, 13))
    colors = plt.get_cmap('tab10')(np.arange(len(models)))
    for j, (metric, label) in enumerate(METRICS.items()):
        ax = axes.flat[j]
        for i, model in enumerate(models):
            values = frame.loc[frame.model.eq(model) & frame.metric.eq(metric), 'value'].to_numpy()
            ax.scatter(i + np.linspace(-.1, .1, len(values)), values, color=colors[i], s=17, alpha=.4)
            ax.errorbar(i, means[i, j], yerr=stds[i, j], fmt='o', color=colors[i], capsize=4, markersize=6, linewidth=1.7)
        ax.set_xticks(range(len(models)), models, rotation=55, ha='right', fontsize=14)
        ax.set_ylim(0, 1.04)
        ax.set_title(label)
        ax.set_ylabel('Metric score')
        ax.grid(axis='y', alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .65, 'Large points: mean\nError bars: ±1 sample SD\nFaint points: individual runs\nn = 5 runs per model\nSD uses ddof = 1', va='center', fontsize=19)
    fig.suptitle(f'{title}: Models vs metrics — mean ± SD', fontsize=29)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, output, 'Models_vs_metrics_mean_std_errorbars')
    (output / 'data/validation.json').write_text(json.dumps(dict(status='passed', models=models, metrics=list(METRICS), n_runs=5, sd_ddof=1, matches_existing_run_data=True, sources=sources), indent=2) + '\n')
    (output / 'README.md').write_text(f'# {title}: Models vs metrics\n\nNew figures only; existing figures are preserved. All {len(models)} models and all seven metrics from the existing Models vs Metrics heatmap are included. Each model has five archived runs. Scalars use the saved value; list-valued test accuracy uses its final entry, matching the existing plotting code. Means reproduce the existing run-level data. SD is the sample standard deviation (ddof=1), not the standard error or a confidence interval. Colors show means on a fixed 0–1 scale. Error bars show mean ± one sample SD; faint points show individual runs. No significance claim is inferred from SD overlap.\n\nRun from the repository root: `python post_evaluation_support/plot_models_metrics_mean_std.py`.\n')
    print(title, ':', len(models), 'models;', len(frame), 'verified observations;', output)


def save(fig, output, stem):
    for extension in ('pdf', 'png', 'svg'):
        fig.savefig(output / 'plots' / f'{stem}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    create('breast_cancer_analysis', 'Breast Cancer')
    create('aml_analysis', 'AML')
