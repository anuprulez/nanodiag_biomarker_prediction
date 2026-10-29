#!/usr/bin/env python3
"""Plot dataset probe degree by propagated soft label (displayed as 0-based)."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def degree_statistics(config):
    """Join preprocessed NeDBIT degrees to labels by probe name, once per probe."""
    import numpy as np
    import pandas as pd

    features = pd.read_csv(config.p_nedbit_features, usecols=['name', 'degree'])
    rankings = pd.read_csv(config.p_out_gene_rankings, sep=r'\s+', header=None,
                           usecols=[0, 2], names=['name', 'label'])
    if features.empty or rankings.empty:
        raise ValueError('Probe degrees and soft labels must be nonempty')
    probes = features.merge(rankings, on='name', how='left', validate='one_to_one')
    if probes['label'].isna().any():
        raise ValueError('Missing soft labels for preprocessed probes')
    for column in ('degree', 'label'):
        probes[column] = pd.to_numeric(probes[column], errors='raise')
        if not np.isfinite(probes[column]).all():
            raise ValueError(f'Non-finite {column} values')
    if (probes['degree'] < 0).any():
        raise ValueError('Probe degrees must be nonnegative')
    if not probes['label'].isin(range(1, 6)).all():
        raise ValueError('Expected propagated soft labels 1 through 5')
    probes['soft_label'] = probes['label'].astype(int) - 1
    return probes.groupby('soft_label')['degree'].agg(['count', 'mean', 'median'])


def plot_degree_per_soft_label(config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    summary = degree_statistics(config)
    output = Path(config.p_base) / 'accumulated_results'
    (output / 'data').mkdir(parents=True, exist_ok=True)
    (output / 'plots').mkdir(parents=True, exist_ok=True)
    stem = 'degree_per_soft_label'
    summary.to_csv(output / 'data' / f'{stem}.csv')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 12,
                         'axes.titleweight': 'semibold', 'axes.titlepad': 16,
                         'text.color': '#262626', 'axes.labelcolor': '#262626',
                         'xtick.color': '#262626', 'ytick.color': '#262626',
                         'axes.edgecolor': '#555555', 'axes.linewidth': 0.7,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'xtick.major.size': 0, 'ytick.major.size': 3,
                         'ytick.major.width': 0.7, 'legend.fontsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
        try:
            x = np.arange(len(summary))
            for offset, metric, color, hatch in [(-0.2, 'mean', '#4477AA', ''),
                                                  (0.2, 'median', '#CCBB44', '//')]:
                ax.bar(x + offset, summary[metric], width=0.38,
                       label=metric.title(), color=color, edgecolor='#333333',
                       linewidth=0.55, hatch=hatch, zorder=3)
            ax.set(xticks=x, xticklabels=summary.index.astype(str),
                   xlabel='Soft label', ylabel='Degree', ylim=(0, None),
                   title='AML: Mean and median degree of probes\nper soft label')
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E6E6E6', linewidth=0.6, zorder=0)
            ax.margins(x=0.08)
            peak = float(summary[['mean', 'median']].to_numpy().max())
            ax.set_ylim(0, peak * 1.22 if peak > 0 else 1)
            ax.legend(frameon=False, ncol=2, loc='upper center',
                      handlelength=1.4, columnspacing=1.8)
            for extension in ('pdf', 'svg', 'png'):
                fig.savefig(output / 'plots' / f'{stem}.{extension}', dpi=600,
                            bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
    print(f'Degree per soft label: {output / "plots" / stem}', flush=True)
    return summary


def main():
    from dataset_configs import downstream_configs
    from preprocessing_grid import add_parameter_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/config.yaml')
    parser.add_argument('--folder', type=Path)
    add_parameter_arguments(parser)
    args = parser.parse_args()
    config_path = args.config.resolve()
    os.chdir(ROOT / 'train_gnn')
    for config in downstream_configs(config_path, args):
        plot_degree_per_soft_label(config)


if __name__ == '__main__':
    main()
