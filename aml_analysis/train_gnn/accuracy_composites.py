#!/usr/bin/env python3
"""Render accumulated GNN accuracy figures without test accuracy.

Run directly with --folder, or through run_pipeline.py --stage accumulate.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd

MODELS = ('GCN', 'GATv2', 'GraphSAGE', 'GraphTransformer', 'PNA')
SERIES = (('tr_acc', 'Training accuracy (mean ± SD)', '#0072B2'),
          ('val_acc', 'Validation accuracy (mean ± SD)', '#D55E00'))


def plot_accuracy_composite(folder, models=None, dpi=200):
    """Use the stage's epoch-wise means/stds, retaining its variable run counts."""
    output = Path(folder) / 'accumulated_results'
    selected = [model for model in MODELS if models is None or model in models]
    if not selected:
        return
    tables = {}
    for model in selected:
        tables[model] = []
        for key, label, color in SERIES:
            path = output / 'data' / f'{model}_{key}_mean_std.csv'
            if path.is_file():
                tables[model].append((pd.read_csv(path), label, color))
        if not tables[model]:
            raise FileNotFoundError(f'No accumulated training/validation accuracy for {model} in {output}')
    plots = output / 'plots'
    plots.mkdir(parents=True, exist_ok=True)

    # Fixed scales allow direct comparison across models and cohorts.
    last_epoch = max(frame['epoch'].max() for entries in tables.values()
                     for frame, _, _ in entries)
    style = {
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.titleweight': 'semibold',
        'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 9, 'axes.linewidth': .7,
        'axes.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#333333', 'text.color': '#222222',
        'axes.labelcolor': '#222222', 'xtick.color': '#333333',
        'ytick.color': '#333333', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'axes.grid': False,
    }

    legend_entries = {model: [] for model in selected}

    def draw(ax, model, panel=None):
        legend_entries[model] = []
        for frame, label, color in tables[model]:
            epochs = frame['epoch'].to_numpy()
            mean = frame['mean'].to_numpy()
            std = frame['std'].to_numpy()
            # Combine the mean line and its SD band into one legend entry.
            line, = ax.plot(epochs, mean, color=color, linewidth=1.5,
                    linestyle='-', zorder=3)
            band = ax.fill_between(epochs, mean-std, mean+std, alpha=.22,
                            color=color, linewidth=0, zorder=2)
            legend_entries[model].append(((band, line), label))
        ax.set_title(f'{panel}. {model}' if panel is not None else model,
                     pad=8, loc='center', fontweight='normal')
        ax.set(xlim=(0, last_epoch), ylim=(0, 1.02))
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
        ax.tick_params(direction='out', length=3, width=.7)
        ax.grid(axis='y', color='#E5E5E5', linewidth=.5, zorder=0)

    def draw_sd(ax, model):
        for frame, _, color in tables[model]:
            ax.plot(frame['epoch'], frame['std'], color=color, linewidth=1)
        # Shared across models and cohorts for direct SD comparison.
        ax.set_ylim(0, .06)
        ax.set_ylabel('SD', fontsize=8, labelpad=3)
        ax.yaxis.set_major_locator(MultipleLocator(.03))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
        ax.tick_params(labelsize=7, direction='out', length=2, width=.6)
        ax.grid(axis='y', color='#E5E5E5', linewidth=.5)

    resolution = max(600, dpi)
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(8.4, 6.6))
        fig.suptitle('AML: Training and validation accuracy across epochs', fontsize=12, y=.985)
        # Center the two lower panels instead of leaving an empty sixth panel.
        grid = fig.add_gridspec(2, 6, left=.09, right=.985, bottom=.19,
                               top=.90, wspace=1.6, hspace=.55)
        positions = ((0, 0), (0, 2), (0, 4), (1, 1), (1, 3))
        try:
            axes = []
            for index, (model, (row, col)) in enumerate(zip(selected, positions)):
                cell = grid[row, col:col+2].subgridspec(2, 1, height_ratios=(3, 1), hspace=.12)
                ax = fig.add_subplot(cell[0])
                draw(ax, model, chr(65+index))
                ax.tick_params(labelbottom=False)
                sd_ax = fig.add_subplot(cell[1], sharex=ax)
                draw_sd(sd_ax, model)
                axes.append(ax)
            fig.supxlabel('Epoch', x=.53, y=.105, fontsize=10)
            fig.supylabel('Accuracy', x=.012, y=.56, fontsize=10)
            handles, labels = zip(*legend_entries[selected[0]])
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .012),
                       ncol=2, frameon=False, handlelength=2.8, columnspacing=2)
            for extension in ('pdf', 'svg', 'png'):
                fig.savefig(plots / f'GNN_accuracy_mean_std_composite.{extension}',
                            dpi=resolution, facecolor='white')
        finally:
            plt.close(fig)
        for model in selected:
            fig, (ax, sd_ax) = plt.subplots(2, 1, figsize=(4.5, 4.0), sharex=True,
                                                gridspec_kw={'height_ratios': (3, 1), 'hspace': .15})
            try:
                draw(ax, model)
                ax.set_ylabel('Accuracy')
                draw_sd(sd_ax, model)
                sd_ax.set_xlabel('Epoch')
                handles, labels = zip(*legend_entries[model])
                ax.legend(handles, labels, loc='lower right', frameon=False, fontsize=7)
                fig.subplots_adjust(left=.16, right=.97, bottom=.13, top=.91)
                fig.savefig(plots / f'{model}_accuracy_mean_std.pdf',
                            dpi=resolution, bbox_inches='tight', facecolor='white')
            finally:
                plt.close(fig)
    counts = []
    for model in selected:
        sizes = pd.concat([frame['n_runs'] for frame, _, _ in tables[model]])
        low, high = int(sizes.min()), int(sizes.max())
        counts.append(f'{model}: n={low}' if low == high else f'{model}: n={low}–{high}')
    caption = (
        'Training and validation accuracy across epochs for five graph neural network models. '
        'Solid blue lines show mean training accuracy; solid vermilion lines show mean '
        'validation accuracy. Shaded regions represent ±1 population standard deviation '
        'across available archived runs at each epoch, not confidence intervals. '
        'The lower panel beneath each accuracy plot shows training and validation SD '
        'directly, using the same colors and a common SD axis range of 0–0.06 across models and cohorts. '
        'Runs contribute until their recorded history ends; no extrapolation is applied. '
        'Test accuracy is omitted. Axes use common scales. Run counts per epoch: '
        + '; '.join(counts) + '.\n'
    )
    (plots / f'GNN_accuracy_mean_std_composite_caption.txt').write_text(caption)
    return plots / f'GNN_accuracy_mean_std_composite.pdf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    print(plot_accuracy_composite(args.folder))


if __name__ == '__main__':
    main()
