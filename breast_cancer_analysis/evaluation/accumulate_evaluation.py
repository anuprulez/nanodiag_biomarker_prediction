"""Accumulate equal-weight run metrics and effects for CpGs shared by all runs."""
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from replication_metrics import (INTERNAL, EXTERNAL, annotate_candidates,
                                 agreement_metrics, validation_plots, sensitivity_plot)

GROUPS = [('likely_positive', 'Predicted likely positive', '', '#dd8452'),
          ('negative', 'Predicted negative', '', '#4c72b0')]


def average_tables(tables, keys):
    """Each run contributes once; report sample SD and the actual run count."""
    for table in tables:
        if table.duplicated(keys).any():
            raise ValueError('Expected one metric row per key in each evaluation run')
    combined = pd.concat(tables, ignore_index=True)
    numeric = [c for c in combined.select_dtypes(include='number') if c not in keys]
    grouped = combined.groupby(keys, sort=False, dropna=False)
    result = grouped[numeric].agg(['mean', 'std', 'count'])
    result.columns = [f'{column}_{stat}' for column, stat in result.columns]
    result['n_runs'] = grouped.size()
    return result.reset_index()


def common_effects(tables):
    """Intersect within prediction group, then average each CpG once per run.

    External inference is cohort-level: retain it unchanged rather than treating
    repeated training runs as independent external experiments.
    """
    for table in tables:
        if table.CpG_ID.duplicated().any():
            raise ValueError('Expected unique CpGs within each evaluation run')
    common = set(tables[0].CpG_ID).intersection(*(set(t.CpG_ID) for t in tables[1:]))
    aligned = [t.set_index('CpG_ID').loc[sorted(common)] for t in tables]
    result = aligned[0].copy()
    for column in (EXTERNAL, 'External_FDR'):
        for table in aligned[1:]:
            if not np.allclose(result[column], table[column], equal_nan=True):
                raise ValueError(f'External cohort statistics differ across runs: {column}')
    result[INTERNAL] = np.mean([t[INTERNAL].to_numpy(float) for t in aligned], axis=0)
    result['External_minus_internal_delta_beta'] = result[EXTERNAL] - result[INTERNAL]
    result['n_runs'] = len(tables)
    return annotate_candidates(result.reset_index())


def accumulate_evaluation(config, models=None):
    base = Path(config.p_base)
    run_dirs = sorted((p for p in (base/'evaluation').glob('run*')
                       if p.is_dir() and p.name[3:].isdigit()),
                      key=lambda p: int(p.name[3:]))
    # Include training archives so absent evaluation directories are reported.
    run_dirs = sorted(set(run_dirs) | {base/'evaluation'/p.name
                      for p in (base/'runs').glob('run*')
                      if p.is_dir() and p.name[3:].isdigit()},
                      key=lambda p: int(p.name[3:]))
    output = base/'accumulated_results'/'evaluation'
    suffix_tail = base.name.removeprefix('data_')
    for model in list(models or config.model_type):
        suffix = f'{model}_{suffix_tail}'
        required = ['summary', 'criterion_sensitivity'] + [
            f'{g}_common_unique_cpgs' for g, _, _, _ in GROUPS]
        included, missing = [], []
        for run in run_dirs:
            if all((run/f'{name}_{suffix}.tsv').is_file() for name in required):
                included.append(run)
            else:
                missing.append(run.name)
        if not included:
            print(f'{model}: no complete archived evaluation runs; skipped', flush=True)
            continue
        if len(included) != 5 or missing:
            warnings.warn(f'{model}: accumulating {len(included)} evaluation runs; '
                          f'expected 5; incomplete/missing models in {missing}')
        output.mkdir(parents=True, exist_ok=True)
        def read(name):
            return [pd.read_csv(run/f'{name}_{suffix}.tsv', sep='\t') for run in included]
        def save(fig, name):
            publication = name == 'delta_beta_original_vs_external'
            formats = ('png', 'pdf', 'svg') if publication else ('png', 'pdf')
            dpi = max(300, config.get('dpi', 300)) if publication else config.get('dpi', 200)
            with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42,
                                        'svg.fonttype': 'none'}):
                for ext in formats:
                    fig.savefig(output/f'{name}_{suffix}.{ext}', dpi=dpi,
                                bbox_inches='tight', facecolor='white', transparent=False)
            plt.close(fig)
        provenance = dict(model=model, runs=[p.name for p in included],
                          n_runs=len(included), skipped_runs=missing,
                          metrics='Equal-weight mean and sample SD across runs',
                          effects='CpGs common to every included run within each prediction group; mean internal effect; unchanged external inference')
        (output/f'included_runs_{suffix}.json').write_text(json.dumps(provenance, indent=2)+'\n')
        summary = average_tables(read('summary'), ['prediction_group'])
        summary.to_csv(output/f'summary_{suffix}.tsv', sep='\t', index=False)
        sensitivity = average_tables(read('criterion_sensitivity'),
                                     ['prediction_group', 'effect_threshold', 'FDR_threshold'])
        sensitivity.to_csv(output/f'criterion_sensitivity_{suffix}.tsv', sep='\t', index=False)
        mean_sensitivity = sensitivity.rename(columns={f'{c}_mean': c for c in ('fraction', 'count', 'denominator')})
        save(sensitivity_plot(mean_sensitivity, GROUPS), 'criterion_sensitivity')
        data, common_summary = {}, []
        for group, _, _, _ in GROUPS:
            frame = common_effects(read(f'{group}_common_unique_cpgs'))
            frame.to_csv(output/f'{group}_common_unique_cpgs_{suffix}.tsv', sep='\t', index=False)
            frame = frame.loc[np.isfinite(frame[[INTERNAL, EXTERNAL]]).all(axis=1)].copy()
            data[group] = frame
            if frame.empty:
                continue
            x, y = frame[INTERNAL].to_numpy(), frame[EXTERNAL].to_numpy()
            varied = len(frame) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
            common_summary.append(dict(prediction_group=group, plotted_unique_cpgs=len(frame),
                pearson_r=float(pearsonr(x,y).statistic) if varied else None,
                spearman_r=float(spearmanr(x,y).statistic) if varied else None,
                same_direction_fraction=float((np.sign(x)==np.sign(y)).mean()), **agreement_metrics(frame)))
        pd.DataFrame(common_summary).to_csv(output/f'common_cpg_summary_{suffix}.tsv', sep='\t', index=False)
        # Mean support rates use all predictions in each run, not only the intersection.
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ordered = summary.set_index('prediction_group').loc[[g[0] for g in GROUPS]]
        ax.bar(range(len(GROUPS)), ordered.exploratory_criterion_fraction_mean*100,
               yerr=ordered.exploratory_criterion_fraction_std.fillna(0)*100,
               color=[g[3] for g in GROUPS], capsize=5)
        ax.set(xticks=range(len(GROUPS)), xticklabels=[g[1] for g in GROUPS],
               ylabel='Common CpGs meeting criterion (%)',
               title=f'{model}: mean external cohort support ({len(included)} runs; ± SD)')
        ax.set_ylim(bottom=0)
        save(fig, 'delta_beta_criterion')
        if any(frame.empty for frame in data.values()):
            warnings.warn(f'{model}: empty CpG intersection; common-effect plots skipped')
            continue
        fig = validation_plots(data, common_summary, GROUPS, 'Mean original cohort', publication=True)
        fig.suptitle(f'{model}: cumulative external cohort support across {len(included)} runs\n'
                     'CpGs common to all included runs; mean original cohort effects',
                     fontsize=30, fontfamily='DejaVu Sans')
        save(fig, 'delta_beta_original_vs_external')
        print(f'{model}: cumulative four-panel plot ({len(included)} runs) -> '
              f'{output}/delta_beta_original_vs_external_{suffix}.png/pdf/svg', flush=True)
        fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
        for ax, field, label in zip(axes, ('External_minus_internal_delta_beta', EXTERNAL),
                                   ('External Δβ − mean original Δβ', '|External Δβ|')):
            values = [data[g[0]][field].to_numpy() for g in GROUPS]
            if field == EXTERNAL:
                values = [np.abs(v) for v in values]
            ax.boxplot(values, showfliers=False)
            ax.set_xticks(range(1, len(GROUPS)+1), [g[1] for g in GROUPS])
            ax.set_ylabel(label, fontsize=24)
            ax.set_title(f'{model}: CpGs common across {len(included)} runs')
        save(fig, 'delta_beta_shift_by_prediction_group')
