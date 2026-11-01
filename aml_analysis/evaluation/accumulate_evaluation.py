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

from replication_metrics import (INTERNAL, EXTERNAL, EFFECT_THRESHOLD, annotate_candidates,
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
    for column in ('Predicted_genes', 'External_genes'):
        if all(column in table for table in aligned):
            result[column] = [
                ';'.join(sorted({gene.strip() for table in aligned
                                 for value in [table.at[cpg, column]] if pd.notna(value)
                                 for gene in str(value).split(';') if gene.strip()}))
                for cpg in result.index]
    result['n_runs'] = len(tables)
    return annotate_candidates(result.reset_index())


def export_top_likely_positive_cpgs(frame, output, suffix):
    """Export absolute-effect rankings of the exact five-run plot intersection."""
    if not frame['n_runs'].eq(5).all():
        raise ValueError('Top-20 common CpG tables require exactly five runs')
    frame = annotate_candidates(frame)
    frame['Associated_genes'] = [
        ';'.join(sorted({gene.strip() for value in values if pd.notna(value)
                         for gene in str(value).split(';') if gene.strip()}))
        for values in zip(frame.get('Predicted_genes', pd.Series('', index=frame.index)),
                          frame.get('External_genes', pd.Series('', index=frame.index)))]
    frame['External_abs_delta_beta_ge_0_10'] = frame[EXTERNAL].abs().ge(EFFECT_THRESHOLD)
    frame['Same_nonzero_direction'] = (frame.Direction_concordant
                                       & frame[INTERNAL].ne(0) & frame[EXTERNAL].ne(0))
    finite = frame.loc[np.isfinite(frame[[INTERNAL, EXTERNAL]]).all(axis=1)]
    paths = []
    for cohort, primary, secondary in (('external', EXTERNAL, INTERNAL),
                                        ('internal', INTERNAL, EXTERNAL)):
        ranked = finite.sort_values([primary, secondary, 'CpG_ID'],
                                    key=lambda values: values if values.name == 'CpG_ID' else values.abs(),
                                    ascending=[False, False, True]).head(20)
        columns = ['CpG_ID', 'Associated_genes', INTERNAL, EXTERNAL, 'Internal_absolute_delta_beta',
                   'External_absolute_delta_beta', 'n_runs',
                   'External_FDR', 'External_abs_delta_beta_ge_0_10',
                   'External_significant', 'Same_nonzero_direction',
                   'Meets_exploratory_criterion']
        result = ranked[columns].rename(columns={INTERNAL: 'Internal_delta_beta',
                                                EXTERNAL: 'External_delta_beta'}).copy()
        result.insert(0, 'Rank', range(1, len(result)+1))
        path = Path(output)/f'top20_likely_positive_common_5_runs_by_{cohort}_delta_beta_{suffix}.tsv'
        result.to_csv(path, sep='\t', index=False)
        paths.append(path)
        print(f'Top {len(result)} common likely-positive CpGs ranked by |{cohort} Δβ| -> {path}', flush=True)
    return paths


def top20_correlation_plot(table, model, ranked_by='external'):
    """Publication scatter of the selected top 20; retain signed effects."""
    if ranked_by not in ('external', 'internal'):
        raise ValueError('ranked_by must be external or internal')
    from matplotlib.ticker import MaxNLocator
    x = table.Internal_delta_beta.to_numpy(float)
    y = table.External_delta_beta.to_numpy(float)
    varied = len(table) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
    correlation = float(pearsonr(x, y).statistic) if varied else None
    limit = min(1.02, max(.15, float(np.abs(np.r_[x, y]).max())*1.12)) if len(table) else .15
    selected = table.Meets_exploratory_criterion.to_numpy(bool)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'axes.titleweight': 'normal'}):
        fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
        ax.scatter(x[~selected], y[~selected], s=22, alpha=.45, color='#dd8452',
                   label='Does not meet criterion', rasterized=True)
        ax.scatter(x[selected], y[selected], s=45, alpha=.8, marker='^',
                   color='#dd8452', edgecolors='black', linewidths=.5,
                   label='Meets external cohort\ncriterion', rasterized=True)
        ax.plot([-limit, limit], [-limit, limit], '--', color='#666666', linewidth=1)
        ax.axhline(0, color='#BBBBBB', linewidth=.7)
        ax.axvline(0, color='#BBBBBB', linewidth=.7)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
               xlabel='Mean original cohort Δβ', ylabel='External cohort Δβ')
        ax.set_title(f'{model}: top 20 likely-positive CpGs by |{ranked_by} Δβ| (n={len(table)})',
                     fontsize=24, fontweight='normal', loc='center', pad=16, wrap=True)
        ax.set_aspect('equal')
        value = f'{correlation:.3f}' if correlation is not None else 'undefined'
        ax.text(.03, .97, f'Pearson r={value}', transform=ax.transAxes,
                va='top', fontsize=24,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        ax.legend(loc='lower right', fontsize=22, markerscale=1.5,
                  frameon=True, facecolor='white', edgecolor='none', framealpha=.95)
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dddddd', linewidth=.6, alpha=.65)
        ax.grid(axis='x', visible=False)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=5))
            axis.label.set(fontsize=24, fontfamily='DejaVu Sans')
            for tick in axis.get_ticklabels():
                tick.set(fontsize=24, fontfamily='DejaVu Sans')
        ax.tick_params(axis='both', pad=8)
    return fig


def export_within_criterion_cpgs(frame, output, suffix, effect_threshold):
    """Export every common likely-positive CpG within this plot's criterion."""
    candidates = annotate_candidates(frame, effect_threshold=effect_threshold, effect_cohort='external',
                                      fdr_threshold=.01, p_value_threshold=.01)
    selected = candidates.loc[candidates.Meets_exploratory_criterion].copy()
    selected['Associated_genes'] = [
        ';'.join(sorted({gene.strip() for value in values if pd.notna(value)
                         for gene in str(value).split(';') if gene.strip()}))
        for values in zip(selected.get('Predicted_genes', pd.Series('', index=selected.index)),
                          selected.get('External_genes', pd.Series('', index=selected.index)))]
    selected['Same_nonzero_direction'] = (selected.Direction_concordant
                                         & selected[INTERNAL].ne(0) & selected[EXTERNAL].ne(0))
    selected['External_absolute_delta_beta_threshold'] = effect_threshold
    selected['External_p_value_threshold'] = .01
    selected['External_FDR_threshold'] = .01
    selected['External_abs_delta_beta_ge_threshold'] = selected[EXTERNAL].abs().ge(effect_threshold)
    selected = selected.sort_values(['External_absolute_delta_beta', 'Internal_absolute_delta_beta', 'CpG_ID'],
                                    ascending=[False, False, True])
    columns = ['CpG_ID', 'Associated_genes', INTERNAL, EXTERNAL,
               'Internal_absolute_delta_beta', 'External_absolute_delta_beta', 'n_runs',
               'External_p_value', 'External_FDR', 'External_p_value_threshold',
               'External_FDR_threshold', 'External_absolute_delta_beta_threshold',
               'External_abs_delta_beta_ge_threshold', 'External_significant',
               'Same_nonzero_direction', 'Meets_exploratory_criterion']
    result = selected[columns].rename(columns={INTERNAL: 'Internal_delta_beta',
                                              EXTERNAL: 'External_delta_beta'}).copy()
    result.insert(0, 'Rank', range(1, len(result)+1))
    path = Path(output)/f'likely_positive_within_criterion_abs_external_ge_{effect_threshold:.2f}_{suffix}.tsv'
    result.to_csv(path, sep='\t', index=False)
    print(f'{len(result)} likely-positive CpGs within criterion -> {path}', flush=True)
    return path


def threshold_delta_beta_composite(data, model, n_runs, effect_threshold):
    """Recompute scatter markers and support counts for a specified effect cutoff."""
    threshold_data, summary = {}, []
    for group, _, _, _ in GROUPS:
        frame = annotate_candidates(data[group], effect_threshold=effect_threshold, effect_cohort='external',
                                      fdr_threshold=.01, p_value_threshold=.01)
        threshold_data[group] = frame
        x, y = frame[INTERNAL].to_numpy(), frame[EXTERNAL].to_numpy()
        varied = len(frame) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
        summary.append(dict(prediction_group=group, plotted_unique_cpgs=len(frame),
                            external_absolute_delta_beta_threshold=effect_threshold,
                            external_FDR_threshold=.01, external_p_value_threshold=.01,
                            pearson_r=float(pearsonr(x, y).statistic) if varied else None,
                            spearman_r=float(spearmanr(x, y).statistic) if varied else None,
                            same_direction_fraction=float((np.sign(x)==np.sign(y)).mean()),
                            **agreement_metrics(frame)))
    fig = validation_plots(threshold_data, summary, GROUPS,
                           'Mean original cohort', publication=True)
    fig.suptitle(f'{model}: external cohort support across {n_runs} runs\n'
                 f'Criterion: |External Δβ| ≥ {effect_threshold:.2f}; P < 0.01; FDR < 0.01; same nonzero direction',
                 fontsize=30, fontfamily='DejaVu Sans', fontweight='normal')
    return fig, pd.DataFrame(summary)


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
            publication = name.startswith('delta_beta_original_vs_external')
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
            if group == 'likely_positive' and len(included) == 5:
                top_paths = export_top_likely_positive_cpgs(frame, output, suffix)
                for cohort, top_path in zip(('external', 'internal'), top_paths):
                    top_table = pd.read_csv(top_path, sep='\t')
                    save(top20_correlation_plot(top_table, model, ranked_by=cohort),
                         f'delta_beta_original_vs_external_top20_{cohort}')
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
        for effect_threshold in (.10, .30):
            export_within_criterion_cpgs(data['likely_positive'], output, suffix, effect_threshold)
            threshold_fig, threshold_summary = threshold_delta_beta_composite(
                data, model, len(included), effect_threshold)
            stem = f'delta_beta_original_vs_external_abs_external_ge_{effect_threshold:.2f}'
            save(threshold_fig, stem)
            threshold_summary.to_csv(output/f'{stem}_summary_{suffix}.tsv', sep='\t', index=False)
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
