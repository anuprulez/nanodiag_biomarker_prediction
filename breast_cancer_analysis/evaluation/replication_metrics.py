"""Exploratory cross-study effect agreement, Welch tests and cohort provenance.

Welch inference uses published sample means, sample SDs (ddof=1), and per-site n.
It cannot separate disease from study when cases and controls are different studies.
"""
import gzip
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind_from_stats

INTERNAL = 'Internal_delta_beta_BC_minus_healthy'
EXTERNAL = 'Delta_beta_BC_minus_healthy'
EFFECT_THRESHOLD = .10
FDR_THRESHOLD = .05


def bh_adjust(pvalues):
    """BH over all supplied sites, conservatively retaining untestable sites as p=1."""
    p = np.asarray(pvalues, dtype=float)
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError('P-values must be in [0,1]')
    safe = np.where(np.isfinite(p), p, 1.)
    order = np.argsort(safe)
    q = np.empty(len(p))
    q[order] = np.minimum(1, np.minimum.accumulate((safe[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    q[~np.isfinite(p)] = np.nan
    return q


def external_inference(frame):
    """One test per unique CpG; apply BH before selecting model predictions."""
    if frame.CpG_ID.duplicated().any():
        raise ValueError('FDR universe must contain unique CpG IDs')
    result = frame.copy()
    h, c = 'Healthy_GSE101961', 'BC_GSE226569'
    hm, hs, hn, cm, cs, cn = [pd.to_numeric(frame[f'{prefix}_{field}'], errors='raise').to_numpy(float)
                             for prefix in (h, c) for field in ('mean_beta', 'SD', 'n')]
    valid = (np.isfinite(np.column_stack([hm, hs, hn, cm, cs, cn])).all(axis=1)
             & (hn >= 2) & (cn >= 2) & (hn == np.floor(hn)) & (cn == np.floor(cn))
             & (hs >= 0) & (cs >= 0) & (hm >= 0) & (hm <= 1) & (cm >= 0) & (cm <= 1))
    delta = cm - hm
    recorded = frame[EXTERNAL].to_numpy(float)
    if not np.allclose(delta[valid], recorded[valid], atol=1e-10, rtol=1e-8):
        raise ValueError('External delta does not agree with group means')
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = cs**2 / cn, hs**2 / hn
        se = np.sqrt(a+b)
        df = (a+b)**2 / (a*a/(cn-1) + b*b/(hn-1))
    valid &= (se > 0) & np.isfinite(df)
    p = np.full(len(frame), np.nan)
    low, high = p.copy(), p.copy()
    p[valid] = ttest_ind_from_stats(cm[valid], cs[valid], cn[valid], hm[valid], hs[valid], hn[valid], equal_var=False).pvalue
    margin = t.ppf(.975, df[valid]) * se[valid]
    low[valid], high[valid] = delta[valid] - margin, delta[valid] + margin
    result['External_p_value'] = p
    result['External_FDR'] = bh_adjust(p)
    result['External_CI95_low'], result['External_CI95_high'] = low, high
    result['External_testable'] = valid
    return result


def annotate_candidates(frame):
    result = frame.copy()
    x, y = result[INTERNAL], result[EXTERNAL]
    finite = np.isfinite(x) & np.isfinite(y)
    result['Direction_concordant'] = finite & (np.sign(x) == np.sign(y))
    result['Internal_absolute_delta_beta'] = x.abs()
    result['External_absolute_delta_beta'] = y.abs()
    result['Absolute_delta_error'] = (y-x).abs()
    result['External_significant'] = result.External_FDR.lt(FDR_THRESHOLD)
    result['Meets_exploratory_criterion'] = (result.External_significant & y.abs().ge(EFFECT_THRESHOLD)
                                              & result.Direction_concordant & x.ne(0) & y.ne(0))
    return result


def agreement_metrics(frame):
    x, y = frame[INTERNAL].to_numpy(), frame[EXTERNAL].to_numpy()
    metrics = {'mean_absolute_error': float(np.mean(np.abs(y-x))),
               'external_significant_fraction': float(frame.External_significant.mean()),
               'external_testable_cpgs': int(frame.External_testable.sum()),
               'external_testable_fraction': float(frame.External_testable.mean()),
               'exploratory_criterion_count': int(frame.Meets_exploratory_criterion.sum()),
               'exploratory_criterion_fraction': float(frame.Meets_exploratory_criterion.mean())}
    for cohort, values in [('internal', x), ('external', y)]:
        metrics[f'{cohort}_delta_beta_SD'] = float(np.std(values, ddof=1))
        for threshold in (.10, .20):
            metrics[f'{cohort}_absolute_delta_gt_{threshold:.2f}_fraction'] = float(np.mean(np.abs(values) > threshold))
    return metrics


def sensitivity_table(plot_data):
    rows = []
    for group, frame in plot_data.items():
        for effect in (.05, .10, .20):
            for fdr in (.01, .05, .10):
                accepted = (frame[EXTERNAL].abs().ge(effect) & frame.External_FDR.lt(fdr)
                            & frame.Direction_concordant & frame[INTERNAL].ne(0) & frame[EXTERNAL].ne(0))
                rows.append(dict(prediction_group=group, effect_threshold=effect, FDR_threshold=fdr,
                                 denominator=len(frame), count=int(accepted.sum()), fraction=float(accepted.mean())))
    return pd.DataFrame(rows)


def cohort_audit(matrix, training_series, root, raw_dir):
    """Check actual sample identifiers; do not equate distinct accessions to distinct patients."""
    from geo_common import read_header
    samples = pd.read_csv(matrix, sep='\t', nrows=0).columns[1:]
    discovery = {match.group() for sample in samples if (match := re.search(r'GSM\d+', sample))}
    ext = {}
    checked = []
    for series in ('GSE101961', 'GSE226569'):
        path = raw_dir / f'{series}_series_matrix.txt.gz'
        if path.is_file():
            with gzip.open(path, 'rt') as handle:
                ids, _ = read_header(handle)
            ext[series] = set(ids)
            checked.append(path)
    overlap = {series: sorted(discovery & ids) for series, ids in ext.items()}
    provenance = {}
    import json
    for filename in ('preparation_report.json', 'meinter_seed_report.json'):
        path = Path(matrix).parent / filename
        if path.is_file():
            provenance[filename] = json.loads(path.read_text())
            checked.append(path)
    return dict(training_series=training_series,
                discovery_GSM_identifiers=len(discovery), discovery_sample_count=len(samples),
                GSM_overlap_with_external=overlap,
                accession_check_complete=len(discovery)==len(samples) and len(ext)==2,
                patient_level_independence='Unverified: different GEO sample accessions cannot exclude reused patients',
                disease_study_confounding=True,
                same_study_external_replication='Not available in current inputs; GSE66695 is discovery and cannot also be external validation',
                external_use_in_seeds_or_tuning='Local seed provenance recorded below; full tuning/input independence is not established by these files',
                local_seed_provenance=provenance,
                QC='Published processing; tumor beta columns exclude detection-p columns but are not detection-p filtered. No common renormalization, batch or cell-composition adjustment.',
                interpretation='Exploratory cross-study evidence; no claim of independent biological validation'), checked


def validation_plots(plot_data, summary, groups, original_axis):
    import matplotlib.pyplot as plt
    # Use the largest existing axis-label size consistently across panels.
    from matplotlib.font_manager import FontProperties
    label_size = max(16., FontProperties(size=plt.rcParams['axes.labelsize']).get_size_in_points())
    font_family = 'DejaVu Sans'
    if len(groups)==2:
        fig, grid = plt.subplots(2, 2, figsize=(24, 19), constrained_layout=True)
        axes = grid.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    values = np.concatenate([frame[[INTERNAL, EXTERNAL]].to_numpy().ravel() for frame in plot_data.values()])
    limit = min(1.02, max(.15, float(np.abs(values).max())*1.12))
    for column, ((group, label, _, color), row) in enumerate(zip(groups, summary)):
        frame = plot_data[group]
        selected = frame.Meets_exploratory_criterion
        ax = axes[column]
        ax.scatter(frame.loc[~selected, INTERNAL], frame.loc[~selected, EXTERNAL], s=22, alpha=.45,
                   color=color, label='Does not meet criterion', rasterized=True)
        ax.scatter(frame.loc[selected, INTERNAL], frame.loc[selected, EXTERNAL], s=45, alpha=.8,
                   marker='^', color=color, edgecolors='black', linewidths=.5,
                   label='Meets external cohort criterion', rasterized=True)
        ax.plot([-limit, limit], [-limit, limit], '--', color='#666666', linewidth=1)
        ax.axhline(0, color='#BBBBBB', linewidth=.7); ax.axvline(0, color='#BBBBBB', linewidth=.7)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel=f'{original_axis} Δβ',
               ylabel='External cohort Δβ', title=f'{chr(65+column)}. {label} (n={len(frame):,})')
        ax.set_aspect('equal')
        pearson = f"{row['pearson_r']:.3f}" if row['pearson_r'] is not None else 'undefined'
        spearman = f"{row['spearman_r']:.3f}" if row['spearman_r'] is not None else 'undefined'
        ax.text(.03, .97, f"Pearson r={pearson}; Spearman ρ={spearman}\nDirection={row['same_direction_fraction']:.1%}; MAE={row['mean_absolute_error']:.3f}",
                transform=ax.transAxes, va='top', fontsize=27,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        ax.legend(loc='lower right', fontsize=15)
    ax = axes[len(groups)]
    rng = np.random.default_rng(32)
    for i, (group, label, _, color) in enumerate(groups):
        frame = plot_data[group]
        for j, (cohort, field) in enumerate([('Original', INTERNAL), ('External', EXTERNAL)]):
            position = i*3+j+1
            data = frame[field].abs().to_numpy()
            box = ax.boxplot([data], positions=[position], widths=.5, patch_artist=True, showfliers=False)
            box['boxes'][0].set(facecolor=color, alpha=.45 if j==0 else .85)
            ax.scatter(position+rng.uniform(-.13,.13,len(data)), data, color=color, alpha=.2, s=9, rasterized=True)
    positions = [i*3+j+1 for i in range(len(groups)) for j in range(2)]
    labels = [label.replace('Predicted ', '')+'\n'+cohort for _,label,_,_ in groups
              for cohort in ('original','external')]
    ax.set(xticks=positions, xticklabels=labels,
           ylabel='|Δβ|', title=f'{chr(65+len(groups))}. Absolute methylation effects')
    ax.tick_params(axis='x', labelsize=12)
    ax = axes[len(groups)+1]
    rates = [100*row['exploratory_criterion_fraction'] for row in summary]
    bars = ax.bar(range(len(groups)), rates, color=[g[3] for g in groups])
    for bar, row in zip(bars, summary):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f"{row['exploratory_criterion_count']}/{row['plotted_unique_cpgs']}\n{100*row['exploratory_criterion_fraction']:.1f}%", ha='center', fontsize=18)
    ax.set(xticks=range(len(groups)), xticklabels=[label.replace('Predicted ', 'Predicted\n') for _,label,_,_ in groups], ylim=(0,110),
           ylabel='Common CpGs meeting criterion (%)', title=f'{chr(66+len(groups))}. External cohort support')
    ax.text(.5, -.24, '|External Δβ| ≥ 0.10, BH FDR < 0.05, consistent nonzero direction',
            transform=ax.transAxes, ha='center', fontsize=15)
    for panel in axes:
        for axis in (panel.xaxis, panel.yaxis):
            axis.label.set(fontsize=label_size, fontfamily=font_family,
                           fontstyle='normal', fontweight='normal')
            for tick in axis.get_ticklabels():
                # Enlarge numeric ticks by 50%; keep category labels consistent.
                text = tick.get_text().replace('−', '-')
                try:
                    float(text)
                    size = label_size * 1.5
                except ValueError:
                    size = label_size
                tick.set(fontsize=size, fontfamily=font_family,
                         fontstyle='normal', fontweight='normal')
            axis.get_offset_text().set(fontsize=label_size * 1.5, fontfamily=font_family)
        panel.title.set(fontsize=24, fontfamily=font_family)
    fig.suptitle('External cohort support for likely positive predicted CpG sites',
                 fontsize=36, fontfamily=font_family)
    return fig


def sensitivity_plot(table, groups):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(groups), figsize=(7*len(groups),6), constrained_layout=True, squeeze=False)
    for ax, (group, label, _, _) in zip(axes[0], groups):
        grid = table.loc[table.prediction_group.eq(group)].pivot(index='effect_threshold', columns='FDR_threshold', values='fraction')*100
        ax.imshow(grid, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        for i in range(len(grid)):
            for j in range(len(grid.columns)):
                value = grid.iloc[i,j]
                ax.text(j,i,f'{value:.1f}%',ha='center',va='center',color='white' if value>60 else 'black')
        ax.set(xticks=range(len(grid.columns)), xticklabels=[f'{v:g}' for v in grid.columns],
               yticks=range(len(grid)), yticklabels=[f'{v:g}' for v in grid.index],
               xlabel='BH FDR cutoff (<)', ylabel='Minimum |external Δβ| (≥)', title=label)
    fig.suptitle('Exploratory criterion sensitivity: percentage of matched CpGs\nAll criteria require consistent nonzero direction', fontsize=18)
    return fig
