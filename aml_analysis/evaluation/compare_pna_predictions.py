#!/usr/bin/env python3
"""Compare PNA-predicted CpGs' AML-minus-healthy effects across two cohorts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA = ROOT / 'data_corr_0.33_knn_n_3000_n_neg_10000'
EXTERNAL_DELTA = 'Delta_beta_AML_minus_healthy'
EXTERNAL_STATS = ['Healthy_GSE40279_mean_beta', 'Healthy_GSE40279_SD', 'Healthy_GSE40279_n',
                  'AML_GSE62298_mean_beta', 'AML_GSE62298_SD', 'AML_GSE62298_n', EXTERNAL_DELTA]
GROUPS = [('negative', 'Predicted negative', 'pred_negatives_PNA.csv', '#4c72b0'),
          ('likely_positive', 'Predicted likely positive', 'pred_likely_pos_aml_PNA.csv', '#dd8452')]


# Match plot_aggregations.py: talk context, deep blue/orange, 24/16/14 pt text.
TITLE_FONT = 24
AXIS_FONT = 16
ANNOTATION_FONT = 14
PLOT_DPI = 300


def set_plot_style():
    sns.set_theme(style='whitegrid', context='talk', font='DejaVu Sans', rc={
        'axes.titlesize': TITLE_FONT, 'axes.titlepad': 12,
        'axes.labelsize': AXIS_FONT, 'xtick.labelsize': AXIS_FONT,
        'ytick.labelsize': AXIS_FONT, 'legend.fontsize': ANNOTATION_FONT,
        'figure.titlesize': TITLE_FONT, 'axes.spines.top': False,
        'axes.spines.right': False, 'grid.linestyle': '--',
        'grid.linewidth': 0.7, 'grid.alpha': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_predictions(path, group):
    frame = pd.read_csv(path, sep='\t')
    if 'test_gene_names' not in frame or frame['test_gene_names'].duplicated().any():
        raise ValueError(f'{path}: require unique test_gene_names')
    names = frame['test_gene_names'].str.split('_', n=1, expand=True)
    if names.shape[1] != 2 or not names[0].str.fullmatch(r'cg\d+').all() or names[1].isna().any():
        raise ValueError(f'{path}: invalid CpG-gene identifiers')
    frame.insert(0, 'prediction_group', group)
    frame.insert(1, 'CpG_ID', names[0])
    frame.insert(2, 'Predicted_gene', names[1])
    return frame


def collapse_external(frame):
    """One beta statistic per CpG; preserve the full gene annotation separately."""
    if frame[['CpG_ID', 'Gene']].duplicated().any():
        raise ValueError('External comparison contains duplicate CpG-gene pairs')
    unique = frame.groupby('CpG_ID', sort=False)[EXTERNAL_STATS].nunique(dropna=False)
    if unique.gt(1).any().any():
        raise ValueError('Different external beta statistics for gene mappings of the same CpG')
    genes = frame.groupby('CpG_ID')['Gene'].agg(lambda values: ';'.join(sorted(set(values.dropna()))))
    result = frame.drop_duplicates('CpG_ID')[['CpG_ID'] + EXTERNAL_STATS].copy()
    result['External_genes'] = result['CpG_ID'].map(genes)
    return result


def internal_statistics(predictions, signal_path, metadata_path, annotated_path):
    """Use exported beta signals, checking their original sample order explicitly."""
    metadata = pd.read_csv(metadata_path, sep='\t')
    sample_ids = pd.read_csv(annotated_path, sep='\t', nrows=0).columns[1:].tolist()
    if metadata['sample_id'].tolist() != sample_ids or metadata['sample_id'].duplicated().any():
        raise ValueError('Metadata sample order differs from original annotated matrix')
    features = sorted(set(pd.concat(predictions)['test_gene_names']))
    signals = pd.read_csv(signal_path, sep='\t', usecols=features)
    if len(signals) != len(metadata):
        raise ValueError('Exported signal rows differ from metadata sample count')
    if not np.isfinite(signals.to_numpy()).all() or not signals.ge(0).all().all() or not signals.le(1).all().all():
        raise ValueError('Internal signals must be finite beta values between zero and one')
    healthy = metadata['group'].eq('healthy_control').to_numpy()
    aml = metadata['group'].eq('AML').to_numpy()
    if not healthy.any() or not aml.any() or not np.all(healthy | aml):
        raise ValueError('Expected healthy_control and AML sample groups')
    # Confirm each CpG's multiple gene mappings refer to identical beta signals.
    by_probe = {}
    for feature in features:
        probe = feature.split('_', 1)[0]
        if probe in by_probe and not np.array_equal(signals[feature], signals[by_probe[probe]]):
            raise ValueError(f'Conflicting internal beta values for {probe}')
        by_probe.setdefault(probe, feature)
    output = pd.DataFrame({'CpG_ID': list(by_probe), 'Internal_feature_used': list(by_probe.values())})
    columns = output['Internal_feature_used'].tolist()
    output['Internal_healthy_mean_beta'] = signals.loc[healthy, columns].mean().to_numpy()
    output['Internal_AML_mean_beta'] = signals.loc[aml, columns].mean().to_numpy()
    output['Internal_delta_beta_AML_minus_healthy'] = output['Internal_AML_mean_beta'] - output['Internal_healthy_mean_beta']
    output['Internal_healthy_n'] = int(healthy.sum())
    output['Internal_AML_n'] = int(aml.sum())
    return output


def match_predictions(frame, external, internal):
    merged = frame.merge(internal, on='CpG_ID', how='left', validate='many_to_one')
    merged = merged.merge(external, on='CpG_ID', how='left', validate='many_to_one', indicator=True)
    merged['Matched_external_CpG'] = merged.pop('_merge').eq('both')
    merged['Predicted_gene_in_external_annotation'] = [
        bool(isinstance(genes, str) and gene in genes.split(';'))
        for gene, genes in zip(merged['Predicted_gene'], merged['External_genes'])]
    merged['External_minus_internal_delta_beta'] = merged[EXTERNAL_DELTA] - merged['Internal_delta_beta_AML_minus_healthy']
    return merged


def output_path(output, filename, suffix):
    path = Path(filename)
    return output / f'{path.stem}_{suffix}{path.suffix}'


def save_plot(fig, output, name, suffix):
    fig.savefig(output_path(output, f'{name}.png', suffix), dpi=PLOT_DPI, bbox_inches='tight')
    fig.savefig(output_path(output, f'{name}.pdf', suffix), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--external', type=Path, default=HERE / 'results/cpg_gene_beta_comparison.tsv')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'raw/harmonised/sample_metadata.tsv')
    parser.add_argument('--annotated-matrix', type=Path, default=ROOT / 'raw/harmonised/combined_beta_cpg_gene.tsv.gz')
    parser.add_argument('--output', type=Path, default=HERE / 'results/pna_prediction_comparison')
    args = parser.parse_args()
    args.data = args.data.resolve()
    suffix = args.data.name.removeprefix('data_')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    signal_path = args.data / 'preprocessed/combined_pos_neg_signals_aml.csv'
    inputs = [args.external, args.metadata, signal_path]
    predictions = []
    for group, label, filename, color in GROUPS:
        path = args.data / 'generated' / filename
        frame = read_predictions(path, group)
        frame.to_csv(output_path(output, f'{group}_extracted_cpg_genes.tsv', suffix), sep='\t', index=False)
        predictions.append(frame)
        inputs.append(path)
    internal = internal_statistics(predictions, signal_path, args.metadata, args.annotated_matrix)
    external_pairs = pd.read_csv(args.external, sep='\t', dtype={'CpG_ID': str, 'Gene': str})
    external = collapse_external(external_pairs)
    matched, plot_data, summary = {}, {}, []
    for (group, label, filename, color), frame in zip(GROUPS, predictions):
        joined = match_predictions(frame, external, internal)
        joined.to_csv(output_path(output, f'{group}_all_predictions.tsv', suffix), sep='\t', index=False)
        common = joined.loc[joined['Matched_external_CpG']].copy()
        common.to_csv(output_path(output, f'{group}_common_cpg_gene_matches.tsv', suffix), sep='\t', index=False)
        unique = common.drop_duplicates('CpG_ID').copy()
        genes = common.groupby('CpG_ID')['Predicted_gene'].agg(lambda values: ';'.join(sorted(set(values))))
        unique['Predicted_genes'] = unique['CpG_ID'].map(genes)
        # Per-CpG tables should not retain one arbitrary gene's prediction score.
        columns = ['prediction_group', 'CpG_ID', 'Predicted_genes', 'External_genes',
                   'Internal_healthy_mean_beta', 'Internal_AML_mean_beta',
                   'Internal_delta_beta_AML_minus_healthy', 'Internal_healthy_n', 'Internal_AML_n'] + EXTERNAL_STATS + ['External_minus_internal_delta_beta']
        unique = unique[columns].sort_values('CpG_ID')
        unique.to_csv(output_path(output, f'{group}_common_unique_cpgs.tsv', suffix), sep='\t', index=False)
        finite = np.isfinite(unique[['Internal_delta_beta_AML_minus_healthy', EXTERNAL_DELTA]].to_numpy()).all(axis=1)
        valid = unique.loc[finite].copy()
        if len(valid) < 2:
            raise ValueError(f'{label}: fewer than two finite common CpGs')
        x = valid['Internal_delta_beta_AML_minus_healthy'].to_numpy()
        y = valid[EXTERNAL_DELTA].to_numpy()
        varied = np.ptp(x) > 0 and np.ptp(y) > 0
        row = dict(prediction_group=group, predicted_features=len(frame), predicted_unique_cpgs=frame['CpG_ID'].nunique(),
                   common_features=len(common), common_unique_cpgs=len(unique), plotted_unique_cpgs=len(valid),
                   excluded_missing_delta=int((~finite).sum()),
                   internal_mean_delta_beta=float(x.mean()), external_mean_delta_beta=float(y.mean()),
                   mean_external_minus_internal_delta_beta=float((y-x).mean()),
                   median_external_minus_internal_delta_beta=float(np.median(y-x)),
                   internal_median_absolute_delta_beta=float(np.median(np.abs(x))),
                   external_median_absolute_delta_beta=float(np.median(np.abs(y))),
                   same_direction_fraction=float(np.mean(np.sign(x) == np.sign(y))),
                   pearson_r=float(pearsonr(x,y).statistic) if varied else None,
                   spearman_r=float(spearmanr(x,y).statistic) if varied else None)
        summary.append(row)
        matched[group], plot_data[group] = unique, valid
    pd.DataFrame(summary).to_csv(output_path(output, 'summary.tsv', suffix), sep='\t', index=False)
    overlap = sorted(set(matched['negative']['CpG_ID']) & set(matched['likely_positive']['CpG_ID']))
    pd.DataFrame({'CpG_ID': overlap}).to_csv(output_path(output, 'cpgs_in_both_prediction_groups.tsv', suffix), sep='\t', index=False)
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(17, 14), constrained_layout=True)
    scatter_values = np.concatenate([df[['Internal_delta_beta_AML_minus_healthy', EXTERNAL_DELTA]].to_numpy().ravel() for df in plot_data.values()])
    limit = min(1.02, max(0.15, float(np.abs(scatter_values).max()) * 1.12))
    rng = np.random.default_rng(32)
    for column, ((group, label, filename, color), row) in enumerate(zip(GROUPS, summary)):
        frame = plot_data[group]
        x = frame['Internal_delta_beta_AML_minus_healthy'].to_numpy()
        y = frame[EXTERNAL_DELTA].to_numpy()
        ax = axes[0, column]
        ax.scatter(x, y, s=24, alpha=0.6, color=color, edgecolors='none', rasterized=True)
        ax.plot([-limit,limit],[-limit,limit],ls='--',lw=2,color='#555555',label='Equal effect')
        ax.axhline(0,color='#BBBBBB',lw=.7); ax.axvline(0,color='#BBBBBB',lw=.7)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='Original cohort Δβ (AML − healthy)',
               ylabel='External cohort Δβ (AML − healthy)',title=f'{label}\n{len(frame):,} common CpGs')
        ax.set_aspect('equal',adjustable='box')
        correlation = f"Pearson r = {row['pearson_r']:.3f}" if row['pearson_r'] is not None else 'Pearson r undefined'
        ax.text(.04,.96,correlation,transform=ax.transAxes,va='top',fontsize=ANNOTATION_FONT,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':4})
        ax = axes[1,column]
        ax.boxplot([x,y],positions=[1,2],widths=.45,showfliers=False,patch_artist=True,
                   boxprops={'facecolor':color,'edgecolor':color,'alpha':.3,'linewidth':1.5},
                   medianprops={'color':'black','linewidth':1.8},
                   whiskerprops={'color':color,'linewidth':1.4},capprops={'color':color,'linewidth':1.4})
        for position, values in [(1,x),(2,y)]:
            ax.scatter(position+rng.uniform(-.14,.14,len(values)),values,s=13,alpha=.3,color=color,edgecolors='none',rasterized=True)
        ax.axhline(0,color='#BBBBBB',lw=.8)
        ax.set(xticks=[1,2],xticklabels=['Original cohort','External cohort'],ylabel='Δβ (AML − healthy)',
               title='Effects at matched CpGs',ylim=(-limit,limit))
    fig.suptitle('PNA prediction groups\nMethylation effects in original and external cohorts',fontsize=TITLE_FONT)
    save_plot(fig,output,'delta_beta_original_vs_external',suffix)
    fig, axes = plt.subplots(1,2,figsize=(17,6.5),constrained_layout=True)
    for ax,column,title in [(axes[0],'External_minus_internal_delta_beta','Change in effect between cohorts'),
                            (axes[1],EXTERNAL_DELTA,'Absolute external methylation effect')]:
        for position,(group,label,filename,color) in enumerate(GROUPS,1):
            values=plot_data[group][column].to_numpy()
            if column==EXTERNAL_DELTA:
                values=np.abs(values)
            box=ax.boxplot([values],positions=[position],widths=.5,showfliers=False,patch_artist=True,
                           medianprops={'color':'black','linewidth':1.8},
                           whiskerprops={'color':color,'linewidth':1.4},capprops={'color':color,'linewidth':1.4})
            box['boxes'][0].set(facecolor=color,edgecolor=color,linewidth=1.5,alpha=.3)
            ax.scatter(position+rng.uniform(-.15,.15,len(values)),values,color=color,alpha=.35,s=14,edgecolors='none',rasterized=True)
        ax.axhline(0,color='#999999',lw=.8)
        ax.set(xticks=[1,2],xticklabels=['Predicted\nnegative','Predicted\nlikely positive'],title=title,
               ylabel='External Δβ − original Δβ' if column!='Delta_beta_AML_minus_healthy' else '|External Δβ|')
    save_plot(fig,output,'delta_beta_shift_by_prediction_group',suffix)
    report=dict(data_directory=str(args.data), output_suffix=suffix, delta_definition='mean beta in AML minus mean beta in healthy controls',
                delta_shift_definition='external delta beta minus original delta beta',
                original_samples={'healthy':int(internal['Internal_healthy_n'].iloc[0]),'AML':int(internal['Internal_AML_n'].iloc[0])},
                external_cohorts={'healthy':'GSE40279','AML':'GSE62298'},
                plot_style={'reference':'train_gnn/plot_aggregations.py','theme':'whitegrid','context':'talk',
                            'title_font':TITLE_FONT,'axis_tick_font':AXIS_FONT,'annotation_font':ANNOTATION_FONT,
                            'negative_color':'#4c72b0','likely_positive_color':'#dd8452','png_dpi':PLOT_DPI},
                match_key='CpG_ID only; gene annotations retained separately',
                plotting_unit='unique CpG within each prediction group; no duplicated points for multiple genes',
                cpgs_in_both_prediction_groups=len(overlap),groups=summary,
                input_sha256={str(path.resolve()):digest(path) for path in inputs},
                sample_order_check='Exported signals aligned to metadata after verifying original annotated matrix sample-ID header',
                limitations='Descriptive cross-study effects; no adjustment for batch, age, sex or cell composition; predicted classes are model outputs, not external ground truth.')
    output_path(output, 'analysis_report.json', suffix).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# PNA prediction groups versus external cohorts','',
           'Δβ = mean(AML beta) − mean(healthy beta). ΔΔβ = external Δβ − original Δβ.',
           f"Original cohort: {report['original_samples']['healthy']} healthy controls and {report['original_samples']['AML']} AML samples. External cohort: GSE40279 healthy controls and GSE62298 AML.",
           'CpGs are matched by exact CpG_ID. Multiple gene mappings count once per CpG in plots and summary statistics.','',
           '| Prediction group | Input CpG-gene rows | Input unique CpGs | Common unique CpGs | Pearson r | Median absolute external Δβ |',
           '|---|---:|---:|---:|---:|---:|']
    for (group,label,filename,color),row in zip(GROUPS,summary):
        r=f"{row['pearson_r']:.4f}" if row['pearson_r'] is not None else 'undefined'
        lines.append(f"| {label} | {row['predicted_features']} | {row['predicted_unique_cpgs']} | {row['common_unique_cpgs']} | {r} | {row['external_median_absolute_delta_beta']:.4f} |")
    lines += ['',f'Common CpGs assigned to both prediction groups: {len(overlap)}. Such CpGs remain in both group panels; the groups are not assumed independent.',
              '', f'Plots: `delta_beta_original_vs_external_{suffix}.png` and `delta_beta_shift_by_prediction_group_{suffix}.png`, with PDF versions.',
              'Tables retain all extracted features, all matching CpG-gene rows, unique matched CpGs, and unmatched predictions (Matched_external_CpG=false).',
              '',report['limitations'],'',
              f'Regenerate from aml_analysis with `python evaluation/compare_pna_predictions.py --data {args.data}`.']
    output_path(output, 'README.md', suffix).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'suffix':suffix,'groups':summary,'cpgs_in_both_prediction_groups':len(overlap)},indent=2))


if __name__=='__main__':
    main()
