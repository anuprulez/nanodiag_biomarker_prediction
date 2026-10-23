#!/usr/bin/env python3
"""Compare model-predicted CpGs' BC-minus-healthy effects across two cohorts."""
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
EXTERNAL_DELTA = 'Delta_beta_BC_minus_healthy'
EXTERNAL_STATS = ['Healthy_GSE101961_mean_beta', 'Healthy_GSE101961_SD', 'Healthy_GSE101961_n',
                  'BC_GSE226569_mean_beta', 'BC_GSE226569_SD', 'BC_GSE226569_n', EXTERNAL_DELTA]
GROUPS = [('negative', 'Predicted negative', 'pred_negatives_PNA.csv', '#4c72b0'),
          ('likely_positive', 'Predicted likely positive', 'pred_likely_pos_bc_PNA.csv', '#dd8452')]


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


def internal_statistics(predictions, signal_path, matrix_path):
    """Verify exported beta rows against original BC/NC sample columns and signals."""
    import csv
    sample_ids = pd.read_csv(matrix_path, sep='\t', nrows=0).columns[1:].tolist()
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Duplicate original sample identifiers')
    cancer = np.array([name.upper().startswith('BC_') for name in sample_ids])
    healthy = np.array([name.upper().startswith('NC_') for name in sample_ids])
    if not healthy.any() or not cancer.any() or not np.all(healthy | cancer):
        raise ValueError('Expected original BC_ and NC_ sample identifiers')
    features = sorted(set(pd.concat(predictions)['test_gene_names']))
    if not features:
        raise ValueError('No predicted features')
    signals = pd.read_csv(signal_path, sep='\t', usecols=features)
    if len(signals) != len(sample_ids):
        raise ValueError('Exported signals differ from original sample count')
    values = signals.to_numpy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Exported signals must be finite beta values in [0,1]')
    by_probe = {}
    for feature in features:
        probe = feature.split('_', 1)[0]
        if probe in by_probe and not np.array_equal(signals[feature], signals[by_probe[probe]]):
            raise ValueError(f'Conflicting beta values for {probe}')
        by_probe.setdefault(probe,feature)
    remaining = set(by_probe)
    # Confirm actual values, not just row count: this detects permuted samples.
    import gzip
    opener = gzip.open if Path(matrix_path).suffix == '.gz' else open
    with opener(matrix_path, 'rt') as source:
        reader = csv.reader(source,delimiter='\t')
        next(reader)
        for row in reader:
            if row[0] not in remaining:
                continue
            exported = signals[by_probe[row[0]]].to_numpy()
            original = np.asarray(row[1:],dtype=float)
            if original.shape != exported.shape or not np.allclose(original,exported,rtol=0,atol=1e-12):
                raise ValueError(f'Exported sample order/beta values differ from original matrix: {row[0]}')
            remaining.remove(row[0])
            if not remaining:
                break
    if remaining:
        raise ValueError(f'Predicted CpGs absent from original matrix: {sorted(remaining)[:5]}')
    output = pd.DataFrame({'CpG_ID':list(by_probe),'Internal_feature_used':list(by_probe.values())})
    columns = output['Internal_feature_used'].tolist()
    output['Internal_healthy_mean_beta'] = signals.loc[healthy,columns].mean().to_numpy()
    output['Internal_BC_mean_beta'] = signals.loc[cancer,columns].mean().to_numpy()
    output['Internal_delta_beta_BC_minus_healthy'] = output['Internal_BC_mean_beta'] - output['Internal_healthy_mean_beta']
    output['Internal_healthy_n'] = int(healthy.sum())
    output['Internal_BC_n'] = int(cancer.sum())
    return output


def match_predictions(frame, external, internal):
    merged = frame.merge(internal, on='CpG_ID', how='left', validate='many_to_one')
    merged = merged.merge(external, on='CpG_ID', how='left', validate='many_to_one', indicator=True)
    merged['Matched_external_CpG'] = merged.pop('_merge').eq('both')
    merged['Predicted_gene_in_external_annotation'] = [
        bool(isinstance(genes, str) and gene in genes.split(';'))
        for gene, genes in zip(merged['Predicted_gene'], merged['External_genes'])]
    merged['External_minus_internal_delta_beta'] = merged[EXTERNAL_DELTA] - merged['Internal_delta_beta_BC_minus_healthy']
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
    parser.add_argument('--matrix', type=Path, default=ROOT / 'raw/GSE237036_matrix_processed.txt')
    parser.add_argument('--model', default='PNA', choices=['PNA','GCN','GraphSAGE','GATv2','GraphTransformer','TabPFN','TabICL','XGBoost','CatBoost'])
    parser.add_argument('--generated-dir', type=Path, help='Override model artifact directory; selected improved PNA is detected automatically')
    parser.add_argument('--output', type=Path, default=HERE / 'results/prediction_comparison')
    args = parser.parse_args()
    args.data = args.data.resolve()
    import re
    match = re.search(r'GSE\d+', args.matrix.name)
    training_series = match.group(0) if match else 'GSE237036'
    original_is_pbmc = training_series == 'GSE237036'
    original_axis = 'PBMC' if original_is_pbmc else 'Original breast tissue'
    original_tissue = 'PBMCs in BC patients and normal controls' if original_is_pbmc else 'breast tumors and normal breast tissue'
    original_platform = 'GPL23976 (EPIC 850K)' if original_is_pbmc else 'GPL16304 (450K)' if training_series == 'GSE69914' else 'GPL13534 (450K)'
    interpretation = ('Exploratory cross-tissue comparison, not matched-tissue replication of a PBMC biomarker signature'
                      if original_is_pbmc else 'Exploratory comparison of independent breast-tissue cohorts; matching platform does not remove study effects')
    limitations = ('PBMC training effects are compared with breast tissue effects on a different array platform. '
                   if original_is_pbmc else 'Breast tissue effects are compared across separate studies and processing protocols. ')
    limitations += 'Tissue composition, study, processing and tumor purity can change effect direction and size. No batch, demographic or cell-composition adjustment. Propagated priority classes are not ground-truth differential methylation labels; internal classification accuracy does not establish biological replication.'
    global GROUPS
    GROUPS = [(g,l,f.replace('PNA',args.model),c) for g,l,f,c in GROUPS]
    suffix = args.model + '_' + args.data.name.removeprefix('data_')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    signal_path = args.data / 'preprocessed/combined_pos_neg_signals_bc.csv'
    generated = args.generated_dir.resolve() if args.generated_dir else args.data / 'generated'
    improved = args.data / 'generated/pna_accuracy_improved'
    if args.generated_dir is None and args.model == 'PNA' and (improved / 'improvement_report.json').is_file():
        generated = improved
    inputs = [args.external, args.matrix, signal_path]
    # Rebuild groups from saved inference IDs, avoiding assumptions about test order.
    from extract_predictions import extract
    extracted = output / ('extracted_' + suffix)
    extract(args.data, args.model, extracted, generated_dir=generated)
    for prefix in ['test_ids','true_labels','pred_labels','best_class_pred_probs']:
        inputs.append(generated / f'{prefix}_bc_{args.model}.pt')
    inputs += [args.data / 'preprocessed/out_genes_bc.csv',
               generated / f'test_probe_genes_bc_{args.model}.csv']
    predictions = []
    for group, label, filename, color in GROUPS:
        path = extracted / filename
        frame = read_predictions(path, group)
        frame.to_csv(output_path(output, f'{group}_extracted_cpg_genes.tsv', suffix), sep='\t', index=False)
        predictions.append(frame)
        inputs.append(path)
    internal = internal_statistics(predictions, signal_path, args.matrix)
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
                   'Internal_healthy_mean_beta', 'Internal_BC_mean_beta',
                   'Internal_delta_beta_BC_minus_healthy', 'Internal_healthy_n', 'Internal_BC_n'] + EXTERNAL_STATS + ['External_minus_internal_delta_beta']
        unique = unique[columns].sort_values('CpG_ID')
        unique.to_csv(output_path(output, f'{group}_common_unique_cpgs.tsv', suffix), sep='\t', index=False)
        finite = np.isfinite(unique[['Internal_delta_beta_BC_minus_healthy', EXTERNAL_DELTA]].to_numpy()).all(axis=1)
        valid = unique.loc[finite].copy()
        if len(valid) < 2:
            raise ValueError(f'{label}: fewer than two finite common CpGs')
        x = valid['Internal_delta_beta_BC_minus_healthy'].to_numpy()
        y = valid[EXTERNAL_DELTA].to_numpy()
        varied = np.ptp(x) > 0 and np.ptp(y) > 0
        row = dict(prediction_group=group, predicted_features=len(frame), predicted_unique_cpgs=frame['CpG_ID'].nunique(),
                   common_features=len(common), common_unique_cpgs=len(unique),
                   common_cpg_fraction=float(len(unique)/frame['CpG_ID'].nunique()) if len(frame) else 0.0, plotted_unique_cpgs=len(valid),
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
    scatter_values = np.concatenate([df[['Internal_delta_beta_BC_minus_healthy', EXTERNAL_DELTA]].to_numpy().ravel() for df in plot_data.values()])
    limit = min(1.02, max(0.15, float(np.abs(scatter_values).max()) * 1.12))
    rng = np.random.default_rng(32)
    for column, ((group, label, filename, color), row) in enumerate(zip(GROUPS, summary)):
        frame = plot_data[group]
        x = frame['Internal_delta_beta_BC_minus_healthy'].to_numpy()
        y = frame[EXTERNAL_DELTA].to_numpy()
        ax = axes[0, column]
        ax.scatter(x, y, s=24, alpha=0.6, color=color, edgecolors='none', rasterized=True)
        ax.plot([-limit,limit],[-limit,limit],ls='--',lw=2,color='#555555',label='Equal effect')
        ax.axhline(0,color='#BBBBBB',lw=.7); ax.axvline(0,color='#BBBBBB',lw=.7)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel=f'{original_axis} Δβ (case − control)',
               ylabel='Breast tissue Δβ (tumor − normal)',title=f'{label}\n{len(frame):,} common CpGs')
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
        ax.set(xticks=[1,2],xticklabels=['PBMCs' if original_is_pbmc else 'Original breast','External breast'],ylabel='Δβ (case − control)',
               title='Effects at matched CpGs',ylim=(-limit,limit))
    fig.suptitle(f'{args.model}: {original_axis} versus external breast tissue\n' + ('Exploratory comparison across different tissues' if original_is_pbmc else 'Exploratory comparison across independent studies'),fontsize=TITLE_FONT)
    save_plot(fig,output,'delta_beta_original_vs_external',suffix)
    fig, axes = plt.subplots(1,2,figsize=(17,6.5),constrained_layout=True)
    for ax,column,title in [(axes[0],'External_minus_internal_delta_beta','Change in effect across cohorts'),
                            (axes[1],EXTERNAL_DELTA,'Absolute breast tissue methylation effect')]:
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
               ylabel=f'External Δβ − {original_axis} Δβ' if column!='Delta_beta_BC_minus_healthy' else '|Breast tissue Δβ|')
    save_plot(fig,output,'delta_beta_shift_by_prediction_group',suffix)
    report=dict(model=args.model, model_artifact_directory=str(generated), data_directory=str(args.data), output_suffix=suffix, delta_definition='mean beta in BC minus mean beta in healthy controls',
                delta_shift_definition='external delta beta minus original delta beta',
                original_samples={'healthy':int(internal['Internal_healthy_n'].iloc[0]),'BC':int(internal['Internal_BC_n'].iloc[0])},
                external_cohorts={'healthy':'GSE101961','BC':'GSE226569'},
                training_series=training_series,
                cohort_tissues={training_series:original_tissue,
                                'GSE226569':'FFPE breast tumors','GSE101961':'normal breast tissue from cancer-free women'},
                cohort_platforms={training_series:original_platform,'GSE226569':'GPL13534 (450K)','GSE101961':'GPL13534 (450K)'},
                interpretation=interpretation,
                plot_style={'reference':'train_gnn/plot_aggregations.py','theme':'whitegrid','context':'talk',
                            'title_font':TITLE_FONT,'axis_tick_font':AXIS_FONT,'annotation_font':ANNOTATION_FONT,
                            'negative_color':'#4c72b0','likely_positive_color':'#dd8452','png_dpi':PLOT_DPI},
                match_key='CpG_ID only; gene annotations retained separately',
                plotting_unit='unique CpG within each prediction group; no duplicated points for multiple genes',
                cpgs_in_both_prediction_groups=len(overlap),groups=summary,
                input_sha256={str(path.resolve()):digest(path) for path in inputs},
                sample_order_check='Exported beta values checked against original matrix; BC_ and NC_ identifiers define groups',
                limitations=limitations)
    output_path(output, 'analysis_report.json', suffix).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=[f'# {args.model}: {original_axis} versus external breast tissue','',
           interpretation,'',
           'Δβ = mean(BC beta) − mean(healthy beta). ΔΔβ = external Δβ − original Δβ.',
           f"Original {training_series} cohort ({original_tissue}, {original_platform}): {report['original_samples']['healthy']} healthy controls and {report['original_samples']['BC']} BC patients. External breast tissue cohorts (GPL13534): GSE101961 cancer-free normal breast and GSE226569 FFPE breast tumors.",
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
              f'Regenerate from breast_cancer_analysis with `python evaluation/compare_predictions.py --model {args.model} --data {args.data}`.']
    output_path(output, 'README.md', suffix).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'suffix':suffix,'groups':summary,'cpgs_in_both_prediction_groups':len(overlap)},indent=2))


if __name__=='__main__':
    main()
