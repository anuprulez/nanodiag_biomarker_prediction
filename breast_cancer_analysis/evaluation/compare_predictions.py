#!/usr/bin/env python3
"""Compare model-predicted CpGs' BC-minus-healthy effects across two cohorts."""
import argparse
import sys
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
from replication_metrics import (external_inference, annotate_candidates, agreement_metrics,
                                 sensitivity_table, cohort_audit, validation_plots, sensitivity_plot)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
from artifact_layout import artifact_file
DEFAULT_DATA = ROOT / 'data_corr_0.33_knn_n_3000_n_neg_10000'
EXTERNAL_DELTA = 'Delta_beta_BC_minus_healthy'
EXTERNAL_STATS = ['Healthy_GSE101961_mean_beta', 'Healthy_GSE101961_SD', 'Healthy_GSE101961_n',
                  'BC_GSE226569_mean_beta', 'BC_GSE226569_SD', 'BC_GSE226569_n', EXTERNAL_DELTA]
GROUPS = [('likely_positive', 'Predicted likely positive', 'pred_likely_pos_bc_PNA.csv', '#dd8452'),
          ('negative', 'Predicted negative', 'pred_negatives_PNA.csv', '#4c72b0')]


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



def configured_matrix(config_path):
    """Resolve input paths using the same train_gnn base as pipeline entry points."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    path = Path(config.p_arrays).expanduser()
    return (path if path.is_absolute() else ROOT / 'train_gnn' / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Dataset folder; defaults to the tuned dataset config')
    parser.add_argument('--external', type=Path, default=HERE / 'results/cpg_gene_beta_comparison.tsv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/config.yaml', help='Dataset config, or preprocessing config used to discover tuned dataset configs')
    parser.add_argument('--external-universe', type=Path, help='Full shared-CpG statistics for BH; defaults to shared_cpg_statistics.tsv beside --external')
    parser.add_argument('--matrix', type=Path, help='Override the original training matrix specified by config')
    parser.add_argument('--model', choices=['PNA','GCN','GraphSAGE','GATv2','GraphTransformer','TabPFN','TabICL','XGBoost','CatBoost'])
    parser.add_argument('--generated-dir', type=Path, help='Override model artifact directory; selected improved PNA is detected automatically')
    parser.add_argument('--output', type=Path, help='Default: <dataset>/evaluation/')
    args = parser.parse_args()
    if args.data is None or args.matrix is None:
        import sys
        sys.path.insert(0, str(ROOT))
        from dataset_configs import downstream_configs
        config_path = args.config
        if args.data is not None and (args.data / 'config.yaml').is_file():
            config_path = args.data / 'config.yaml'
        configurations = downstream_configs(config_path)
        if len(configurations) != 1:
            parser.error('Multiple datasets selected; use evaluation/run_comparisons.py or --config <dataset>/config.yaml')
        selected = configurations[0]
        if args.model is None: args.model = str(selected.best_trained_model)
        if args.generated_dir is None and selected.get('config_scope')=='dataset':
            from train_gnn.model_settings import apply_model_settings
            current = apply_model_settings(selected,args.model)
            args.generated_dir = Path(current.p_generated_files)
        if args.data is None:
            args.data = Path(selected.p_base)
            if not args.data.is_absolute(): args.data = ROOT / 'train_gnn' / args.data
        if args.matrix is None:
            args.matrix = Path(selected.p_arrays)
            if not args.matrix.is_absolute(): args.matrix = ROOT / 'train_gnn' / args.matrix
    if args.model is None: args.model = 'PNA'
    args.matrix = args.matrix.resolve()
    if not args.matrix.is_file():
        parser.error(f'Training matrix does not exist: {args.matrix}. Check --config or pass --matrix.')
    args.data = args.data.resolve()
    import re
    match = re.search(r'GSE\d+', args.matrix.name)
    training_series = match.group(0) if match else 'unknown'
    original_is_pbmc = training_series == 'GSE237036'
    original_axis = 'PBMC' if original_is_pbmc else 'Original breast tissue'
    original_tissue = 'PBMCs in BC patients and normal controls' if original_is_pbmc else 'breast tumors and normal breast tissue'
    original_platform = 'GPL23976 (EPIC 850K)' if original_is_pbmc else 'GPL16304 (450K)' if training_series == 'GSE69914' else 'GPL13534 (450K)'
    interpretation = ('Exploratory cross-tissue comparison, not matched-tissue replication of a PBMC biomarker signature'
                      if original_is_pbmc else 'Exploratory cross-study breast-tissue comparison; disease and study are confounded and patient independence is unverified')
    limitations = ('PBMC training effects are compared with breast tissue effects on a different array platform. '
                   if original_is_pbmc else 'Breast tissue effects are compared across separate studies and processing protocols. ')
    limitations += 'External cases and controls come from different studies: disease and batch are confounded. Nominal Welch p-values and BH FDR cannot resolve this confounding; confidence intervals are unadjusted pointwise intervals. Candidate thresholds are exploratory, not prospectively registered. Patient-level independence and all external use in tuning remain unverified. Tissue composition, study, processing and tumor purity can change effect direction and size. No batch, demographic or cell-composition adjustment. Propagated priority classes are not ground-truth differential methylation labels; internal classification accuracy does not establish biological replication.'
    global GROUPS
    GROUPS = [(g,l,f.replace('PNA',args.model),c) for g,l,f,c in GROUPS]
    suffix = args.model + '_' + args.data.name.removeprefix('data_')
    output = args.output.resolve() if args.output else args.data / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    signal_path = args.data / 'preprocessed/combined_pos_neg_signals_bc.csv'
    generated = args.generated_dir.resolve() if args.generated_dir else args.data / 'generated'
    improved = args.data / 'generated/pna_accuracy_improved'
    if args.generated_dir is None and args.model == 'PNA' and artifact_file(improved,'improvement_report.json').is_file():
        generated = improved
    inputs = [args.external, args.matrix, signal_path]
    # Preserve XAI-exported group selections and validate them against inference IDs.
    from extract_predictions import comparison_groups
    extracted = output / ('extracted_' + suffix)
    _, group_sources = comparison_groups(args.data, args.model, extracted, generated_dir=generated)
    inputs.extend(group_sources)
    for prefix in ['test_ids','true_labels','pred_labels','best_class_pred_probs']:
        inputs.append(artifact_file(generated,f'{prefix}_bc_{args.model}.pt'))
    inputs += [args.data / 'preprocessed/out_genes_bc.csv',
               artifact_file(generated,f'test_probe_genes_bc_{args.model}.csv')]
    predictions = []
    for group, label, filename, color in GROUPS:
        path = extracted / filename
        frame = read_predictions(path, group)
        if group == 'negative':
            # Archived training exports may include classes 3–5; compare class 5 only.
            frame = frame.loc[frame['pred_labels'].eq(5)].copy()
        frame.to_csv(output_path(output, f'{group}_extracted_cpg_genes.tsv', suffix), sep='\t', index=False)
        predictions.append(frame)
        inputs.append(path)
    internal = internal_statistics(predictions, signal_path, args.matrix)
    external_pairs = pd.read_csv(args.external, sep='\t', dtype={'CpG_ID': str, 'Gene': str})
    external = collapse_external(external_pairs)
    universe_path = args.external_universe or args.external.parent / 'shared_cpg_statistics.tsv'
    if not universe_path.is_file():
        raise FileNotFoundError(f'{universe_path}: full shared-CpG universe required for FDR; run evaluation/evaluate_datasets.py')
    universe = pd.read_csv(universe_path, sep='\t', dtype={'CpG_ID': str})
    inferred = external_inference(universe)
    inputs.append(universe_path)
    inference_columns = ['External_p_value','External_FDR','External_CI95_low','External_CI95_high','External_testable']
    # Ensure the inferential universe describes the exact effect table being plotted.
    check = external.merge(universe[['CpG_ID'] + EXTERNAL_STATS], on='CpG_ID', how='left',
                           validate='one_to_one', suffixes=('_plot','_universe'), indicator=True)
    if not check['_merge'].eq('both').all():
        raise ValueError('External comparison CpGs absent from FDR universe')
    for field in EXTERNAL_STATS:
        if not np.allclose(check[field+'_plot'], check[field+'_universe'], equal_nan=True, atol=1e-10):
            raise ValueError(f'External comparison differs from FDR universe: {field}')
    external = external.merge(inferred[['CpG_ID'] + inference_columns], on='CpG_ID', validate='one_to_one')
    audit, audit_paths = cohort_audit(args.matrix, training_series, ROOT, HERE/'raw')
    inputs.extend(audit_paths)
    output_path(output, 'cohort_audit.json', suffix).write_text(json.dumps(audit, indent=2)+'\n')
    inferred[['CpG_ID', EXTERNAL_DELTA] + inference_columns].to_csv(
        output_path(output, 'external_differential_methylation_universe.tsv', suffix), sep='\t', index=False)
    matched, plot_data, summary = {}, {}, []
    for (group, label, filename, color), frame in zip(GROUPS, predictions):
        joined = annotate_candidates(match_predictions(frame, external, internal))
        joined.to_csv(output_path(output, f'{group}_all_predictions.tsv', suffix), sep='\t', index=False)
        common = joined.loc[joined['Matched_external_CpG']].copy()
        common.to_csv(output_path(output, f'{group}_common_cpg_gene_matches.tsv', suffix), sep='\t', index=False)
        unique = common.drop_duplicates('CpG_ID').copy()
        genes = common.groupby('CpG_ID')['Predicted_gene'].agg(lambda values: ';'.join(sorted(set(values))))
        unique['Predicted_genes'] = unique['CpG_ID'].map(genes)
        # Per-CpG tables should not retain one arbitrary gene's prediction score.
        columns = ['prediction_group', 'CpG_ID', 'Predicted_genes', 'External_genes',
                   'Internal_healthy_mean_beta', 'Internal_BC_mean_beta',
                   'Internal_delta_beta_BC_minus_healthy', 'Internal_healthy_n', 'Internal_BC_n'] + EXTERNAL_STATS + ['External_minus_internal_delta_beta'] + inference_columns + ['Direction_concordant', 'Internal_absolute_delta_beta', 'External_absolute_delta_beta', 'Absolute_delta_error', 'External_significant', 'Meets_exploratory_criterion']
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
        row.update(agreement_metrics(valid))
        summary.append(row)
        matched[group], plot_data[group] = unique, valid
    pd.DataFrame(summary).to_csv(output_path(output, 'summary.tsv', suffix), sep='\t', index=False)
    overlap = sorted(set(matched['negative']['CpG_ID']) & set(matched['likely_positive']['CpG_ID']))
    pd.DataFrame({'CpG_ID': overlap}).to_csv(output_path(output, 'cpgs_in_both_prediction_groups.tsv', suffix), sep='\t', index=False)
    set_plot_style()
    sensitivity = sensitivity_table(plot_data)
    sensitivity.to_csv(output_path(output, 'criterion_sensitivity.tsv', suffix), sep='\t', index=False)
    plot_groups = GROUPS
    save_plot(validation_plots(plot_data,summary,plot_groups,original_axis), output, 'delta_beta_original_vs_external', suffix)
    save_plot(sensitivity_plot(sensitivity,plot_groups), output, 'criterion_sensitivity', suffix)
    rng = np.random.default_rng(32)
    fig, axes = plt.subplots(1,2,figsize=(17,6.5),constrained_layout=True)
    for ax,column,title in [(axes[0],'External_minus_internal_delta_beta','Change in effect across cohorts'),
                            (axes[1],EXTERNAL_DELTA,'Absolute breast tissue methylation effect')]:
        for position,(group,label,filename,color) in enumerate(plot_groups,1):
            values=plot_data[group][column].to_numpy()
            if column==EXTERNAL_DELTA:
                values=np.abs(values)
            box=ax.boxplot([values],positions=[position],widths=.5,showfliers=False,patch_artist=True,
                           medianprops={'color':'black','linewidth':1.8},
                           whiskerprops={'color':color,'linewidth':1.4},capprops={'color':color,'linewidth':1.4})
            box['boxes'][0].set(facecolor=color,edgecolor=color,linewidth=1.5,alpha=.3)
            ax.scatter(position+rng.uniform(-.15,.15,len(values)),values,color=color,alpha=.35,s=14,edgecolors='none',rasterized=True)
        ax.axhline(0,color='#999999',lw=.8)
        ax.set(xticks=[1,2],xticklabels=[label.replace('Predicted ', 'Predicted\n') for _,label,_,_ in plot_groups],title=title,
               ylabel=f'External Δβ − {original_axis} Δβ' if column!='Delta_beta_BC_minus_healthy' else '|Breast tissue Δβ|')
    save_plot(fig,output,'delta_beta_shift_by_prediction_group',suffix)
    report=dict(external_inference={'method':'Two-sided Welch t-test from sample mean, sample SD (ddof=1), per-CpG n',
                'FDR_method':'Benjamini-Hochberg before candidate selection; untestable sites retained as p=1 in family',
                'FDR_universe':str(universe_path.resolve()), 'FDR_family_size':len(universe),
                'testable_CpGs':int(inferred.External_testable.sum()), 'CI':'95% pointwise Welch interval; no covariate or batch adjustment',
                'criterion':'abs(external delta)>=0.10 AND BH FDR<0.05 AND consistent nonzero direction',
                'threshold_status':'Exploratory defaults selected for this analysis; not prospectively registered',
                'sensitivity':'effect thresholds 0.05,0.10,0.20; FDR cutoffs 0.01,0.05,0.10',
                'method_reference':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html'},
                cohort_audit=audit,
                prediction_group_source='exported_csv' if group_sources else 'reconstructed_from_tensors',
                prediction_group_source_files=[str(path.resolve()) for path in group_sources], model=args.model, model_artifact_directory=str(generated), data_directory=str(args.data), output_suffix=suffix, delta_definition='mean beta in BC minus mean beta in healthy controls',
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
    lines += ['', '### Complementary reproducibility metrics', '',
              '| Group | Spearman | Direction agreement | MAE | Median original / external absolute Δβ | External absolute Δβ > 0.10 / > 0.20 | FDR < 0.05 | Exploratory criterion |',
              '|---|---:|---:|---:|---|---|---:|---:|']
    for row in summary:
        spearman = f"{row['spearman_r']:.3f}" if row['spearman_r'] is not None else 'undefined'
        lines.append(f"| {row['prediction_group']} | {spearman} | {row['same_direction_fraction']:.1%} | {row['mean_absolute_error']:.4f} | {row['internal_median_absolute_delta_beta']:.4f} / {row['external_median_absolute_delta_beta']:.4f} | {row['external_absolute_delta_gt_0.10_fraction']:.1%} / {row['external_absolute_delta_gt_0.20_fraction']:.1%} | {row['external_significant_fraction']:.1%} | {row['exploratory_criterion_count']}/{row['plotted_unique_cpgs']} ({row['exploratory_criterion_fraction']:.1%}) |")
    lines += ['', 'Welch tests use saved sample means, sample SDs, and per-CpG counts, sufficient for the unadjusted test and pointwise 95% confidence intervals. BH uses the full shared CpG universe before selecting predictions. Untestable sites remain in the family as p=1; missing inference is not counted as support.',
              'Support criterion: |external Δβ| ≥ 0.10, BH FDR < 0.05, consistent nonzero direction. These are exploratory defaults, with sensitivity plots at effect thresholds 0.05/0.10/0.20 and FDR 0.01/0.05/0.10.',
              'Higher Pearson correlation can reflect a wider effect range. SD, direction concordance, and MAE quantify complementary properties; no correlation value alone establishes biological relevance.',
              'Direction concordance uses exact signs, including zero-zero matches; the support criterion requires nonzero effects. Effect-threshold summaries use strict > cutoffs, whereas the support criterion uses ≥.',
              'The denominator is finite matched unique CpGs per prediction group. Group selections depend on graph/methylation-derived labels; overlapping CpGs and correlated sites preclude an independent LP-versus-negative test. Stronger descriptive effects do not establish GNN improvement over selection.',
              'The existing external data do not supply a same-study case/control replication cohort. GSE66695 is discovery; reusing it would leak discovery data. GSE69914 could provide a separate future same-study analysis after input/QC and patient-overlap review.',
              'No NIAPU/Random Forest external ranking comparison is claimed: comparable held-out prediction outputs are not supplied. --models can compare existing saved models without retraining.',
              '',f'Common CpGs assigned to both prediction groups: {len(overlap)}. Such CpGs remain in both group panels; the groups are not assumed independent.',
              '', f'Plots: `delta_beta_original_vs_external_{suffix}.png` and `delta_beta_shift_by_prediction_group_{suffix}.png`, with PDF versions.',
              'Additional outputs: criterion_sensitivity PNG/PDF/TSV, per-CpG external p/FDR/95% CI, full external test universe, and cohort_audit JSON.',
              'Predicted-negative plots and comparison tables use predicted class 5 only. Source exports retain their original selections.',
              'Tables retain selected features, matching CpG-gene rows, unique matched CpGs, and unmatched predictions (Matched_external_CpG=false).',
              '',report['limitations'],'',
              f'Regenerate from breast_cancer_analysis with `python evaluation/compare_predictions.py --model {args.model} --data {args.data}`.']
    output_path(output, 'README.md', suffix).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'suffix':suffix,'groups':summary,'cpgs_in_both_prediction_groups':len(overlap)},indent=2))


if __name__=='__main__':
    main()
