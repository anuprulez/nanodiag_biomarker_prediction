"""Frozen original-trained BC classifiers evaluated once on GSE52865.

Run with the original nanodiag_biomarker_gnn Python environment:
python -m post_evaluation_support.independent_feature_verification --parent PATH
"""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .publication import STYLE, save_publication

SERIES = 'GSE52865'
URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52865/matrix/GSE52865_series_matrix.txt.gz'
MODELS = {'Random Forest': 'random_forest', 'TabPFN3': 'TabPFN3',
          'TabICLv2': 'TabICLv2', 'XGBoost': 'xgboost'}
DISPLAY = {'Random Forest': 'RF', 'TabPFN3': 'TabPFN', 'TabICLv2': 'TabICL', 'XGBoost': 'XGBoost'}
COLORS = dict(zip(MODELS, ['#7B2CBF', '#009E73', '#CC79A7', '#D55E00']))
SEED = 20261031


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(parent, out, plan):
    """Extract fixed columns and audit sample identity before any inference."""
    source = out / 'source/GSE52865_series_matrix.txt.gz'
    wanted = set().union(*map(set, plan['panels'].values()))
    metadata = {}
    rows = {}
    with gzip.open(source, 'rt') as handle:
        for line in handle:
            fields = next(csv.reader([line], delimiter='\t'))
            if not fields:
                continue
            if fields[0] == '!series_matrix_table_begin':
                samples = next(csv.reader([next(handle)], delimiter='\t'))[1:]
                break
            if fields[0].startswith('!Sample_'):
                metadata.setdefault(fields[0], []).append(fields[1:])
        assert samples == metadata['!Sample_geo_accession'][0]
        assert len(samples) == len(set(samples)) == 57
        assert metadata['!Sample_platform_id'] == [['GPL13534'] * 57]
        count = 0
        seen = set()
        for fields in csv.reader(handle, delimiter='\t'):
            if fields[0] == '!series_matrix_table_end':
                break
            assert len(fields) == len(samples) + 1
            assert fields[0] not in seen
            seen.add(fields[0]); count += 1
            if fields[0] in wanted:
                rows[fields[0]] = pd.to_numeric(pd.Series(fields[1:]), errors='coerce').to_numpy(float)
        else:
            raise ValueError('Truncated GEO matrix')
    missing = sorted(wanted - set(rows))
    if missing:
        raise ValueError(f'Frozen features missing from GEO matrix: {missing}')
    matrix = pd.DataFrame(rows, index=pd.Index(samples, name='sample_id'))
    values = matrix.to_numpy()
    assert not np.isinf(values).any()
    assert ((values[np.isfinite(values)] >= 0) & (values[np.isfinite(values)] <= 1)).all()
    sources = metadata['!Sample_source_name_ch1'][0]
    label_map = {'Breast tumour': 1, 'Normal breast tissue': 0}
    assert set(sources) == set(label_map)
    labels = [label_map[s] for s in sources]
    assert labels.count(1) == 40 and labels.count(0) == 17
    disease_fields = metadata['!Sample_characteristics_ch1'][0]
    assert all((field == 'disease state: Tumor sample') == (label == 1)
               for field, label in zip(disease_fields, labels))
    meta = pd.DataFrame(dict(sample_id=samples, series=SERIES,
                            title=metadata['!Sample_title'][0],
                            source_name=sources, label=labels,
                            group=['breast_tumour' if y else 'morphologically_normal_breast' for y in labels],
                            characteristics=[' | '.join(row[i] for row in metadata['!Sample_characteristics_ch1'])
                                             for i in range(len(samples))])).set_index('sample_id')
    write_json(out / 'source/sample_metadata.json', metadata)
    matrix.to_csv(out / 'data/independent_beta.tsv.gz', sep='\t',
                  compression={'method': 'gzip', 'mtime': 0})
    meta.to_csv(out / 'data/independent_sample_annotations.tsv', sep='\t')
    coverage = []
    for panel, features in plan['panels'].items():
        for order, cpg in enumerate(features):
            coverage.append(dict(panel=panel, feature_order=order, CpG_ID=cpg,
                                 present=True, missing_values=int(matrix[cpg].isna().sum())))
    pd.DataFrame(coverage).to_csv(out / 'data/frozen_feature_coverage.tsv', sep='\t', index=False)
    overlaps = {}
    fingerprints = {}
    candidate_features = plan['panels']['candidates_only']
    for cohort in ['original', 'external']:
        prior_meta = pd.read_csv(parent / f'cache/{cohort}_metadata.tsv', sep='\t', index_col=0) if (parent / f'cache/{cohort}_metadata.tsv').exists() else None
        prior = pd.read_pickle(parent / f'cache/{cohort}.pkl')
        accessions = set(map(str, prior.index))
        if prior_meta is not None and 'geo_accession' in prior_meta:
            accessions.update(prior_meta.geo_accession.astype(str))
        # Original IDs include prefixes such as BC_GSM1629194.
        accessions.update(match.group(0) for value in list(accessions)
                          if (match := re.search(r'GSM\d+', value)))
        overlaps[cohort] = sorted(set(samples) & accessions)
        # Exact candidate profiles detect duplicate deposited methylation vectors
        # even when GEO accession aliases differ. No labels are used here.
        prior_values = prior[candidate_features].to_numpy(float)
        hits = []
        for sample, vector in matrix[candidate_features].iterrows():
            matches = np.flatnonzero(np.all(np.isclose(prior_values, vector.to_numpy(), atol=1e-12, rtol=0, equal_nan=True), axis=1))
            hits.extend(dict(independent_sample=sample, previous_sample=str(prior.index[j])) for j in matches)
        fingerprints[cohort] = hits
    assert not any(overlaps.values()), overlaps
    assert not any(fingerprints.values()), fingerprints
    audit = dict(dataset=SERIES, source_url=URL, source_sha256=digest(source),
                 platform='GPL13534 / Illumina HumanMethylation450',
                 matrix_probe_rows=count, n_samples=57, n_tumours=40, n_normal=17,
                 selected_CpGs=len(wanted), missing_values=int(matrix.isna().sum().sum()),
                 missing_CpGs=missing, accession_overlap=overlaps,
                 duplicate_candidate_beta_profiles=fingerprints,
                 processing=metadata['!Sample_data_processing'][0][0],
                 independence_limit='No matching GEO sample accessions or exact 33-CpG profiles. Public metadata do not establish complete donor identity across all studies; no claim of a donor-level identity assay.',
                 control_definition='Morphologically normal breast tissue, not established cancer-free donors.',
                 inference_policy='All 33 candidates retained, same original-trained imputers and models; no external fitting, tuning, normalization, calibration, feature ranking or reselection.')
    write_json(out / 'data/input_audit.json', audit)
    return matrix, meta, audit


def predict(parent, out, plan, matrix):
    """Never receive evaluation labels: these models are predict-only here."""
    torch.set_num_threads(4)
    model_audit = {}
    frames = []
    matrix_hash = digest(out / 'data/independent_beta.tsv.gz')
    for panel in ['candidates_only']:
        features = plan['panels'][panel]
        for name, suffix in MODELS.items():
            path = parent / f'models/{panel}_{suffix}.joblib'
            key = panel + '_' + suffix
            record = dict(source_model=str(path), sha256=digest(path), panel=panel,
                          model=name, features=features, input_matrix_sha256=matrix_hash,
                          versions={pkg: version(pkg) for pkg in ['scikit-learn', 'xgboost', 'torch', 'tabpfn', 'tabicl', 'joblib']})
            token = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            cache = out / f'data/{key}_probabilities.tsv'
            stamp = out / f'models/{key}.json'
            if cache.exists() and stamp.exists() and json.loads(stamp.read_text()).get('token') == token:
                frame = pd.read_csv(cache, sep='\t')
                print(f'Reuse frozen predictions: {panel} / {name}', flush=True)
            else:
                print(f'Independent inference: {panel} ({len(features)} CpGs) / {name}', flush=True)
                model = joblib.load(path)
                assert list(model.feature_names_in_) == features
                np.testing.assert_array_equal(model.classes_, [0, 1])
                # The persisted pipeline contains the original training median.
                imputer = model.steps[0][1]
                assert imputer.statistics_.shape == (len(features),)
                probs = model.predict_proba(matrix[features])[:, 1]
                assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
                frame = pd.DataFrame(dict(sample_id=matrix.index, panel=panel,
                                          model=name, probability=probs,
                                          predicted_label_at_0_5=(probs >= .5).astype(int)))
                frame.to_csv(cache, sep='\t', index=False)
                write_json(stamp, dict(token=token, **record))
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            assert path.exists() and digest(path) == record['sha256']
            model_audit[key] = record
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), model_audit


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                Brier=brier_score_loss(y, p), accuracy=accuracy_score(y, p >= .5),
                balanced_accuracy=balanced_accuracy_score(y, p >= .5),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                F1=f1_score(y, p >= .5, zero_division=0),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def exact_binomial_interval(successes, total):
    """Clopper–Pearson intervals remain informative at zero observed errors."""
    low = 0. if successes == 0 else beta_distribution.ppf(.025, successes, total - successes + 1)
    high = 1. if successes == total else beta_distribution.ppf(.975, successes + 1, total - successes)
    return float(low), float(high)


def evaluate(predictions, meta, out, repeats):
    predictions = predictions.merge(meta.reset_index()[['sample_id', 'series', 'label', 'group']],
                                    on='sample_id', validate='many_to_one')
    predictions.to_csv(out / 'data/patient_predictions.tsv', sep='\t', index=False)
    rows = []
    for (panel, name), frame in predictions.groupby(['panel', 'model'], sort=False):
        y = frame.label.to_numpy(int); p = frame.probability.to_numpy(float)
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(repeats):
            indices = np.concatenate([rng.choice(np.flatnonzero(y == k), (y == k).sum(), replace=True) for k in [0, 1]])
            boot.append(metrics(y[indices], p[indices]))
        ci = pd.DataFrame(boot)
        row = dict(panel=panel, model=name, n_samples=len(frame), n_cancer=int(y.sum()),
                   n_normal=int((y == 0).sum()), n_features=33,
                   threshold=.5, **metrics(y, p))
        for metric in ['AUROC', 'AUPRC', 'Brier', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'F1']:
            row[metric + '_CI95_low'], row[metric + '_CI95_high'] = ci[metric].quantile([.025, .975]).to_numpy(float)
        for metric, successes, total in [('accuracy', row['TN'] + row['TP'], len(y)),
                                         ('sensitivity', row['TP'], row['TP'] + row['FN']),
                                         ('specificity', row['TN'], row['TN'] + row['FP'])]:
            row[metric + '_CI95_low'], row[metric + '_CI95_high'] = exact_binomial_interval(successes, total)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'data/classifier_metrics.tsv', sep='\t', index=False)
    result.loc[result.panel == 'candidates_only'].to_csv(out / 'candidate_classifier_metrics.tsv', sep='\t', index=False)
    return predictions, result


def plots(predictions, out):
    for panel in ['candidates_only']:
        data = predictions.loc[predictions.panel == panel]
        caption = '33 candidate CpGs'
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
            for name in MODELS:
                f = data.loc[data.model == name]
                y, p = f.label.to_numpy(int), f.probability.to_numpy(float)
                x, z, _ = roc_curve(y, p)
                axes[0].plot(x, z, color=COLORS[name], lw=2.5,
                             label=f'{DISPLAY[name]} · AUROC = {roc_auc_score(y,p):.3f}')
                precision, recall, _ = precision_recall_curve(y, p)
                axes[1].step(recall, precision, where='post', color=COLORS[name], lw=2.5,
                             label=f'{DISPLAY[name]} · AP = {average_precision_score(y,p):.3f}')
                axes[1].plot(recall[0], precision[0], 'o', color=COLORS[name], ms=4)
            axes[0].plot([0, 1], [0, 1], ':', color='#999999')
            axes[1].axhline(40/57, ls=':', color='#999999', label='Cancer prevalence = 0.702')
            for i, ax in enumerate(axes):
                ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025),
                       xlabel='1 − specificity' if i == 0 else 'Recall',
                       ylabel='Sensitivity' if i == 0 else 'Precision',
                       title='A   ROC' if i == 0 else 'B   Precision–recall')
                ax.set_aspect('equal'); ax.grid(color='#e8e8e8', lw=.6)
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(loc='lower right' if i == 0 else 'lower left', fontsize=10)
            fig.suptitle(f'Breast cancer: independent GSE52865 | {caption}\nOriginal-trained models · no class weighting', fontsize=18)
            fig.tight_layout(rect=(0, 0, 1, .90))
            save_publication(fig, out, panel + '_independent_ROC_PR')
            fig, axes = plt.subplots(1, 4, figsize=(17, 5))
            for ax, name in zip(axes, MODELS):
                f = data.loc[data.model == name]
                cm = confusion_matrix(f.label, f.probability >= .5, labels=[0, 1])
                ax.imshow(cm, cmap='Blues', vmin=0, vmax=40)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20,
                                color='white' if cm[i, j] > 20 else '#202020')
                ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Cancer'],
                       yticklabels=['Normal', 'Cancer'], xlabel='Predicted', ylabel='Observed', title=DISPLAY[name])
            fig.suptitle(f'Breast cancer: independent GSE52865 | {caption}\nDecision threshold = 0.5', fontsize=18)
            fig.tight_layout(rect=(0, 0, 1, .85))
            save_publication(fig, out, panel + '_independent_confusion_matrices')


def report(out, audit, results, repeats):
    lines = ['# Breast cancer: independent feature verification', '',
             'The fixed 33-CpG candidate panel and saved original-trained RF, TabPFN, TabICL and XGBoost models were evaluated on GSE52865. No new cohort labels were used to select features, fit models, tune settings, choose thresholds, or calibrate probabilities.', '',
             '## Dataset', '',
             '- [GSE52865](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52865): 40 breast tumours and 17 morphologically normal breast tissues; all 57 samples retained. Both classes are in the same study on GPL13534 (Illumina 450K).',
             '- GEO processing: ' + audit['processing'],
             f'- All 33 previously selected candidate CpGs are measured; {audit["missing_values"]} missing beta values in this panel.',
             '- No overlapping GEO accessions or exact 33-CpG profiles were found with the original or previous external cohort. The dataset was selected for study design and platform before classifier outcomes were inspected.', '',
             '## Frozen evaluation', '',
             '- Only the 33 candidate CpGs selected in the earlier original-versus-external comparison are used, in their original model order. No seed panel, additional features, or feature reselection is used.',
             '- The four saved 33-CpG pipelines were loaded directly. Training remains the original cohort, GSE66695; no fitting is performed on GSE52865. All imputation parameters come from the stored original-trained pipelines.',
             '- RF and XGBoost retain their prior fixed settings. TabPFN3 and TabICLv2 use the same checkpoint files and 16-estimator settings as before. No class weighting or foundation prior correction is applied.',
             '- Classification threshold is fixed at 0.5. No joint normalization or batch adjustment was fitted across the original and independent cohorts.',
             f'- AUROC, AP, Brier, balanced accuracy and F1 intervals use {repeats} stratified specimen bootstrap resamples, conditional on the fixed panel and predictions. Accuracy, sensitivity and specificity use exact Clopper–Pearson binomial intervals, which remain informative with zero observed errors. Intervals do not include model-training or feature-selection uncertainty. Perfect-ranking empirical bootstrap intervals can collapse to 1.0 and do not establish perfect population discrimination.', '',
             '## Primary results: 33 candidate CpGs', '',
             '| Model | AUROC (95% CI) | AP | Accuracy | Sensitivity | Specificity | TN / FP / FN / TP |',
             '|---|---|---|---|---|---|---|']
    for name in MODELS:
        r = results.loc[(results.panel == 'candidates_only') & (results.model == name)].iloc[0]
        lines.append(f'| {DISPLAY[name]} | {r.AUROC:.3f} ({r.AUROC_CI95_low:.3f}–{r.AUROC_CI95_high:.3f}) | {r.AUPRC:.3f} | {r.accuracy:.1%} | {r.sensitivity:.1%} | {r.specificity:.1%} | {r.TN} / {r.FP} / {r.FN} / {r.TP} |')
    lines += ['', 'AP is average precision; its prevalence baseline is 40/57 = 0.702. Threshold metrics use 0.5.', '',
              '## Interpretation', '',
              'This is an independent-cohort test relative to the documented panel-selection pipeline. Earlier original-cohort CV and selected-external-cohort scores remain discovery-conditioned; they are not made independent by this new evaluation.', '',
              'The outcome is tumour tissue versus morphologically normal tissue. Normal tissue is not proof that a donor is cancer-free, so these results do not establish patient screening or blood-test performance. Sample titles suggest that some normal and tumour specimens may share donors; GEO does not provide validated donor linkage here. Bootstrap intervals treat specimens independently and can understate uncertainty if specimens are paired. Complete cross-study donor identity cannot be proven from public accession metadata. The cohort is small (17 normal specimens), and processing and subtype differences can affect transfer.', '',
              'The data were chosen before observing these scores. Any later feature or model changes based on this cohort would make it development data for those changes.', '',
              '## Files', '',
              '- [Candidate ROC and precision–recall](plots/candidates_only_independent_ROC_PR.pdf)',
              '- [Candidate confusion matrices](plots/candidates_only_independent_confusion_matrices.pdf)',
              '- [All model metrics and intervals](data/classifier_metrics.tsv)',
              '- [Per-sample predictions](data/patient_predictions.tsv)',
              '- [Sample and input audit](data/input_audit.json)',
              '- [Frozen feature coverage](data/frozen_feature_coverage.tsv)',
              '- [Reproducibility manifest](manifest.json)', '',
              'Run the script with the original nanodiag_biomarker_gnn environment; the source matrix and saved original models are required. Per-model probability caches are reused only if model/input hashes and package versions match.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    parent = args.parent.resolve()
    out = parent / 'independent_feature_verification'
    plan = json.loads((out / 'frozen_analysis_plan.json').read_text())
    assert len(plan['panels']['candidates_only']) == 33
    assert set(plan['panels']) == {'candidates_only'}
    matrix, meta, audit = prepare(parent, out, plan)
    print('Input audit passed: 57 samples, 40 tumour / 17 normal; all frozen features present.', flush=True)
    raw, model_audit = predict(parent, out, plan, matrix)
    predictions, results = evaluate(raw, meta, out, args.bootstrap)
    assert len(predictions) == 57 * 4
    assert not predictions.duplicated(['sample_id', 'model', 'panel']).any()
    plots(predictions, out)
    report(out, audit, results, args.bootstrap)
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), dataset=SERIES,
                    source_url=URL, source_sha256=audit['source_sha256'],
                    frozen_plan_sha256=digest(out / 'frozen_analysis_plan.json'),
                    source_classifier_manifest_sha256=digest(parent / 'classifier_comparison_manifest.json'),
                    runner_sha256=digest(__file__), models=model_audit,
                    panels=plan['panels'], inference_only=True, external_fit=False,
                    threshold=.5, class_weighting=False, bootstrap_repeats=args.bootstrap,
                    interval_methods={'accuracy_sensitivity_specificity': 'exact Clopper–Pearson binomial',
                                      'AUROC_AP_Brier_balanced_accuracy_F1': 'stratified specimen percentile bootstrap'},
                    bootstrap_seed=SEED, input_audit=audit,
                    outputs={str(p.relative_to(out)): digest(p) for folder in ['data', 'plots']
                             for p in sorted((out / folder).glob('*')) if p.is_file()})
    write_json(out / 'manifest.json', manifest)
    print(results.loc[results.panel == 'candidates_only', ['model', 'AUROC', 'AUPRC', 'accuracy', 'sensitivity', 'specificity']].to_string(index=False), flush=True)
    print('Complete:', out, flush=True)


if __name__ == '__main__':
    main()
