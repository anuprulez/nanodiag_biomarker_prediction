"""Patient-level follow-up of fixed, externally selected methylation panels.

All performance estimates are exploratory: the fixed panel has already seen
cohort labels during its discovery. Patient CV does not undo that selection.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import zipfile
from urllib.request import urlopen, Request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, fisher_exact
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .annotation_publication import annotation_figure, reactome_figure
from .publication import performance_figure, external_distribution_figure, forest_figure, save_publication

RANDOM_SEED = 20261031
GMT_URL = 'https://reactome.org/download/current/ReactomePathways.gmt.zip'
CAUTION = ('Exploratory fixed-panel analysis: candidates were selected using original and external '
           'cohort labels. Patient CV does not remove panel-selection bias. External cases and '
           'controls are from different studies; disease and study cannot be separated. '
           'Five training runs are not five independent biological replications.')


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out/'plots'/f'{name}.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_panel(base, disease, expected):
    suffix = 'PNA_' + base.name.removeprefix('data_')
    src = base/'accumulated_results/evaluation'/f'likely_positive_common_unique_cpgs_{suffix}.tsv'
    frame = pd.read_csv(src, sep='\t')
    internal = f'Internal_delta_beta_{disease}_minus_healthy'
    external = f'Delta_beta_{disease}_minus_healthy'
    flags = (frame.n_runs.eq(5) & frame[external].abs().ge(.30)
             & frame.External_p_value.lt(.01) & frame.External_FDR.lt(.01)
             & frame[internal].ne(0) & frame[external].ne(0)
             & np.sign(frame[internal]).eq(np.sign(frame[external])))
    panel = frame.loc[flags].sort_values(external, key=lambda x: x.abs(), ascending=False).copy()
    if len(panel) != expected or panel.CpG_ID.duplicated().any():
        raise ValueError(f'{src}: expected fixed {expected}-CpG {disease} panel; found {len(panel)}. '
                         'Run the five-run comparison/accumulation first; no automatic panel substitution.')
    reference=Path(__file__).parent/'resources'/f'fixed_{disease}_{expected}_CpGs.tsv'
    if reference.exists():
        fixed=set(pd.read_csv(reference,sep='\t').CpG_ID)
        if set(panel.CpG_ID)!=fixed:
            raise ValueError('Selected CpG identities differ from the frozen reference panel; refusing substitution')
    return panel, src, internal, external


def load_common_prediction_panels(base, disease):
    """Use the unthresholded five-run intersections underlying the delta-beta plots."""
    suffix='PNA_'+base.name.removeprefix('data_')
    counts={'BC':{'likely_positive':189,'negative':70},
            'AML':{'likely_positive':259,'negative':218}}[disease]
    panels={};sources={}
    for group,expected in counts.items():
        path=base/'accumulated_results/evaluation'/f'{group}_common_unique_cpgs_{suffix}.tsv'
        table=pd.read_csv(path,sep='\t')
        internal=f'Internal_delta_beta_{disease}_minus_healthy'
        external=f'Delta_beta_{disease}_minus_healthy'
        if table.CpG_ID.duplicated().any() or not table.n_runs.eq(5).all():
            raise ValueError(f'{path}: expected unique CpGs common to five runs')
        table=table.loc[np.isfinite(table[[internal,external]]).all(axis=1)]
        if len(table)!=expected:
            raise ValueError(f'{path}: expected {expected} common {group} CpGs, found {len(table)}')
        name='all_common_likely_positive' if group=='likely_positive' else 'all_common_predicted_negative'
        panels[name]=sorted(table.CpG_ID)
        sources[name]={'path':str(path),'sha256':digest(path),'n_features':len(table)}
    return panels,sources


def read_original(path, wanted):
    """Stream a CpG or CpG-gene matrix, verifying duplicate annotation rows."""
    rows = {}
    header = pd.read_csv(path, sep='\t', nrows=0).columns
    samples = list(header[1:])
    if len(samples) != len(set(samples)):
        raise ValueError('Duplicate original sample IDs')
    with pd.read_csv(path, sep='\t', chunksize=1500) as reader:
        for chunk in reader:
            ids = chunk.iloc[:, 0].astype(str).str.extract(r'^(cg\d+)', expand=False)
            for idx in chunk.index[ids.isin(wanted)]:
                cpg = ids.loc[idx]
                values = chunk.loc[idx].iloc[1:].to_numpy(dtype=float)
                if cpg in rows and not np.allclose(rows[cpg], values, equal_nan=True):
                    raise ValueError(f'Conflicting original CpG-gene rows: {cpg}')
                rows[cpg] = values
    if not rows:
        raise ValueError('No requested CpGs in original matrix')
    return pd.DataFrame(rows, index=pd.Index(samples, name='sample_id'))


def load_geo_module(root, disease):
    path = root/('evaluation/geo_common.py' if disease == 'BC' else 'preprocess_raw_data/geo_common.py')
    spec = importlib.util.spec_from_file_location('post_eval_geo_'+disease, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_external(root, disease, wanted):
    geo = load_geo_module(root, disease)
    matrices, records = [], []
    series = ('GSE101961', 'GSE226569') if disease == 'BC' else ('GSE40279', 'GSE62298')
    for label, study in enumerate(series):
        path = root/'evaluation/raw'/f'{study}_series_matrix.txt.gz'
        with gzip.open(path, 'rt') as source:
            samples, meta = geo.read_header(source)
        rows = {}
        if disease == 'BC':
            iterator = geo.rows(study, root/'evaluation/raw')
            for cpg, values in iterator:
                if cpg in wanted:
                    rows[cpg] = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
        else:
            with gzip.open(path, 'rt') as source:
                geo.read_header(source)
                for cpg, values in geo.beta_rows(source, len(samples)):
                    if cpg in wanted:
                        rows[cpg] = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
        matrices.append(pd.DataFrame(rows, index=pd.Index(samples, name='sample_id')))
        for i, sample in enumerate(samples):
            chars = ' | '.join(row[i] for row in meta.get('!Sample_characteristics_ch1', []))
            records.append(dict(sample_id=sample, series=study, label=label,
                                group='healthy_control' if label == 0 else disease,
                                characteristics=chars))
        print(f'Loaded external {study}: {len(samples)} patients; {len(rows)} requested CpGs', flush=True)
    matrix = pd.concat(matrices)
    if not matrix.index.is_unique:
        raise ValueError('External cohort contains repeated sample IDs')
    return matrix, pd.DataFrame(records).set_index('sample_id')


def original_metadata(config, matrix):
    meta = pd.read_csv(config.p_sample_metadata, sep='\t', dtype={'sample_id': str})
    if meta.sample_id.duplicated().any():
        raise ValueError('Duplicate original metadata sample IDs')
    meta = meta.set_index('sample_id').loc[matrix.index].copy()
    meta['label'] = pd.to_numeric(meta.label, errors='raise').astype(int)
    if set(meta.label) != {0, 1}:
        raise ValueError('Expected two patient-level disease groups')
    return meta


def get_matrices(root, config, wanted, out):
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    disease = 'BC' if root.name == 'breast_cancer_analysis' else 'AML'
    inputs = [Path(config.p_arrays), Path(config.p_sample_metadata)] + sorted((root/'evaluation/raw').glob('*.gz'))
    fingerprint = dict(wanted=sorted(wanted), files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in inputs])
    token = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
    stamp = cache/'fingerprint.json'
    if stamp.exists() and json.loads(stamp.read_text()).get('token') == token:
        original = pd.read_pickle(cache/'original.pkl')
        external = pd.read_pickle(cache/'external.pkl')
        ext_meta = pd.read_csv(cache/'external_metadata.tsv', sep='\t', index_col=0)
    else:
        print('Streaming original patient matrix (requested CpGs only)', flush=True)
        original = read_original(config.p_arrays, wanted)
        external, ext_meta = read_external(root, disease, wanted)
        original.to_pickle(cache/'original.pkl'); external.to_pickle(cache/'external.pkl')
        ext_meta.to_csv(cache/'external_metadata.tsv', sep='\t')
        stamp.write_text(json.dumps(dict(token=token, inputs=fingerprint), indent=2))
    for name, matrix in [('original', original), ('external', external)]:
        values = matrix.to_numpy(float)
        finite = values[np.isfinite(values)]
        if ((finite < 0) | (finite > 1)).any() or np.isinf(values).any():
            raise ValueError(f'{name}: invalid beta values')
    return original, original_metadata(config, original), external, ext_meta


def effects_and_heatmaps(panel, original, ometa, external, emeta, internal, external_col, out):
    panel = panel.copy()
    for cohort, matrix, meta in [('Original', original, ometa), ('External', external, emeta)]:
        cases = matrix.loc[meta.label.eq(1), panel.CpG_ID]
        controls = matrix.loc[meta.label.eq(0), panel.CpG_ID]
        delta = cases.mean()-controls.mean()
        a, b = cases.var(ddof=1)/cases.count(), controls.var(ddof=1)/controls.count()
        df = (a+b)**2/(a*a/(cases.count()-1)+b*b/(controls.count()-1))
        margin = t.ppf(.975, df)*np.sqrt(a+b)
        panel[f'{cohort}_patient_delta'] = panel.CpG_ID.map(delta)
        panel[f'{cohort}_CI95_low'] = panel.CpG_ID.map(delta-margin)
        panel[f'{cohort}_CI95_high'] = panel.CpG_ID.map(delta+margin)
        expected = panel[internal if cohort == 'Original' else external_col]
        if not np.allclose(panel[f'{cohort}_patient_delta'], expected, atol=2e-9):
            raise ValueError(f'{cohort} patient effects do not reproduce the selected panel')
        # Patient identity is retained; no patient is clustered across cohorts.
        order = meta.sort_values(['label', 'series']).index
        fig, ax = plt.subplots(figsize=(15, max(6, len(panel)*.23)))
        image = ax.imshow(matrix.loc[order, panel.CpG_ID].T, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                          interpolation='nearest', rasterized=True)
        ax.set_yticks(range(len(panel)), panel.CpG_ID, fontsize=8)
        ax.set_xticks([]); ax.set_ylabel('Fixed candidate CpGs')
        ax.set_xlabel('Patients ordered by disease and study (not an independent classification test)')
        cut = int(meta.label.eq(0).sum())
        ax.axvline(cut-.5, color='white', lw=2)
        track=ax.inset_axes([0,1.005,1,.055])
        studies={value:i for i,value in enumerate(sorted(meta.series.unique()))}
        track.imshow(np.vstack([meta.loc[order,'label'].to_numpy(),meta.loc[order,'series'].map(studies).to_numpy()]),
                     aspect='auto',cmap='tab10',vmin=0,vmax=9,interpolation='nearest')
        track.set_xticks([]);track.set_yticks([0,1],['Disease','Study'],fontsize=7)
        key='; '.join(f'{i}={study}' for study,i in studies.items())
        ax.set_title(f'{cohort}: healthy n={cut}, disease n={len(meta)-cut}\n'
                     f'Disease: 0=healthy, 1=disease; study: {key}',pad=42,fontsize=11)
        fig.colorbar(image, ax=ax, label='Raw methylation β')
        save(fig, out, cohort.lower()+'_patient_heatmap')
        # Export all available covariates instead of silently inferring absent age/sex values.
        meta.loc[order].to_csv(out/'data'/f'{cohort.lower()}_patient_annotations.tsv', sep='\t')
        matrix.loc[order, panel.CpG_ID].to_csv(out/'data'/f'{cohort.lower()}_candidate_beta.tsv.gz', sep='\t')
        fig, ax = plt.subplots(figsize=(9, 7))
        pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), PCA(n_components=2))
        pcs = pipe.fit_transform(matrix[panel.CpG_ID])
        pca = pipe[-1]
        for label, color in [(0, '#4c78a8'), (1, '#e45756')]:
            keep = meta.label.eq(label)
            ax.scatter(pcs[keep, 0], pcs[keep, 1], s=16, alpha=.6, color=color,
                       label='Healthy' if label == 0 else 'Disease')
        ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
               ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
               title=f'{cohort}: descriptive PCA of the selected panel')
        ax.legend();save(fig, out, cohort.lower()+'_candidate_pca')
        pd.DataFrame(pcs, index=matrix.index, columns=['PC1','PC2']).join(meta[['label','series']]).to_csv(
            out/'data'/f'{cohort.lower()}_pca.tsv', sep='\t')
        if cohort == 'External':
            disease='BC' if 'BC' in internal else 'AML'
            fig=external_distribution_figure(panel,matrix,meta,disease)
            save_publication(fig,out,'external_cpg_distributions')
            for previous in (out/'plots').glob('external_cpg_distributions_[0-9]*.*'):
                previous.unlink()
        else:
            # Multipage per-CpG distributions make interpatient heterogeneity visible.
            for start in range(0, len(panel), 12):
                subset = list(panel.CpG_ID.iloc[start:start+12])
                fig, axes = plt.subplots(int(np.ceil(len(subset)/3)), 3, figsize=(14, 3.2*np.ceil(len(subset)/3)), squeeze=False)
                for ax, cpg in zip(axes.flat, subset):
                    groups = [matrix.loc[meta.label.eq(k), cpg].dropna() for k in (0,1)]
                    ax.boxplot(groups, showfliers=False)
                    ax.set_xticks([1,2], ['Healthy','Disease'])
                    ax.set(title=cpg, ylim=(-.03,1.03), ylabel='β')
                for ax in list(axes.flat)[len(subset):]: ax.set_visible(False)
                fig.suptitle(f'{cohort}: patient methylation distributions');fig.tight_layout()
                save(fig, out, f'{cohort.lower()}_cpg_distributions_{start//12+1}')
    fig=forest_figure(panel,'BC' if 'BC' in internal else 'AML')
    save_publication(fig,out,'delta_beta_forest')
    panel.to_csv(out/'data/panel_effects_and_intervals.tsv',sep='\t',index=False)
    return panel


def estimator():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=1., solver='liblinear', max_iter=2000, random_state=RANDOM_SEED))


def oof_predictions(matrix, labels, features, splits, top_k=None):
    """Every fit, imputation, scaling and optional ranking uses training patients only."""
    result = np.full(len(labels), np.nan)
    chosen=[]
    for fold, (train, test) in enumerate(splits):
        cols = list(features)
        if top_k is not None:
            training = matrix.iloc[train][cols]
            y = labels[train]
            ranked = (training.loc[y==1].mean()-training.loc[y==0].mean()).abs().sort_values(ascending=False)
            cols = list(ranked.index[:top_k])
        model=estimator();model.fit(matrix.iloc[train][cols], labels[train])
        result[test]=model.predict_proba(matrix.iloc[test][cols])[:,1]
        chosen.extend(dict(fold=fold+1,CpG_ID=c) for c in cols)
    return result, chosen


def score(labels, probabilities):
    fpr,tpr,_=roc_curve(labels, probabilities)
    return dict(AUROC=roc_auc_score(labels,probabilities), AUPRC=average_precision_score(labels,probabilities),
                Brier=brier_score_loss(labels,probabilities), sensitivity_at_95_specificity=float(tpr[fpr<=.05].max()))


def bootstrap_metrics(labels, probabilities, repeats=200):
    rng=np.random.default_rng(RANDOM_SEED); rows=[]
    for _ in range(repeats):
        idx=np.r_[rng.choice(np.where(labels==0)[0],sum(labels==0),replace=True),
                  rng.choice(np.where(labels==1)[0],sum(labels==1),replace=True)]
        rows.append(score(labels[idx],probabilities[idx]))
    frame=pd.DataFrame(rows)
    return {f'{c}_CI95_{side}':float(frame[c].quantile(q)) for c in frame for side,q in [('low',.025),('high',.975)]}


def panel_comparisons(original,ometa,external,emeta,candidates,seeds,pool,out,repeats,disease,external_panels=None):
    y=ometa.label.to_numpy(int);ey=emeta.label.to_numpy(int)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(original,y))
    fold_ids=np.zeros(len(y),int)
    for k,(_,test) in enumerate(splits):fold_ids[test]=k+1
    # All comparisons use identical patients, CpGs measured in both cohorts, and folds.
    common=set(original.columns)&set(external.columns)
    panels={'seeds_only':sorted(set(seeds)&common),'candidates_only':list(candidates),
            'seeds_plus_candidates':sorted((set(seeds)|set(candidates))&common)}
    if not panels['seeds_only']: raise ValueError('No externally measurable seed CpGs')
    records=[];prediction_rows=[];transfer={};oof={};size_rows=[]
    for name,cols in panels.items():
        probabilities,chosen=oof_predictions(original,y,cols,splits)
        oof[name]=probabilities
        records.append(dict(panel=name,cohort='original_patient_CV',n_features=len(cols),
                            **score(y,probabilities),**bootstrap_metrics(y,probabilities)))
        fitted=estimator().fit(original[cols],y)
        ep=fitted.predict_proba(external[cols])[:,1];transfer[name]=ep
        records.append(dict(panel=name,cohort='external_transfer_selected_cohort',n_features=len(cols),
                            **score(ey,ep),**bootstrap_metrics(ey,ep)))
        for cohort,meta,pr,folds in [('original_patient_CV',ometa,probabilities,fold_ids),
                                     ('external_transfer_selected_cohort',emeta,ep,np.zeros(len(ey),int))]:
            prediction_rows.extend(dict(sample_id=s,cohort=cohort,panel=name,label=int(label),
                                        probability=float(p),fold=int(f),series=study)
                                   for s,label,p,f,study in zip(meta.index,meta.label,pr,folds,meta.series))
        pd.DataFrame({'CpG_ID':cols}).to_csv(out/'data'/f'{name}_features.tsv',sep='\t',index=False)
        # Scaling is documented: coefficient per one SD in original training patients.
        if name=='candidates_only':
            pd.DataFrame({'CpG_ID':cols,'standardized_coefficient':fitted[-1].coef_[0]}).to_csv(
                out/'data/candidate_logistic_coefficients.tsv',sep='\t',index=False)
    # Graph-predicted negatives describe CpG groups, not healthy patient labels.
    # Both additional classifiers still predict disease vs healthy in patients.
    external_panels=external_panels or {}
    for name,cols in external_panels.items():
        missing=set(cols)-common
        if missing:raise ValueError(f'{name}: missing patient beta columns: {sorted(missing)}')
        if original[cols].isna().all().any() or external[cols].isna().all().any():
            raise ValueError(f'{name}: entirely missing CpG measurements')
        fitted=estimator().fit(original[cols],y)
        ep=fitted.predict_proba(external[cols])[:,1]
        transfer[name]=ep
        records.append(dict(panel=name,cohort='external_transfer_selected_cohort',n_features=len(cols),
                            **score(ey,ep),**bootstrap_metrics(ey,ep)))
        prediction_rows.extend(dict(sample_id=sample,cohort='external_transfer_selected_cohort',
                                    panel=name,label=int(label),probability=float(prob),fold=0,series=study)
                               for sample,label,prob,study in zip(emeta.index,ey,ep,emeta.series))
        pd.DataFrame({'CpG_ID':cols}).to_csv(out/'data'/f'{name}_features.tsv',sep='\t',index=False)
    metrics=pd.DataFrame(records);metrics.to_csv(out/'data/panel_performance.tsv',sep='\t',index=False)
    pd.DataFrame(prediction_rows).to_csv(out/'data/patient_predictions.tsv',sep='\t',index=False)
    pd.DataFrame({'sample_id':original.index,'fold':fold_ids,'label':y}).to_csv(out/'data/patient_cv_folds.tsv',sep='\t',index=False)
    for cohort,labels,preds in [('original_patient_CV',y,oof),('external_transfer_selected_cohort',ey,transfer)]:
        fig=performance_figure(labels,preds,{name:len(cols) for name,cols in (panels|external_panels).items()},cohort,disease)
        save_publication(fig,out,cohort+'_performance')
    # Pair patients in each bootstrap so the added-value comparison is paired.
    rng=np.random.default_rng(RANDOM_SEED);diff=[]
    for _ in range(500):
        idx=np.r_[rng.choice(np.where(y==0)[0],sum(y==0),replace=True),rng.choice(np.where(y==1)[0],sum(y==1),replace=True)]
        diff.append(roc_auc_score(y[idx],oof['seeds_plus_candidates'][idx])-roc_auc_score(y[idx],oof['seeds_only'][idx]))
    pd.DataFrame([dict(comparison='seeds_plus_candidates minus seeds_only',
                       delta_AUROC=roc_auc_score(y,oof['seeds_plus_candidates'])-roc_auc_score(y,oof['seeds_only']),
                       CI95_low=np.quantile(diff,.025),CI95_high=np.quantile(diff,.975),
                       interpretation='Conditional on fixed panel and fitted OOF predictions; not selection-adjusted')]).to_csv(
        out/'data/incremental_value.tsv',sep='\t',index=False)
    # Size is ranked inside each fold, never by held-out patient effects.
    ranks=[]
    for size in sorted(set([1,3,5,10,len(candidates)])):
        if size>len(candidates):continue
        p,selected=oof_predictions(original,y,candidates,splits,top_k=size)
        size_rows.append(dict(n_features=size,**score(y,p),**bootstrap_metrics(y,p)))
        ranks.extend(dict(panel_size=size,**r) for r in selected)
    sizes=pd.DataFrame(size_rows);sizes.to_csv(out/'data/panel_size_performance.tsv',sep='\t',index=False)
    pd.DataFrame(ranks).to_csv(out/'data/panel_size_training_fold_selections.tsv',sep='\t',index=False)
    fig,ax=plt.subplots(figsize=(8,6));ax.errorbar(sizes.n_features,sizes.AUROC,
        yerr=[sizes.AUROC-sizes.AUROC_CI95_low,sizes.AUROC_CI95_high-sizes.AUROC],fmt='o-',capsize=3)
    ax.set(xlabel='Number of candidate CpGs (ranked within training folds)',ylabel='Patient CV AUROC',
           title='Exploratory panel-size comparison',ylim=(0,1.02));save(fig,out,'panel_size_curve')
    # Match to mean and SD without disease labels, using each fold's training patients.
    pool=sorted(set(pool)&common-set(candidates)-set(seeds))
    if len(pool)<len(candidates):raise ValueError('Insufficient random-panel background')
    random_scores=[];matched_records=[]
    for rep in range(repeats):
        p=np.full(len(y),np.nan)
        for fold,(train,test) in enumerate(splits):
            features=original.iloc[train][list(candidates)+pool]
            stats=np.column_stack([features.mean(),features.std()])
            scale=np.nanstd(stats,axis=0);scale[scale==0]=1
            distance=np.nansum(((stats[:len(candidates),None,:]-stats[None,len(candidates):,:])/scale)**2,axis=2)
            rng=np.random.default_rng(RANDOM_SEED+rep*100+fold);used=set();cols=[]
            for i in rng.permutation(len(candidates)):
                nearest=[pool[j] for j in np.argsort(distance[i]) if pool[j] not in used][:20]
                chosen=str(rng.choice(nearest));cols.append(chosen);used.add(chosen)
                matched_records.append(dict(repeat=rep+1,fold=fold+1,candidate=candidates[i],matched_CpG=chosen))
            fitted=estimator().fit(original.iloc[train][cols],y[train])
            p[test]=fitted.predict_proba(original.iloc[test][cols])[:,1]
        random_scores.append(dict(repeat=rep+1,**score(y,p)))
    random_frame=pd.DataFrame(random_scores);random_frame.to_csv(out/'data/matched_random_panel_performance.tsv',sep='\t',index=False)
    pd.DataFrame(matched_records).to_csv(out/'data/matched_random_panel_memberships.tsv',sep='\t',index=False)
    observed=roc_auc_score(y,oof['candidates_only'])
    fig,ax=plt.subplots(figsize=(8,6));ax.hist(random_frame.AUROC,bins=min(12,repeats),alpha=.7,color='#8ba9c4')
    ax.axvline(observed,color='#e45756',lw=2,label=f'Candidate panel: {observed:.3f}')
    ax.set(xlabel='Patient CV AUROC',ylabel='Matched random panels',title='Same-size, mean/SD-matched background comparison')
    ax.legend();save(fig,out,'matched_random_panel_comparison')
    return metrics,panels|external_panels


def bh(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);q=np.empty(len(p))
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q


def context_annotation(root):
    """Extract the GEO platform table once; do not scan the subsequent sample tables."""
    resource=Path(__file__).parent/'resources/GPL13534_context.tsv.gz'
    source=root.parent/'aml_analysis/raw/GPL13534_family.soft.gz'
    if not resource.exists() and source.exists():
        import os
        resource.parent.mkdir(exist_ok=True)
        temporary=resource.with_name(f'context_{os.getpid()}.tmp.gz')
        fields=['ID','UCSC_RefGene_Group','Relation_to_UCSC_CpG_Island','Enhancer','DHS']
        with gzip.open(source,'rt') as src,gzip.open(temporary,'wt') as target:
            for line in src:
                if line.startswith('!platform_table_begin'):break
            reader=csv.DictReader(src,delimiter='\t')
            writer=csv.DictWriter(target,fieldnames=fields,delimiter='\t',lineterminator='\n')
            writer.writeheader()
            for row in reader:
                if row['ID'].startswith('!platform_table_end'):break
                writer.writerow({k:row.get(k,'') for k in fields})
        temporary.replace(resource)
    return resource if resource.exists() else None


def annotation_and_pathways(root,panel,background,out,gmt=None,offline=False):
    # Optional resource failures must not leave an older pathway result visible.
    for previous in [out/'data/reactome_CpG_set_enrichment.tsv',
                     out/'plots/reactome_enrichment.png',out/'plots/reactome_enrichment.pdf',
                     out/'plots/reactome_enrichment.svg']:
        previous.unlink(missing_ok=True)
    candidates=list(panel.CpG_ID)
    annotation=root/'raw/GPL13534_probe_annotation.tsv'
    if not annotation.exists():annotation=root/'evaluation/raw/GPL13534_probe_annotation.tsv'
    ann=pd.read_csv(annotation,sep='\t',dtype=str).rename(columns={'ID':'CpG_ID'})
    ann=ann.drop_duplicates('CpG_ID').set_index('CpG_ID')
    genes_col=next((c for c in ['UCSC_RefGene_Name','Genes','Gene'] if c in ann),None)
    if genes_col is None:raise ValueError('Missing probe-to-gene annotation')
    selected=ann.reindex(candidates).reset_index()
    context=context_annotation(root)
    statuses={'pathways':'not run'}
    if context:
        details=pd.read_csv(context,sep='\t',dtype=str).rename(columns={'ID':'CpG_ID'})
        selected=selected.merge(details,on='CpG_ID',how='left',validate='one_to_one')
        statuses['promoter_gene_body_CpG_island']=dict(source=str(context),sha256=digest(context),
            origin='GEO GPL13534 full platform table; legacy annotation, not regulatory evidence')
    else:
        statuses['promoter_gene_body_CpG_island']='Unavailable: full GPL13534 context source not present'
    selected.to_csv(out/'data/candidate_genomic_annotation.tsv',sep='\t',index=False)
    disease = 'Breast cancer' if 'breast_cancer' in str(root) else 'AML'
    save_publication(annotation_figure(selected, disease), out, 'genomic_annotation')
    resources=root/'evaluation/post_evaluation_resources';resources.mkdir(parents=True,exist_ok=True)
    if gmt is None:
        gmt=resources/'ReactomePathways.gmt'
        if not gmt.exists() and not offline:
            try:
                print('Downloading official Reactome GMT resource',flush=True)
                archive=resources/'ReactomePathways.gmt.zip'
                with urlopen(Request(GMT_URL,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as source:archive.write_bytes(source.read())
                with zipfile.ZipFile(archive) as z:
                    filename=next(n for n in z.namelist() if n.endswith('.gmt'))
                    gmt.write_bytes(z.read(filename))
            except Exception as error:
                statuses['pathways']='Unavailable: '+str(error)
    if not Path(gmt).exists():
        (out/'data/annotation_analysis_status.json').write_text(json.dumps(statuses,indent=2))
        return statuses
    universe=sorted(set(background)&set(ann.index));universe_set=set(universe);selected_set=set(candidates)&universe_set
    # Test CpG membership, not a naive hypergeometric test on gene names.
    # Permutation of measured CpGs accounts for probe coverage and multi-gene mappings.
    gene_to_cpg={}
    for cpg,value in ann.loc[universe,genes_col].fillna('').items():
        for gene in set(g.strip() for g in value.split(';') if g.strip()):
            gene_to_cpg.setdefault(gene,set()).add(cpg)
    results=[]
    with Path(gmt).open() as f:
        for line in f:
            name,description,*genes=line.rstrip('\n').split('\t')
            mapped=set().union(*(gene_to_cpg.get(g,set()) for g in genes))
            k=len(mapped&selected_set);n=len(mapped)
            if n<10 or n>len(universe)*.5:continue
            # Exact CpG-set enrichment is equivalent to uniform size-matched
            # CpG permutation; all tested pathways enter BH, including zero hits.
            p=fisher_exact([[k,len(selected_set)-k],[n-k,len(universe)-n-len(selected_set)+k]],alternative='greater').pvalue
            results.append(dict(pathway=name,description=description,overlap_CpGs=k,
                                background_CpGs=n,p_value=p,
                                CpG_IDs=';'.join(sorted(mapped&selected_set)),
                                mapped_genes=';'.join(sorted(set(genes)&set(gene_to_cpg)))))
    result=pd.DataFrame(results)
    if len(result):
        result['FDR']=bh(result.p_value);result=result.sort_values(['FDR','p_value'])
        result.to_csv(out/'data/reactome_CpG_set_enrichment.tsv',sep='\t',index=False)
        save_publication(reactome_figure(result, len(selected_set), len(universe), disease),
                         out, 'reactome_enrichment')
    statuses['pathways']=dict(method='One-sided exact enrichment on CpG membership, BH over all tested pathways; '
                             'probe-level alternative to naive gene-list enrichment, not missMethyl gometh. '
                             'No matching for genomic context or CpG correlation; interpret cautiously.',
                             resource=str(gmt),sha256=digest(gmt),download_url=GMT_URL,
                             tested=len(result),significant_FDR_lt_0_05=int(result.FDR.lt(.05).sum()) if len(result) else 0,
                             background_CpGs=len(universe))
    (out/'data/annotation_analysis_status.json').write_text(json.dumps(statuses,indent=2))
    return statuses


def main(root=None):
    from omegaconf import OmegaConf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=['PNA'])
    parser.add_argument('--random-panels',type=int,default=30)
    parser.add_argument('--background-size',type=int,default=800)
    parser.add_argument('--pathway-gmt',type=Path)
    parser.add_argument('--offline',action='store_true',help='Use cached resources only, including expression')
    from .cross_cohort_pipeline import add_arguments,run as run_cross_cohort
    add_arguments(parser)
    args=parser.parse_args()
    if args.models!=['PNA']:parser.error('This fixed-panel stage supports only --models PNA')
    if args.random_panels<2 or args.background_size<100:parser.error('Use at least 2 random panels and 100 background CpGs')
    root=Path(root) if root else args.config.resolve().parent.parent
    config=OmegaConf.load(args.config.resolve());base=Path(config.p_base).resolve()
    disease='BC' if root.name=='breast_cancer_analysis' else 'AML';expected=33 if disease=='BC' else 17
    if args.post_evaluation_analysis=='cross-cohort':
        run_cross_cohort(root,config,args.rscript,args.offline,args.refresh_expression)
        return
    panel,source,internal,external_col=load_panel(base,disease,expected)
    out=base/'accumulated_results/post_evaluation'
    for name in ('data','plots'): (out/name).mkdir(parents=True,exist_ok=True)
    panel.to_csv(out/'data/fixed_candidate_panel.tsv',sep='\t',index=False)
    seed_frame=pd.read_csv(config.p_seeds,sep='\t')
    seeds=set(seed_frame['probeID' if disease=='BC' else 'CpG_ID'])
    shared=pd.read_csv(root/'evaluation/results/shared_cpg_statistics.tsv',sep='\t',usecols=['CpG_ID'])
    universe=sorted(set(shared.CpG_ID));rng=np.random.default_rng(RANDOM_SEED)
    available=sorted(set(universe)-seeds-set(panel.CpG_ID))
    pool=list(rng.choice(available,min(args.background_size,len(available)),replace=False))
    external_panels,external_panel_sources=load_common_prediction_panels(base,disease)
    wanted=set(pool)|seeds|set(panel.CpG_ID)|set().union(*(set(cols) for cols in external_panels.values()))
    original,ometa,external,emeta=get_matrices(root,config,wanted,out)
    missing=set(panel.CpG_ID)-set(original.columns)
    missing|=set(panel.CpG_ID)-set(external.columns)
    if missing:raise ValueError(f'Missing fixed panel sites: {sorted(missing)}')
    if original[panel.CpG_ID].isna().all().any() or external[panel.CpG_ID].isna().all().any():
        raise ValueError('A candidate has no measured patient beta values')
    panel['In_original_seed_list']=panel.CpG_ID.isin(seeds)
    panel['Original_healthy_SD']=panel.CpG_ID.map(original.loc[ometa.label.eq(0)].std(ddof=1))
    panel['Original_abs_delta_ge_0_30']=panel[internal].abs().ge(.30)
    panel=effects_and_heatmaps(panel,original,ometa,external,emeta,internal,external_col,out)
    suffix='PNA_'+base.name.removeprefix('data_')
    negative_source=base/'accumulated_results/evaluation'/f'negative_common_unique_cpgs_{suffix}.tsv'
    negatives=pd.read_csv(negative_source,sep='\t').sort_values(external_col,key=lambda x:x.abs(),ascending=False)
    if set(negatives.CpG_ID)!=set(external_panels['all_common_predicted_negative']):
        raise ValueError('Predicted-negative distribution panel differs from performance panel')
    fig=external_distribution_figure(negatives,external,emeta,disease,group_label='predicted-negative')
    save_publication(fig,out,'external_predicted_negative_cpg_distributions')
    external[negatives.CpG_ID].to_csv(out/'data/external_predicted_negative_beta.tsv.gz',sep='\t')
    negatives.to_csv(out/'data/predicted_negative_distribution_panel.tsv',sep='\t',index=False)
    print('Running patient-level model comparisons',flush=True)
    metrics,panels=panel_comparisons(original,ometa,external,emeta,list(panel.CpG_ID),seeds,pool,out,args.random_panels,disease,external_panels)
    statuses=annotation_and_pathways(root,panel,universe,out,args.pathway_gmt,args.offline)
    requested={'seeds_only':seeds,'candidates_only':set(panel.CpG_ID),
               'seeds_plus_candidates':seeds|set(panel.CpG_ID),
               **{name:set(cols) for name,cols in external_panels.items()}}
    coverage=pd.DataFrame([dict(panel=name,requested=len(requested[name]),
                               original_measured=len(requested[name]&set(original.columns)),
                               both_cohorts_measured=len(cols)) for name,cols in panels.items()])
    coverage.to_csv(out/'data/panel_coverage.tsv',sep='\t',index=False)
    confounding={}
    for name,meta in [('original',ometa),('external',emeta)]:
        table=pd.crosstab(meta.series,meta.label);table.to_csv(out/'data'/f'{name}_study_disease_counts.tsv',sep='\t')
        confounding[name]=bool((table.gt(0).sum(axis=1)==1).all())
    from importlib.metadata import version
    original_accessions=set(ometa.index.str.extract(r'(GSM\d+)',expand=False).dropna())
    external_accessions=set(emeta.index.str.extract(r'(GSM\d+)',expand=False).dropna())
    accession_overlap=sorted(original_accessions&external_accessions)
    if accession_overlap:raise ValueError(f'Original/external GEO accession overlap: {accession_overlap}')
    manifest=dict(disease=disease,panel_size=expected,panel_source=str(source),panel_source_sha256=digest(source),
                  fixed_CpGs=list(panel.CpG_ID),external_comparison_panels=external_panel_sources,
                  displayed_external_panels=['seeds_only','candidates_only'],seed_source=str(config.p_seeds),seed_sha256=digest(config.p_seeds),
                  threshold=dict(external_abs_delta_beta_ge=.30,external_p_value_lt=.01,external_FDR_lt=.01,
                                 same_nonzero_direction=True,common_training_runs=5),
                  patients=dict(original=len(original),external=len(external)),random_seed=RANDOM_SEED,
                  original_external_GSM_overlap=accession_overlap,
                  config_sha256=digest(args.config.resolve()),
                  matrix_input_fingerprints=json.loads((out/'cache/fingerprint.json').read_text())['inputs'],
                  random_panels=args.random_panels,background_pool_size=args.background_size,
                  patient_CV='5-fold stratified; fixed C=1 logistic regression; train-only imputation/scaling/ranking; identical folds across panels',
                  intervals='Stratified patient bootstrap conditional on fixed OOF/transfer scores; not full model/selection uncertainty',
                  fully_confounded_study_disease=confounding,annotation_analysis=statuses,
                  software=dict(python=sys.version,**{name:version(name) for name in ['numpy','pandas','scipy','matplotlib','scikit-learn','omegaconf']}),caution=CAUTION)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=[f'# {disease}: fixed {expected}-CpG post-evaluation', '',CAUTION,'',
           '## Inputs and selection','',
           'Fixed panel: common likely-positive CpGs in all five runs, |external Δβ| ≥ 0.30, external P < 0.01, external BH FDR < 0.01, and matching nonzero direction.',
           'Patient means were independently recomputed from raw beta matrices and checked against the selected CpG table.',
           '', '## Patient-level performance (exploratory)', '',
           '| Panel | Cohort | CpGs | AUROC | AUPRC | Brier | Sensitivity at 95% specificity |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.panel} | {row.cohort} | {row.n_features} | {row.AUROC:.3f} | {row.AUPRC:.3f} | {row.Brier:.3f} | {row.sensitivity_at_95_specificity:.3f} |')
    lines += ['', '## Interpretation and limits','',
              f'Study completely predicts disease in original cohort: {confounding["original"]}; external cohort: {confounding["external"]}.',
              'Heatmaps and PCA are descriptive views of sites selected for group differences, not independent evidence of classification.',
              'Seed comparison uses only seeds measured in both cohorts. See data/panel_coverage.tsv for exclusions.',
              'All fitting uses patient-level folds; graph-node splits and training runs are never treated as independent patient cohorts.',
              'Panel-size ranking and random-panel mean/SD matching use training-fold patients only. The candidate pool itself remains selected using full original and external cohorts.',
              'Random-panel results are descriptive matched-background comparisons, not selection-adjusted significance tests. Matching uses a reproducible sampled background; it is not exhaustive.',
              'The selected external cohort is a transfer stress test, not untouched validation. Patient bootstrap intervals do not capture selection uncertainty, study effects or model refitting.',
              'Predicted probabilities reflect these study sample proportions; no population prevalence or clinical risk interpretation is established.',
              'CpG-level pathway tests use all externally shared, annotated CpGs as background and control BH FDR across all tested pathways. They do not establish regulation or causality.',
              'Genomic context is extracted from the full local GPL13534 platform table when available; source/status is recorded. Multi-region mappings may count a CpG in multiple categories.',
              'Clinical decision curves, prognosis and disease-subtype claims require suitable outcomes, clinical thresholds and independent cohorts; none are invented.',
              '', '## Artifacts','',
              'Plots: forest, original/external patient heatmaps, PCA, candidate and common predicted-negative CpG distributions, ROC/PR comparisons, panel-size curve, matched-random baseline, genomic annotation and Reactome enrichment when available.',
              'External performance tables additionally compare all common likely-positive and predicted-negative CpGs from the unthresholded five-run delta-beta tables. The external ROC/PR figure displays seeds only and candidates only. All models predict disease vs healthy patients; predicted negative refers to CpG-node status, not patient status. No 0.30/P/FDR filter is applied to those two complete groups.',
              'Data: fixed panel, pointwise Welch intervals, patient betas/annotations, folds, predictions, performance with bootstrap intervals, paired incremental AUROC, panel-size selections, random-panel memberships, coverage, pathway results and annotation status.',
              'Reproducibility: manifest.json, input fingerprints, resource checksum and cache/. Pickle caches are local outputs, not third-party inputs.',
              '', '## Next independent validation','',
              'Freeze this panel and original-trained classifier, then evaluate new patients with cases and controls recruited within the same study. Refit or feature-select only inside a new development cohort, never on the untouched test cohort.']
    increment=pd.read_csv(out/'data/incremental_value.tsv',sep='\t').iloc[0]
    random_results=pd.read_csv(out/'data/matched_random_panel_performance.tsv',sep='\t')
    candidate_auc=float(metrics.loc[(metrics.panel=='candidates_only')&(metrics.cohort=='original_patient_CV'),'AUROC'].iloc[0])
    lines += ['', '## Findings for this fixed panel', '',
              f'- Original seed-list overlap: {int(panel.In_original_seed_list.sum())}/{len(panel)} CpGs.',
              f'- Original |Δβ| ≥ 0.30: {int(panel.Original_abs_delta_ge_0_30.sum())}/{len(panel)} CpGs; the selection cutoff applies to external effects.',
              f'- Median |original Δβ|: {panel[internal].abs().median():.3f}; median |external Δβ|: {panel[external_col].abs().median():.3f}.',
              f'- Added patient-CV AUROC from candidates: {increment.delta_AUROC:.4f} (conditional paired bootstrap interval {increment.CI95_low:.4f} to {increment.CI95_high:.4f}). A ceiling-level seed baseline limits measurable AUROC improvement.',
              f'- Candidate patient-CV AUROC: {candidate_auc:.3f}; matched random-panel median: {random_results.AUROC.median():.3f}, range {random_results.AUROC.min():.3f}–{random_results.AUROC.max():.3f}. This is a descriptive comparison, not a selection-adjusted P-value.',
              '- These findings do not establish that adding the panel improves clinical decisions. A compact candidate-only assay and incremental value beyond seeds are separate questions.']
    lines += ['', '## Figure gallery', '']
    lines += [f'- [{path.stem.replace("_", " ")}](plots/{path.name})' for path in sorted((out/'plots').glob('*.png'))]
    lines += ['', '## Biological resource status', '', '```json', json.dumps(statuses,indent=2), '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from .classifiers import compare_patient_classifiers
    compare_patient_classifiers(original,ometa,external,emeta,panels,out,disease)
    from .foundations import augment_foundation_models
    augment_foundation_models(root,args.config.resolve())
    run_cross_cohort(root,config,args.rscript,args.offline,args.refresh_expression)
    print(f'{disease}: post-evaluation complete -> {out}',flush=True)
