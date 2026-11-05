"""NIAPU class-2 CpGs, external methylation concordance and patient classification."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
from .analysis import RANDOM_SEED,digest,get_matrices,score,bootstrap_metrics
from .classifiers import fit_patient_model,classification_metrics,PARAMETERS


def load_rankings(path):
    """Use NIAPU class 2 directly, never GNN predictions or seed class 1."""
    frame=pd.read_csv(path,sep=r'\s+',header=None,names=['node_name','niapu_score','niapu_class'])
    if frame.empty or frame.node_name.duplicated().any():raise ValueError('NIAPU rankings must contain unique nodes')
    if not set(frame.niapu_class).issubset({1,2,3,4,5}) or not np.isfinite(frame.niapu_score).all():raise ValueError('Invalid NIAPU scores/classes')
    frame['CpG_ID']=frame.node_name.str.extract(r'^(cg\d+)(?:_|$)',expand=False)
    if frame.CpG_ID.isna().any():raise ValueError('NIAPU nodes must use cg probe identifiers')
    frame['annotated_gene']=frame.node_name.str.replace(r'^cg\d+_?','',regex=True)
    lp=frame.loc[frame.niapu_class.eq(2)].copy()
    if lp.empty:raise ValueError('No NIAPU class-2 likely positives')
    cpgs=lp.groupby('CpG_ID',sort=True).agg(niapu_score=('niapu_score','max'),
        niapu_node_count=('node_name','size'),annotated_genes=('annotated_gene',lambda v:';'.join(sorted(set(v)-{''}))),
        niapu_nodes=('node_name',lambda v:';'.join(sorted(v)))).reset_index()
    cpgs=cpgs.sort_values(['niapu_score','CpG_ID'],ascending=[False,True]).reset_index(drop=True)
    cpgs.insert(0,'niapu_rank',np.arange(1,len(cpgs)+1))
    return frame,lp,cpgs


def select_candidates(frame,internal,external):
    result=frame.copy()
    valid=np.isfinite(result[[internal,external]]).all(axis=1)
    result['Same_nonzero_direction']=valid&result[internal].ne(0)&result[external].ne(0)&np.sign(result[internal]).eq(np.sign(result[external]))
    result['Meets_exploratory_criterion']=(valid&result[external].abs().ge(.30)&result.External_p_value.lt(.01)&result.External_FDR.lt(.01)&result.Same_nonzero_direction)
    return result


def replication_module(root):
    spec=importlib.util.spec_from_file_location('niapu_replication_'+root.name,root/'evaluation/replication_metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare(root,config,out):
    disease='BC' if root.name=='breast_cancer_analysis' else 'AML'
    for name in ['data','plots','models','cache']:(out/name).mkdir(parents=True,exist_ok=True)
    ranking_path=Path(config.p_out_gene_rankings)
    nodes,lp,cpgs=load_rankings(ranking_path)
    nodes.to_csv(out/'data/all_niapu_nodes.tsv',sep='\t',index=False)
    lp.to_csv(out/'data/niapu_likely_positive_nodes.tsv',sep='\t',index=False)
    cpgs.to_csv(out/'data/niapu_likely_positive_unique_CpGs.tsv',sep='\t',index=False)
    reference=root/'evaluation/results/shared_cpg_statistics.tsv'
    if not reference.exists():raise FileNotFoundError(f'{reference}: prepare evaluation/download_datasets.py and evaluation/evaluate_datasets.py first')
    rep=replication_module(root)
    universe=rep.external_inference(pd.read_csv(reference,sep='\t'))
    universe.to_csv(out/'data/external_full_test_universe.tsv.gz',sep='\t',index=False)
    common=cpgs.merge(universe,on='CpG_ID',validate='one_to_one',how='inner')
    if common.empty:raise ValueError('No NIAPU likely positives measured in the external cohorts')
    cpgs.loc[~cpgs.CpG_ID.isin(common.CpG_ID)].to_csv(out/'data/likely_positive_not_in_external.tsv',sep='\t',index=False)
    print(f'{disease}: NIAPU class 2: {len(lp)} nodes, {len(cpgs)} unique CpGs; {len(common)} measured externally',flush=True)
    original,ometa,external,emeta=get_matrices(root,config,set(common.CpG_ID),out)
    ometa=ometa.loc[original.index];emeta=emeta.loc[external.index]
    if set(ometa.label)!={0,1} or set(emeta.label)!={0,1}:raise ValueError('Both patient classes are required')
    if not original.index.is_unique or not external.index.is_unique:raise ValueError('Repeated patient IDs')
    if set(original.index)&set(external.index):raise ValueError('Original/external sample IDs overlap')
    internal=rep.INTERNAL;ex=rep.EXTERNAL
    common[internal]=common.CpG_ID.map(original.loc[ometa.label.eq(1)].mean()-original.loc[ometa.label.eq(0)].mean())
    actual=external.loc[emeta.label.eq(1)].mean()-external.loc[emeta.label.eq(0)].mean()
    np.testing.assert_allclose(common[ex],common.CpG_ID.map(actual),atol=2e-9,rtol=0)
    common=select_candidates(common,internal,ex)
    common.to_csv(out/'data/all_common_likely_positive_CpGs.tsv',sep='\t',index=False)
    selected=common.loc[common.Meets_exploratory_criterion].sort_values('niapu_rank').copy()
    if selected.empty:raise ValueError('No NIAPU CpGs pass the fixed external criterion; no classifiers fitted')
    features=selected.CpG_ID.tolist()
    if original[features].isna().all().any() or external[features].isna().all().any():raise ValueError('Selected CpG lacks patient measurements')
    selected.to_csv(out/'data/selected_niapu_CpGs.tsv',sep='\t',index=False)
    pd.DataFrame({'CpG_ID':features}).to_csv(out/'data/candidates_only_features.tsv',sep='\t',index=False)
    ometa.to_csv(out/'data/original_patient_annotations.tsv',sep='\t');emeta.to_csv(out/'data/external_patient_annotations.tsv',sep='\t')
    for name,matrix in [('original',original),('external',external)]:matrix[features].to_csv(out/'data'/f'{name}_selected_beta.tsv.gz',sep='\t')
    for name,meta in [('original',ometa),('external',emeta)]:pd.crosstab(meta.series,meta.label).to_csv(out/'data'/f'{name}_study_disease_counts.tsv',sep='\t')
    audit=dict(disease=disease,ranking_source=str(ranking_path),ranking_sha256=digest(ranking_path),
        source_type='NIAPU preprocessing label propagation; class 2 only; no GNN probability or five-run intersection',
        niapu_nodes=len(nodes),likely_positive_nodes=len(lp),likely_positive_unique_CpGs=len(cpgs),common_CpGs=len(common),selected_CpGs=len(selected),
        external_FDR_family=len(universe),external_FDR_method='Welch tests and BH on all shared external CpGs before NIAPU filtering; untestable p retained as 1',
        criterion='abs(external delta beta)>=0.30; raw p<0.01; BH FDR<0.01; same nonzero original/external direction',
        CpG_score_aggregation='maximum class-2 node score; display order only, no score cutoff',
        original_n=ometa.label.value_counts().sort_index().to_dict(),external_n=emeta.label.value_counts().sort_index().to_dict(),
        input_sha256={str(p):digest(p) for p in [ranking_path,reference,Path(config.p_sample_metadata)]})
    (out/'selection_manifest.json').write_text(json.dumps(audit,indent=2))
    print(f'{disease}: {len(selected)} NIAPU CpGs pass criterion',flush=True)
    return disease,features,original,ometa,external,emeta,audit


def prediction_rows(model,meta,labels,probabilities,folds,cohort):
    return pd.DataFrame([dict(sample_id=s,panel='candidates_only',model=model,cohort=cohort,label=int(y),
        probability=float(p),predicted_label_at_0_5=int(p>=.5),fold=int(fold),series=study)
        for s,y,p,fold,study in zip(meta.index,labels,probabilities,folds,meta.series)])


def classical_predictions(original,ometa,external,emeta,features,out,disease):
    y=ometa.label.to_numpy(int);ey=emeta.label.to_numpy(int)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(original,y))
    fold_ids=np.zeros(len(y),int)
    for k,(_,test) in enumerate(folds,1):fold_ids[test]=k
    results=[]
    for name in ['Random Forest','XGBoost']:
        key=name.lower().replace(' ','_');cache=out/'cache'/f'{key}_predictions.tsv';stamp=out/'cache'/f'{key}_fingerprint.json';artifact=out/'models'/f'candidates_only_{key}.joblib'
        inputs=dict(parameters=PARAMETERS[name],original=digest(out/'cache/original.pkl'),external=digest(out/'cache/external.pkl'),
                    features=features,labels=y.tolist(),external_labels=ey.tolist(),folds=fold_ids.tolist(),version=version('scikit-learn' if name=='Random Forest' else 'xgboost'))
        token=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        if cache.exists() and stamp.exists() and artifact.exists() and json.loads(stamp.read_text()).get('token')==token:
            results.append(pd.read_csv(cache,sep='\t'));print(f'{disease}: {name}: reuse predictions',flush=True);continue
        p=np.full(len(y),np.nan)
        for k,(train,test) in enumerate(folds,1):
            print(f'{disease}: {name}: original-patient fold {k}/5',flush=True)
            fitted=fit_patient_model(name,original.iloc[train][features],y[train]);p[test]=fitted.predict_proba(original.iloc[test][features])[:,1]
        fitted=fit_patient_model(name,original[features],y);ep=fitted.predict_proba(external[features])[:,1]
        joblib.dump(fitted,artifact,compress=3)
        frame=pd.concat([prediction_rows(name,ometa,y,p,fold_ids,'original_patient_CV'),prediction_rows(name,emeta,ey,ep,np.zeros(len(ey),int),'external_transfer_selected_cohort')],ignore_index=True)
        frame.to_csv(cache,sep='\t',index=False);stamp.write_text(json.dumps(dict(token=token,inputs=inputs),indent=2));results.append(frame)
    return pd.concat(results,ignore_index=True)


def run(root,config):
    root=Path(root).resolve();out=Path(config.p_base).resolve()/'accumulated_results/niapu_comparison'
    out.mkdir(parents=True,exist_ok=True);status=out/'manifest.json'
    status.write_text(json.dumps({'status':'running'}))
    try:
        disease,features,original,ometa,external,emeta,audit=prepare(root,config,out)
        classical=classical_predictions(original,ometa,external,emeta,features,out,disease)
        from .foundations import foundation_predictions
        foundation,settings,artifacts=foundation_predictions(root,config,original,ometa,external,emeta,{'candidates_only':features},out,disease)
        predictions=pd.concat([classical,foundation],ignore_index=True)
        records=[]
        for (model,cohort),data in predictions.groupby(['model','cohort'],sort=False):
            y=data.label.to_numpy(int);p=data.probability.to_numpy(float)
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
            k=int((y==(p>=.5)).sum());ci=binomtest(k,len(y)).proportion_ci()
            records.append(dict(panel='candidates_only',model=model,cohort=cohort,n_features=len(features),n_patients=len(data),threshold=.5,
                                **score(y,p),**classification_metrics(y,p),accuracy_CI95_low=ci.low,accuracy_CI95_high=ci.high,**bootstrap_metrics(y,p)))
        predictions.to_csv(out/'data/classifier_patient_predictions.tsv',sep='\t',index=False)
        metrics=pd.DataFrame(records);metrics.to_csv(out/'data/classifier_performance.tsv',sep='\t',index=False)
        report=f'''# {disease}: NIAPU likely-positive methylation comparison

NIAPU class 2 only: {audit['likely_positive_nodes']} annotated nodes, {audit['likely_positive_unique_CpGs']} unique CpGs; {audit['common_CpGs']} measured externally; **{len(features)} pass the criterion**. The merged rankings combine disjoint preprocessing splits. There is no GNN prediction filter, top-N cutoff or five-training-run intersection. Seeds (class 1) and classes 3–5 are not selected as ranking nodes. One feature is one CpG, even with multiple gene annotations. Scores determine display order only.

Criterion: absolute external delta beta >=0.30, external Welch raw p<0.01, BH FDR<0.01 across all {audit['external_FDR_family']} shared external CpGs, and matching nonzero original/external directions. Original delta beta is recomputed from the original patient matrix. External contrasts are verified against raw patient matrices. See all_common_likely_positive_CpGs.tsv for both passing and failing sites.

Four classifiers: Random Forest, XGBoost, TabPFN and TabICL, with the same fixed hyperparameters/checkpoints as the candidate-panel comparison. Five-fold stratified CV uses original patients only. Final models are fitted on all original patients, then predict disease (1) vs healthy (0) in the external cohort at threshold 0.5. No class weighting or external fitting/tuning. Accuracy, balanced accuracy, sensitivity, specificity, F1, AUROC, AUPRC, Brier and confidence intervals are saved. Accuracy intervals are exact binomial intervals conditional on the selected features.

This panel uses external disease labels for feature selection, so external accuracy is exploratory and selection-biased, not an independent validation estimate. NIAPU preprocessing also used the original methylation cohort; original-patient CV is conditional on the already-selected panel. Case/control study and cell composition are confounded in the external cohorts, and in the original AML cohort. Reported confidence intervals do not account for feature-selection bias or study confounding.

Data sources: original and external methylation match the disease's existing evaluation pipeline (BC: GSE66695 original, GSE101961/GSE226569 external; AML: GSE50660/GSE58477 original, GSE40279/GSE62298 external). Expression data are not used in this comparison.

Outputs: data/selected_niapu_CpGs.tsv; data/all_common_likely_positive_CpGs.tsv; data/candidate_cohort_performance.tsv; data/candidate_cohort_predictions.tsv; candidate_cohort_comparison.pdf; plots/candidate_cohort_*.pdf/svg/png; fitted models and model/selection manifests. Existing post-evaluation results are not modified.
'''
        (out/'REPORT.md').write_text(report)
        from .candidate_comparison import render_candidate_comparison
        render_candidate_comparison(out,disease,comparison_label='NIAPU likely-positive')
        (out/'classifier_manifest.json').write_text(json.dumps(dict(classical_parameters={n:PARAMETERS[n] for n in ['Random Forest','XGBoost']},foundation_settings=settings,foundation_artifacts=artifacts,features=features,seed=RANDOM_SEED),indent=2))
        from .validate_niapu_rankings import main as validate
        validate(out)
        status.write_text(json.dumps(dict(status='complete',**audit,models=['Random Forest','XGBoost','TabPFN3','TabICLv2'],sample_level_target='disease vs healthy',class_weighting=False),indent=2))
        print(metrics[['model','cohort','n_features','accuracy_at_0_5','balanced_accuracy_at_0_5','AUROC']].to_string(index=False),flush=True)
        print(f'{disease}: NIAPU comparison complete -> {out}',flush=True)
    except Exception as error:
        status.write_text(json.dumps(dict(status='failed',error=str(error)),indent=2));raise
    return out



def resolve_config(root,config):
    """Preprocessing-only configs use paths relative to train_gnn, like preprocessing."""
    shared_path=root/'config/config.yaml'
    shared=OmegaConf.load(shared_path) if shared_path.exists() else OmegaConf.create({})
    defaults_path=root/'config/optuna.yaml'
    defaults=OmegaConf.load(defaults_path).get('runtime_defaults',{}) if defaults_path.exists() else {}
    config=OmegaConf.merge(shared,defaults,config)
    for key in ['p_base','p_arrays','p_sample_metadata','p_out_gene_rankings','tabpfn_model_path','tabicl_model_path']:
        value=config.get(key)
        if value is None or value=='auto':continue
        path=Path(value).expanduser()
        config[key]=str((path if path.is_absolute() else root/'train_gnn'/path).resolve())+('/' if key=='p_base' else '')
    return config


def main(root=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--models',nargs='+',default=['PNA'])
    parser.add_argument('--run',default='run1',help='Archived PNA run for UMAP (default: run1)')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--umap-only',action='store_true')
    group.add_argument('--skip-umap',action='store_true')
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'(?:run)?[1-9][0-9]*',args.run):parser.error('--run must be a positive run number or run<number>')
    run_name=args.run if args.run.startswith('run') else 'run'+args.run
    if args.models!=['PNA']:parser.error('NIAPU rankings do not select GNN models; omit --models or use PNA for pipeline compatibility')
    config=OmegaConf.load(args.config.resolve());root=Path(root).resolve() if root else args.config.resolve().parent.parent
    config=resolve_config(root,config)
    config.niapu_configuration_source=str(args.config.resolve())
    if not args.skip_umap:
        from .niapu_umap import run as run_umap
        run_umap(root,config,run_name)
    if not args.umap_only:run(root,config)

if __name__=='__main__':main()
