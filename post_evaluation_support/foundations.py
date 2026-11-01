"""Exact original-analysis TabPFN3/TabICLv2 checkpoints in patient comparisons."""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
from importlib.metadata import version
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from .analysis import RANDOM_SEED, score, bootstrap_metrics, original_metadata, digest
from .classifiers import (FOUNDATION_MODELS, MODELS, balanced_training_weights, classification_metrics,
                          plotted_models, comparison_figure, single_curve_figure, confusion_figure,
                          render_weighted_unweighted_sets)
from .publication import save_publication


def prior_correct(probabilities, weights):
    probabilities=np.asarray(probabilities,float)
    result=probabilities*np.asarray(weights,float)
    return result/result.sum(axis=1,keepdims=True)


class PriorCorrectedFoundation:
    """Predict-only artifact: native foundation fit has no sample_weight API."""
    def __init__(self, model_path, class_weights):
        self.model_path=str(model_path)
        self.class_weights=np.asarray(class_weights,float)
        self.classes_=np.array([0,1])

    def predict_proba(self,x):
        model=joblib.load(self.model_path)
        try:return prior_correct(model.predict_proba(x),self.class_weights)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()

    def predict(self,x):
        return self.classes_[np.argmax(self.predict_proba(x),axis=1)]


def original_settings(root,config):
    path=root/'train_gnn/pretrained_checkpoints.py'
    spec=importlib.util.spec_from_file_location('patient_pretrained_checkpoints',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result={}
    for name,key in [('TabPFN3','TabPFN'),('TabICLv2','TabICL')]:
        checkpoint=module.ensure_checkpoint(key,config.get('tabpfn_model_path' if key=='TabPFN' else 'tabicl_model_path'))
        configured=config.get('model_overrides',{}).get(key,{})
        overrides=OmegaConf.to_container(configured,resolve=True) if OmegaConf.is_config(configured) else dict(configured)
        params=dict(n_estimators=int(config.get('foundation_estimators',4)),random_state=RANDOM_SEED,
                    device='cuda' if torch.cuda.is_available() else 'cpu',model_path=str(checkpoint))
        params.update(overrides.get('estimator_params',{}))
        if key=='TabICL':
            params.update(checkpoint_version='tabicl-classifier-v2-20260212.ckpt',allow_auto_download=False,n_jobs=4)
        # Foundation API built-in balancing is disabled; use one documented
        # post-ensemble formula for both models in the separate weighted set.
        if key=='TabPFN':params['balance_probabilities']=False
        result[name]=dict(parameters=params,checkpoint_sha256=module.file_digest(checkpoint),
                          package_version=version('tabpfn' if key=='TabPFN' else 'tabicl'),
                          settings_source=config.get('niapu_configuration_source',str(Path(config.p_base)/'config.yaml')))
    return result


def foundation_estimator(name,settings):
    if name=='TabPFN3':
        from tabpfn import TabPFNClassifier
        model=TabPFNClassifier(**settings['parameters'])
    else:
        from tabicl import TabICLClassifier
        model=TabICLClassifier(**settings['parameters'])
    return make_pipeline(SimpleImputer(strategy='median'),model)


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def foundation_predictions(root,config,original,ometa,external,emeta,panels,out,disease):
    settings=original_settings(root,config)
    y=ometa.label.to_numpy(int);ey=emeta.label.to_numpy(int)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(original,y))
    folds=np.zeros(len(y),int)
    for fold,(_,test) in enumerate(splits,1):folds[test]=fold
    cache=out/'cache/foundation_predictions';cache.mkdir(exist_ok=True)
    all_rows=[]; artifacts={}
    torch.set_num_threads(4)
    for panel,features in panels.items():
        for name in FOUNDATION_MODELS:
            key=panel+'_'+name
            fingerprint=dict(settings=settings[name],features=features,
                             original=digest(out/'cache/original.pkl'),external=digest(out/'cache/external.pkl'),
                             original_labels=y.tolist(),external_sample_ids=external.index.tolist(),
                             seed=RANDOM_SEED)
            token=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
            stamp=cache/(key+'.json');table=cache/(key+'.tsv')
            fitted_path=out/'models'/(key+'.joblib')
            if stamp.exists() and table.exists() and fitted_path.exists() and json.loads(stamp.read_text()).get('token')==token:
                print(f'{disease}: reuse {panel} / {name} patient predictions',flush=True)
                frame=pd.read_csv(table,sep='\t')
            else:
                probs=np.full(len(y),np.nan)
                for fold,(train,test) in enumerate(splits,1):
                    print(f'{disease}: {panel} / {name}: original-patient fold {fold}/5',flush=True)
                    model=foundation_estimator(name,settings[name])
                    model.fit(original.iloc[train][features],y[train])
                    probs[test]=model.predict_proba(original.iloc[test][features])[:,1]
                    del model;gc.collect()
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                print(f'{disease}: {panel} / {name}: final original context; external inference',flush=True)
                model=foundation_estimator(name,settings[name]);model.fit(original[features],y)
                ep=model.predict_proba(external[features])[:,1]
                joblib.dump(model,fitted_path,compress=3)
                del model;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                rows=[]
                for cohort,meta,labels,pred,f in [('original_patient_CV',ometa,y,probs,folds),
                                                ('external_transfer_selected_cohort',emeta,ey,ep,np.zeros(len(ey),int))]:
                    rows.extend(dict(sample_id=sample,panel=panel,model=name,cohort=cohort,label=int(label),
                                     probability=float(p),predicted_label_at_0_5=int(p>=.5),fold=int(k),series=study)
                                for sample,label,p,k,study in zip(meta.index,labels,pred,f,meta.series))
                frame=pd.DataFrame(rows);frame.to_csv(table,sep='\t',index=False)
                stamp.write_text(json.dumps(dict(token=token,inputs=fingerprint),indent=2)+'\n')
            all_rows.append(frame)
            artifacts[key]=dict(path=str(fitted_path),features=features,**settings[name])
    return pd.concat(all_rows,ignore_index=True),settings,artifacts


def weighted_foundation_predictions(raw,original_labels):
    """OOF correction uses each fold's training priors; external uses all original priors."""
    result=raw.copy()
    original_labels=np.asarray(original_labels,int)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(np.zeros(len(original_labels)),original_labels))
    weights={fold:balanced_training_weights(original_labels[train])[2] for fold,(train,_) in enumerate(splits,1)}
    weights[0]=balanced_training_weights(original_labels)[2]
    for fold,group in result.groupby('fold'):
        p=group.probability.to_numpy(float)
        corrected=prior_correct(np.column_stack([1-p,p]),weights[int(fold)])[:,1]
        result.loc[group.index,'probability']=corrected
        result.loc[group.index,'predicted_label_at_0_5']=(corrected>=.5).astype(int)
    return result


def write_augmented_results(parent,raw,settings,artifacts,y,disease):
    for weighted in (False,):
        out=parent/'inverse_class_weighted' if weighted else parent
        extra=weighted_foundation_predictions(raw,y) if weighted else raw
        old=pd.read_csv(out/'data/classifier_patient_predictions.tsv',sep='\t')
        baseline=old.loc[old.model.isin(MODELS)].copy()
        predictions=pd.concat([baseline,extra],ignore_index=True)
        records=[]
        for (cohort,panel,name),frame in predictions.groupby(['cohort','panel','model'],sort=False):
            labels=frame.label.to_numpy(int);p=frame.probability.to_numpy(float)
            n_features=len(artifacts[panel+'_'+name]['features']) if name in FOUNDATION_MODELS else int(
                pd.read_csv(out/'data/classifier_performance.tsv',sep='\t').query('panel==@panel and model==@name').n_features.iloc[0])
            records.append(dict(panel=panel,model=name,cohort=cohort,n_features=n_features,n_patients=len(frame),threshold=.5,
                                **score(labels,p),**classification_metrics(labels,p),**bootstrap_metrics(labels,p)))
        metrics=pd.DataFrame(records)
        predictions.to_csv(out/'data/classifier_patient_predictions.tsv',sep='\t',index=False)
        metrics.to_csv(out/'data/classifier_performance.tsv',sep='\t',index=False)
        pairs=[]
        for (cohort,panel),frame in predictions.groupby(['cohort','panel'],sort=False):
            pivot=frame.pivot(index='sample_id',columns='model',values='probability')
            labels=frame.drop_duplicates('sample_id').set_index('sample_id').loc[pivot.index,'label'].to_numpy(int)
            for left,right in combinations(plotted_models(predictions),2):
                a,b=pivot[left].to_numpy(),pivot[right].to_numpy();rng=np.random.default_rng(RANDOM_SEED);deltas=[]
                for _ in range(500):
                    idx=np.r_[rng.choice(np.where(labels==0)[0],sum(labels==0),replace=True),
                              rng.choice(np.where(labels==1)[0],sum(labels==1),replace=True)]
                    deltas.append(roc_auc_score(labels[idx],b[idx])-roc_auc_score(labels[idx],a[idx]))
                pairs.append(dict(panel=panel,cohort=cohort,reference=left,comparison=right,
                                  delta_AUROC=roc_auc_score(labels,b)-roc_auc_score(labels,a),
                                  delta_AUROC_CI95_low=np.quantile(deltas,.025),delta_AUROC_CI95_high=np.quantile(deltas,.975),
                                  probability_correlation=np.corrcoef(a,b)[0,1],class_agreement_at_0_5=float(((a>=.5)==(b>=.5)).mean())))
        pd.DataFrame(pairs).to_csv(out/'data/classifier_pairwise_comparisons.tsv',sep='\t',index=False)
        if weighted:
            prior_rows=[]
            folds=list(StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED).split(np.zeros(len(y)),y))
            for panel in ('seeds_only','candidates_only'):
                for name in FOUNDATION_MODELS:
                    for fold in range(6):
                        train_y=y if fold==0 else y[folds[fold-1][0]]
                        _,counts,weights=balanced_training_weights(train_y)
                        prior_rows.extend(dict(panel=panel,model=name,fold=fold,label=k,
                                               n_training=int(counts[k]),prior_correction_weight=float(weights[k]),
                                               method='post-ensemble probability correction') for k in (0,1))
            pd.DataFrame(prior_rows).to_csv(out/'data/foundation_prior_weights.tsv',sep='\t',index=False)
        manifest_path=out/'classifier_comparison_manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['foundation_models']=settings
        manifest['foundation_weighting']=('Post-ensemble probability correction p_k*w_k/sum(p_j*w_j), w_k=N_train/(2*n_k), using training-fold priors for CV and all original priors for external; not loss weighting or model retraining' if weighted else 'No class-prior correction')
        manifest['models'].update({name:item['parameters'] for name,item in settings.items()})
        for key,item in artifacts.items():
            if weighted:
                _,_,weights=balanced_training_weights(y)
                name=key+'.joblib';joblib.dump(PriorCorrectedFoundation(item['path'],weights),out/'models'/name)
                manifest['saved_models'][key]=dict(path='models/'+name,features=item['features'],method='Predict-only training-prior-correction wrapper',base_model=item['path'])
            else:
                manifest['saved_models'][key]=dict(path=item['path'],features=item['features'])
        manifest['versions'].update({n:version(n) for n in ['tabpfn','tabicl','torch']})
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        for cohort in predictions.cohort.unique():
            save_publication(comparison_figure(predictions,metrics,disease,cohort,weighted),out,cohort+'_classifier_comparison')
            if cohort=='external_transfer_selected_cohort':
                save_publication(comparison_figure(predictions,metrics,disease,cohort,weighted),out,cohort+'_performance')
            for kind in ('ROC','PRC'):
                save_publication(single_curve_figure(predictions,metrics,disease,cohort,kind,weighted),out,cohort+'_classifier_'+kind)
            save_publication(confusion_figure(predictions,disease,cohort,weighted,metrics.groupby('panel').n_features.first().to_dict()),out,cohort+'_classifier_confusion_matrices')
        save_publication(confusion_figure(predictions,disease,inverse_class_weighting=weighted,feature_counts=metrics.groupby('panel').n_features.first().to_dict()),out,'external_classifier_confusion_matrices')
        lines=[f'# {disease}: five-classifier comparison','',
               'Logistic regression, Random Forest, XGBoost, TabPFN3 and TabICLv2 use the same seed/candidate features, patients and five original-patient folds. External data are never used to fit the classifiers.',
               'Foundation checkpoints and estimator overrides are copied from the original dataset config and SHA256-verified. The patient-fold random seed is shared with the classical-model comparison.',
               '', '## Weighting','',
               manifest['foundation_weighting'],
               'The existing three models use sample-weighted training in the weighted set. The two foundation APIs have no sample_weight fit parameter; prior correction is explicitly a different method. External AUROC/AP ranking is unchanged by binary constant prior correction, while probability values and 0.50-threshold predictions can change.',
               '', '| Cohort | Panel | Model | CpGs | AUROC | AP | Brier | Balanced accuracy |',
               '|---|---|---|---:|---:|---:|---:|---:|']
        for row in metrics.itertuples():
            lines.append(f'| {row.cohort} | {row.panel} | {row.model} | {row.n_features} | {row.AUROC:.4f} | {row.AUPRC:.4f} | {row.Brier:.4f} | {row.balanced_accuracy_at_0_5:.4f} |')
        lines+=['', 'Candidate selection used external labels and disease is confounded with study; these are exploratory transfer comparisons, not untouched clinical validation.',
                '', '[External ROC/PR](plots/external_transfer_selected_cohort_performance.pdf)',
                '[External confusion matrices](plots/external_classifier_confusion_matrices.pdf)',
                '[Original-patient CV](plots/original_patient_CV_classifier_comparison.pdf)']
        (out/'CLASSIFIER_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    report=parent/'REPORT.md'
    if report.exists():
        text=report.read_text().replace('Three-classifier comparison','Five-classifier comparison')
        text=text.replace('logistic regression, Random Forest and XGBoost using',
                          'logistic regression, Random Forest, XGBoost, TabPFN3 and TabICLv2 using')
        if '## Foundation classifier weighting' not in text:
            text+='\n## Foundation classifier weighting\n\nTabPFN3 and TabICLv2 reuse the exact original-analysis checkpoints and overrides. Their weighted versions use training-prior probability correction because their fit APIs lack sample weights; this differs from loss/split weighting of the other three models. Checkpoint hashes, software versions and methods are recorded in each comparison manifest.\n'
        report.write_text(text)
    from .candidate_comparison import render_candidate_comparison
    render_candidate_comparison(parent,disease)


def augment_foundation_models(root,config_path):
    root=Path(root);config=OmegaConf.load(config_path)
    parent=Path(config.p_base).resolve()/'accumulated_results/post_evaluation'
    for sub in ('',):
        if not (parent/sub/'data/classifier_patient_predictions.tsv').exists():
            raise FileNotFoundError('Run classical-model comparisons before adding foundation classifiers')
    stamp=json.loads((parent/'cache/fingerprint.json').read_text())
    for path,size,mtime in stamp['inputs']['files']:
        p=Path(path)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime):raise ValueError('Stale patient cache; rerun post-evaluation')
    original=pd.read_pickle(parent/'cache/original.pkl');external=pd.read_pickle(parent/'cache/external.pkl')
    ometa=original_metadata(config,original)
    emeta=pd.read_csv(parent/'cache/external_metadata.tsv',sep='\t',index_col=0).loc[external.index]
    panels={panel:pd.read_csv(parent/'data'/f'{panel}_features.tsv',sep='\t').CpG_ID.tolist() for panel in ('seeds_only','candidates_only')}
    disease='BC' if root.name=='breast_cancer_analysis' else 'AML'
    raw,settings,artifacts=foundation_predictions(root,config,original,ometa,external,emeta,panels,parent,disease)
    write_augmented_results(parent,raw,settings,artifacts,ometa.label.to_numpy(int),disease)
    print(f'{disease}: five-model candidate cohort comparison complete',flush=True)


def main(root):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();augment_foundation_models(root,args.config)
