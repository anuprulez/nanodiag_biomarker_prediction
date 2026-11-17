#!/usr/bin/env python3
"""Five-fold specimen CV on GSE52865 using the frozen 33-CpG panel/settings."""
from pathlib import Path
import gc
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from post_evaluation_support.foundations import foundation_estimator


def main():
    source=ROOT/'data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/independent_feature_verification'
    out=ROOT/'evaluation/independent_classification/GSE52865/cross_validation'
    out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((source/'frozen_analysis_plan.json').read_text())
    features=plan['panels']['candidates_only']
    x=pd.read_csv(source/'data/independent_beta.tsv.gz',sep='\t',index_col=0).loc[:,features]
    meta=pd.read_csv(source/'data/independent_sample_annotations.tsv',sep='\t',index_col=0).loc[x.index]
    y=meta.label.to_numpy(int)
    assert x.shape==(57,33) and len(set(x.index))==57 and y.sum()==40
    seed=20261031
    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(x,y))
    rows=[];scores=[]
    for name in ['Random Forest','XGBoost','TabPFN3','TabICLv2']:
        for fold,(train,test) in enumerate(splits,1):
            assert not set(train)&set(test)
            cache=out/f'{name.replace(" ","_")}_fold{fold}.tsv'
            if cache.exists():
                d=pd.read_csv(cache,sep='\t');assert d.sample_id.tolist()==x.index[test].tolist()
            else:
                params=plan['models'][name].copy()
                if name in ['TabPFN3','TabICLv2']:
                    params['device']='cuda' if torch.cuda.is_available() else 'cpu'
                    model=foundation_estimator(name,{'parameters':params})
                else:
                    cls=RandomForestClassifier if name=='Random Forest' else XGBClassifier
                    model=make_pipeline(SimpleImputer(strategy='median'),cls(**params))
                model.fit(x.iloc[train],y[train])
                probabilities=model.predict_proba(x.iloc[test])[:,1]
                d=pd.DataFrame(dict(sample_id=x.index[test],model=name,fold=fold,label=y[test],probability=probabilities,
                                    predicted_label_at_0_5=(probabilities>=.5).astype(int),cohort='GSE52865_patient_CV',panel='candidates_only'))
                d.to_csv(cache,sep='\t',index=False)
                del model;gc.collect();torch.cuda.empty_cache()
            assert np.array_equal(d.label,y[test])
            rows.append(d)
            scores.append(dict(model=name,fold=fold,n_train=len(train),n_test=len(test),accuracy=accuracy_score(d.label,d.predicted_label_at_0_5)))
            print(name,'fold',fold,'accuracy',scores[-1]['accuracy'],flush=True)
    predictions=pd.concat(rows,ignore_index=True)
    assert not predictions.duplicated(['sample_id','model']).any()
    assert predictions.groupby('model').size().eq(57).all()
    predictions.to_csv(out/'out_of_fold_predictions.tsv',sep='\t',index=False)
    pd.DataFrame(scores).to_csv(out/'fold_accuracy.tsv',sep='\t',index=False)
    summary=[]
    for name,d in predictions.groupby('model',sort=False):
        folds=[r['accuracy'] for r in scores if r['model']==name]
        summary.append(dict(model=name,pooled_oof_accuracy=accuracy_score(d.label,d.predicted_label_at_0_5),
                            mean_fold_accuracy=np.mean(folds),sd_fold_accuracy=np.std(folds,ddof=1),
                            macro_F1=f1_score(d.label,d.predicted_label_at_0_5,average='macro'),n=57))
    pd.DataFrame(summary).to_csv(out/'accuracy_summary.tsv',sep='\t',index=False)
    manifest=dict(dataset='GSE52865',n_features=33,folds=5,splitter='StratifiedKFold',shuffle=True,random_seed=seed,
        evaluation='Pooled held-out specimen predictions; training-only median imputation; fixed threshold 0.5; no feature reselection or tuning',
        settings=plan['models'],sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'data/independent_beta.tsv.gz',source/'frozen_analysis_plan.json']},
        caveat='Specimen-level CV; validated donor identifiers unavailable. Some tumour and normal specimens may share donors. GSE52865 is now also used for within-cohort fitting in panel D; panel C remains the frozen original-trained transfer evaluation.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
