#!/usr/bin/env python3
"""Frozen 17-CpG AML transfer evaluation and donor-grouped CV on GSE63409."""
from pathlib import Path
import csv,gzip,json,hashlib,gc,sys
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_auc_score,average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from post_evaluation_support.foundations import foundation_estimator
MODELS={'Random Forest':'random_forest','XGBoost':'xgboost','TabPFN3':'TabPFN3','TabICLv2':'TabICLv2'}
SEED=20261031

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(parent,out,plan):
    source=out/'source/GSE63409_series_matrix.txt.gz'
    wanted=set(plan['features']);meta={};rows={}
    with gzip.open(source,'rt') as handle:
        for line in handle:
            fields=next(csv.reader([line],delimiter='\t'))
            if not fields:continue
            if fields[0]=='!series_matrix_table_begin':
                samples=next(csv.reader([next(handle)],delimiter='\t'))[1:];break
            if fields[0].startswith('!Sample_'):meta.setdefault(fields[0],[]).append(fields[1:])
        assert samples==meta['!Sample_geo_accession'][0] and len(set(samples))==74
        assert meta['!Sample_platform_id']==[['GPL13534']*74]
        for fields in csv.reader(handle,delimiter='\t'):
            if not fields:continue
            if fields[0]=='!series_matrix_table_end':break
            if fields[0] in wanted:
                assert fields[0] not in rows
                rows[fields[0]]=pd.to_numeric(pd.Series(fields[1:]),errors='coerce').to_numpy(float)
        else:raise ValueError('Truncated matrix')
    assert set(rows)==wanted, wanted-set(rows)
    x=pd.DataFrame(rows,index=pd.Index(samples,name='sample_id'))[plan['features']]
    assert np.isfinite(x.to_numpy()).all() and ((x>=0)&(x<=1)).all().all()
    chars=meta['!Sample_characteristics_ch1']
    def field(prefix):
        found=[row for row in chars if all(v.startswith(prefix+': ') for v in row)]
        assert len(found)==1
        return [v.split(': ',1)[1] for v in found[0]]
    status=field('subject status');print('Subject status:',set(status),flush=True)
    assert set(status)=={'normal','AML'},set(status)
    annotations=pd.DataFrame(dict(sample_id=samples,series='GSE63409',donor=field('subject id'),
        label=[int(v=='AML') for v in status],status=status,title=meta['!Sample_title'][0],phenotype=field('phenotype'))).set_index('sample_id')
    donors=annotations.groupby('donor').label.agg(['first','nunique'])
    assert donors['nunique'].eq(1).all()
    assert len(donors)==20 and donors['first'].sum()==15
    previous_ids=set();duplicates=[]
    for name in ['original','external']:
        old=pd.read_pickle(parent/f'cache/{name}.pkl')
        previous_ids.update(map(str,old.index))
        assert wanted.issubset(old.columns)
        a=old[plan['features']].to_numpy(float)
        for sid,row in x.iterrows():
            if np.any(np.all(np.isclose(a,row.to_numpy()[None,:],rtol=0,atol=1e-12,equal_nan=True),axis=1)):duplicates.append(sid)
    overlap=sorted(set(samples)&previous_ids)
    assert not overlap and not duplicates,(overlap,duplicates)
    for folder in ['data','models','cross_validation']:(out/folder).mkdir(exist_ok=True)
    x.to_csv(out/'data/independent_beta.tsv.gz',sep='\t',compression={'method':'gzip','mtime':0})
    annotations.to_csv(out/'data/sample_annotations.tsv',sep='\t')
    (out/'source/sample_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    audit=dict(n_specimens=74,AML_specimens=44,normal_specimens=30,AML_donors=15,normal_donors=5,n_features=17,
               missing_values=int(x.isna().sum().sum()),overlapping_accessions=overlap,duplicate_17_CpG_profiles=duplicates,
               source_sha256=digest(source),selection_history='No GSE63409 inputs found in existing AML feature-selection scripts/manifests; previous studies GSE58477/GSE50660 and GSE62298/GSE40279')
    (out/'data/input_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return x,annotations

def main():
    torch.set_num_threads(4)
    parent=ROOT/'data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation'
    out=ROOT/'evaluation/independent_classification/GSE63409'
    manifest=json.loads((parent/'classifier_comparison_manifest.json').read_text())
    features=manifest['features']['candidates_only'];assert len(features)==17
    plan=dict(dataset='GSE63409',features=features,model_settings={name:(manifest['models'][name] if name in manifest['models'] else manifest['foundation_models'][name]['parameters']) for name in MODELS},
              selection_basis='Same study AML and healthy marrow, matching 450K platform; chosen before inspecting classifier outcomes',
              CV='Five stratified donor folds; fit preprocessing and classifier on training donors only; fixed CpGs and hyperparameters; no external tuning',threshold=.5,
              source_models={name:dict(path=str(parent/f'models/candidates_only_{suffix}.joblib'),sha256=digest(parent/f'models/candidates_only_{suffix}.joblib')) for name,suffix in MODELS.items()})
    plan_path=out/'frozen_analysis_plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    x,meta=prepare(parent,out,plan)
    transfer=[]
    for name,suffix in MODELS.items():
        path=out/f'data/transfer_{suffix}.tsv'
        if path.exists():d=pd.read_csv(path,sep='\t')
        else:
            print('Transfer',name,flush=True)
            model=joblib.load(parent/f'models/candidates_only_{suffix}.joblib')
            assert list(model.feature_names_in_)==features
            np.testing.assert_array_equal(model.classes_,[0,1])
            p=model.predict_proba(x)[:,1]
            d=pd.DataFrame(dict(sample_id=x.index,model=name,label=meta.label.to_numpy(),donor=meta.donor.to_numpy(),probability=p,
                                predicted_label_at_0_5=(p>=.5).astype(int),cohort='independent_GSE63409',panel='candidates_only',series='GSE63409'))
            d.to_csv(path,sep='\t',index=False)
            del model;gc.collect();torch.cuda.empty_cache()
        transfer.append(d)
    pd.concat(transfer,ignore_index=True).to_csv(out/'data/transfer_predictions.tsv',sep='\t',index=False)
    donors=meta.groupby('donor',sort=True).label.first()
    donor_folds={}
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(donors)),donors),1):
        for donor in donors.index[te]:donor_folds[donor]=fold
    meta['fold']=meta.donor.map(donor_folds)
    assert meta.groupby('donor').fold.nunique().eq(1).all()
    assert meta.groupby('fold').label.nunique().eq(2).all()
    meta.to_csv(out/'cross_validation/fold_assignments.tsv',sep='\t')
    cv=[]
    for name,suffix in MODELS.items():
        for fold in range(1,6):
            train=np.flatnonzero(meta.fold.to_numpy()!=fold);test=np.flatnonzero(meta.fold.to_numpy()==fold)
            assert not set(meta.donor.iloc[train])&set(meta.donor.iloc[test])
            path=out/f'cross_validation/{suffix}_fold{fold}.tsv'
            if path.exists():d=pd.read_csv(path,sep='\t')
            else:
                print('CV',name,fold,flush=True)
                params=plan['model_settings'][name].copy()
                if name in ['TabPFN3','TabICLv2']:
                    params['device']='cuda' if torch.cuda.is_available() else 'cpu'
                    model=foundation_estimator(name,{'parameters':params})
                else:model=make_pipeline(SimpleImputer(strategy='median'),(RandomForestClassifier if name=='Random Forest' else XGBClassifier)(**params))
                model.fit(x.iloc[train],meta.label.iloc[train].to_numpy())
                p=model.predict_proba(x.iloc[test])[:,1]
                d=pd.DataFrame(dict(sample_id=x.index[test],model=name,label=meta.label.iloc[test].to_numpy(),donor=meta.donor.iloc[test].to_numpy(),
                    probability=p,predicted_label_at_0_5=(p>=.5).astype(int),cohort='GSE63409_donor_CV',fold=fold,panel='candidates_only',series='GSE63409'))
                d.to_csv(path,sep='\t',index=False)
                del model;gc.collect();torch.cuda.empty_cache()
            assert d.sample_id.tolist()==x.index[test].tolist()
            cv.append(d)
    cv=pd.concat(cv,ignore_index=True)
    cv.to_csv(out/'cross_validation/out_of_fold_predictions.tsv',sep='\t',index=False)
    all_data=pd.concat(transfer+[cv],ignore_index=True);metrics=[]
    for (cohort,name),d in all_data.groupby(['cohort','model'],sort=False):
        assert len(d)==74 and not d.sample_id.duplicated().any()
        assert np.isfinite(d.probability).all() and d.probability.between(0,1).all()
        assert np.array_equal(d.predicted_label_at_0_5,(d.probability>=.5).astype(int))
        tn,fp,fn,tp=confusion_matrix(d.label,d.predicted_label_at_0_5,labels=[0,1]).ravel()
        metrics.append(dict(cohort=cohort,model=name,n_specimens=len(d),n_donors=d.donor.nunique(),accuracy=accuracy_score(d.label,d.predicted_label_at_0_5),
            macro_F1=f1_score(d.label,d.predicted_label_at_0_5,average='macro'),AUROC=roc_auc_score(d.label,d.probability),AP=average_precision_score(d.label,d.probability),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'data/classification_metrics.tsv',sep='\t',index=False)
    (out/'evaluation_manifest.json').write_text(json.dumps(dict(plan_sha256=digest(plan_path),versions={p:version(p) for p in ['torch','scikit-learn','xgboost','tabpfn','tabicl']},
        CV_seed=SEED,CV_unit='donor; 15 AML patients and 5 normal donors',threshold=.5,
        caveat='Sorted AML blast/stem cell fractions versus normal hematopoietic progenitor fractions; cell composition differs. Confusion matrices count 74 cell-fraction specimens, not 74 people. No outcomes used for feature or model selection.'),indent=2)+'\n')
    print(metrics.to_string(index=False),flush=True)

if __name__=='__main__':main()
