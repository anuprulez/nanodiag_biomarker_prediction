#!/usr/bin/env python3
"""Audit cached AML cohort metadata and frozen-panel sensitivity to control choice.

Reads existing inputs only. Writes separate audit artifacts; does not alter
training, feature selection, fixed panels, or existing evaluation outputs.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'preprocess_raw_data'))
from geo_common import read_header,beta_rows


def design_audit(frame,group_column,study_column):
    group=pd.get_dummies(frame[group_column],drop_first=True,dtype=float)
    study=pd.get_dummies(frame[study_column],drop_first=True,dtype=float)
    design=np.column_stack([np.ones(len(frame)),group,study])
    return dict(n=len(frame),columns=design.shape[1],rank=int(np.linalg.matrix_rank(design)),
                full_rank=bool(np.linalg.matrix_rank(design)==design.shape[1]))


def run(dataset,out):
    out.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(ROOT/'raw/harmonised/all_sample_metadata.tsv',sep='\t')
    selected=original.loc[original.selected].copy()
    extframes=[];summaries=[]
    for series,group in [('GSE40279','control'),('GSE62298','AML')]:
        meta=json.loads((ROOT/'evaluation/raw'/f'{series}_metadata.json').read_text())
        ids=meta['!Sample_geo_accession'][0]
        frame=pd.DataFrame(dict(sample_id=ids,series=series,group=group))
        for row in meta['!Sample_characteristics_ch1']:
            key=row[0].partition(': ')[0]
            frame[key]=[v.partition(': ')[2] for v in row]
        extframes.append(frame)
        item=dict(series=series,n=len(ids),platforms=sorted(set(meta['!Sample_platform_id'][0])),
                  processing=sorted(set(meta['!Sample_data_processing'][0])),available_characteristics=list(frame.columns[3:]))
        age=next((c for c in frame if c.startswith('age')),None)
        if age:
            a=pd.to_numeric(frame[age],errors='coerce');item['age']=dict(n=int(a.notna().sum()),min=float(a.min()),median=float(a.median()),max=float(a.max()))
        if 'dnmt3a mutation status' in frame:item['DNMT3A']=frame['dnmt3a mutation status'].value_counts().to_dict()
        summaries.append(item)
    external=pd.concat(extframes,ignore_index=True)
    original.groupby(['series','group','selected']).size().rename('n').to_csv(out/'original_cohort_counts.tsv',sep='\t')
    pd.crosstab(selected.series,selected.group).to_csv(out/'original_study_by_disease.tsv',sep='\t')
    pd.crosstab(external.series,external.group).to_csv(out/'external_study_by_disease.tsv',sep='\t')
    in_study=original.loc[original.series.eq('GSE58477')]
    pd.crosstab(in_study['characteristic:batch'],in_study.group).to_csv(out/'GSE58477_batch_by_group.tsv',sep='\t')
    expr=dataset/'accumulated_results/post_evaluation/methylation_expression/cross_cohort_comparison/data'
    emeta=pd.read_csv(expr/'GSE15061_sample_metadata.tsv',sep='\t')
    center=pd.crosstab(emeta.center,emeta.group);center.to_csv(out/'expression_center_by_group.tsv',sep='\t')
    assert emeta.biological_sample_id.is_unique and emeta.normalization_variant.eq('DS').all()
    panel=pd.read_csv(dataset/'accumulated_results/post_evaluation/data/fixed_candidate_panel.tsv',sep='\t')
    assert panel.CpG_ID.is_unique and len(panel)==17
    source=ROOT/'raw/raw/GSE58477_series_matrix.txt.gz'
    targets=set(panel.CpG_ID);rows=[]
    with gzip.open(source,'rt') as f:
        ids,_=read_header(f)
        idx=in_study.set_index('sample_id').loc[ids]
        cases=idx['characteristic:cell type'].eq('leukemic blast').to_numpy()
        controls=idx['characteristic:cell type'].eq('normal bone marrow progenitor cells').to_numpy()
        assert cases.sum()==62 and controls.sum()==10
        for probe,values in beta_rows(f,len(ids)):
            if probe not in targets:continue
            values=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(float)
            a=values[cases];b=values[controls];a=a[np.isfinite(a)];b=b[np.isfinite(b)]
            rows.append(dict(CpG_ID=probe,AML_n=len(a),CD34_control_n=len(b),AML_mean_beta=float(a.mean()),
                             CD34_control_mean_beta=float(b.mean()),within_study_delta_beta=float(a.mean()-b.mean()),
                             targeted_Welch_p=float(ttest_ind(a,b,equal_var=False).pvalue)))
            if len(rows)==len(targets):break
    assert len(rows)==17
    sensitivity=panel[['CpG_ID','Internal_AML_mean_beta','Internal_delta_beta_AML_minus_healthy','Delta_beta_AML_minus_healthy']].merge(pd.DataFrame(rows),on='CpG_ID',validate='one_to_one')
    np.testing.assert_allclose(sensitivity.AML_mean_beta,sensitivity.Internal_AML_mean_beta,atol=1e-8,rtol=0)
    sensitivity['targeted_BH_FDR_17']=multipletests(sensitivity.targeted_Welch_p,method='fdr_bh')[1]
    sensitivity['direction_matches_original']=np.sign(sensitivity.within_study_delta_beta)==np.sign(sensitivity.Internal_delta_beta_AML_minus_healthy)
    sensitivity['direction_matches_external']=np.sign(sensitivity.within_study_delta_beta)==np.sign(sensitivity.Delta_beta_AML_minus_healthy)
    sensitivity.to_csv(out/'frozen_17_same_study_control_sensitivity.tsv',sep='\t',index=False)
    ages=[]
    for series,frame in selected.groupby('series'):
        col='characteristic:age' if series=='GSE50660' else 'characteristic:age at diagnosis in years'
        age=pd.to_numeric(frame[col],errors='coerce')
        ages.append(dict(series=series,n=len(frame),age_n=int(age.notna().sum()),age_min=float(age.min()),age_median=float(age.median()),age_max=float(age.max()),
                         female_n=int(frame['characteristic:gender'].eq('Female').sum()),male_n=int(frame['characteristic:gender'].eq('Male').sum())))
    pd.DataFrame(ages).to_csv(out/'original_demographic_summary.tsv',sep='\t',index=False)
    smoking=selected.loc[selected.series.eq('GSE50660'),'characteristic:smoking (0, 1 and 2, which represent never, former and current smokers)'].dropna().astype(int).value_counts().sort_index().to_dict()
    overlap=sorted(set(selected.sample_id)&set(external.sample_id))
    audit=dict(status='complete',dataset=str(dataset),original_design=design_audit(selected,'group','series'),
               external_design=design_audit(external,'group','series'),expression_design=design_audit(emeta,'group','center'),
               original_external_shared_GSM_IDs=overlap,original_demographics=ages,control_smoking_code_counts=smoking,
               external_metadata=summaries,expression_n=len(emeta),expression_counts=emeta.group.value_counts().to_dict(),
               excluded_same_study_controls=10,
               sensitivity=dict(n=17,direction_matches_original=int(sensitivity.direction_matches_original.sum()),
                                direction_matches_external=int(sensitivity.direction_matches_external.sum()),
                                targeted_FDR_lt_0_05=int(sensitivity.targeted_BH_FDR_17.lt(.05).sum()),
                                original_median_abs_delta=float(sensitivity.Internal_delta_beta_AML_minus_healthy.abs().median()),
                                within_study_median_abs_delta=float(sensitivity.within_study_delta_beta.abs().median()),
                                inference='Post-selection sensitivity analysis; BH across these 17 only, not genome-wide or independent validation; no batch/cell-composition adjustment'),
               limitations='GSM disjointness does not prove donor independence. No raw-IDAT QC or re-normalization performed. Metadata availability does not establish biological matching.')
    (out/'metadata_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data_corr_0.33_knn_n_3000_n_neg_10000')
    parser.add_argument('--out',type=Path,default=ROOT/'evaluation/dataset_comparability')
    args=parser.parse_args();run(args.dataset.resolve(),args.out.resolve())
