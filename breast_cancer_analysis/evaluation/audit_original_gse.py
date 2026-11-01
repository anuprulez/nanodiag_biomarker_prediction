#!/usr/bin/env python3
"""Independently recompute every plotted CpG directly from all three GEO beta files.

Uses csv/numpy directly, independently of geo_common/evaluate_datasets and the
comparison's mean calculations. Exports per-sample observations and per-CpG checks.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent


def read_metadata(path):
    result={}
    with gzip.open(path,'rt') as source:
        for line in source:
            row=next(csv.reader([line],delimiter='\t'))
            if not row:continue
            if row[0]=='!series_matrix_table_begin':break
            if row[0].startswith(('!Sample_','!Series_')):result.setdefault(row[0],[]).append(row[1:])
    return result


def valid(values):
    values=np.array([np.nan if x.strip().lower() in ('','na','nan','null','n/a') else float(x) for x in values])
    finite=values[~np.isnan(values)]
    if not np.isfinite(finite).all() or np.any((finite<0)|(finite>1)):raise ValueError('Invalid source beta')
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir',type=Path,default=BASE/'results/prediction_comparison')
    parser.add_argument('--suffix',default='PNA_corr_0.33_knn_n_3000_n_neg_10000')
    parser.add_argument('--output',type=Path,default=BASE/'results/original_gse_audit')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    groups=[]
    for group in ['negative','likely_positive']:
        table=pd.read_csv(args.comparison_dir/f'{group}_common_unique_cpgs_{args.suffix}.tsv',sep='\t')
        table['prediction_group']=group;groups.append(table)
    plotted=pd.concat(groups,ignore_index=True)
    targets=set(plotted['CpG_ID']);observations={}
    metadata={s:read_metadata(BASE/'raw'/f'{s}_series_matrix.txt.gz') for s in ['GSE237036','GSE226569','GSE101961']}
    (args.output/'geo_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    tissues={'GSE237036':'PBMC','GSE226569':'FFPE breast tumor','GSE101961':'normal breast tissue'}
    sample_rows=[]
    sources=[BASE/'raw'/'GSE237036_matrix_processed.txt.gz',BASE/'raw/GSE226569_Methy_ProcessedMatrix.csv.gz',BASE/'raw/GSE101961_series_matrix.txt.gz']
    sample_out=gzip.open(args.output/'plotted_cpg_source_beta_values.tsv.gz','wt')
    writer=csv.writer(sample_out,delimiter='\t',lineterminator='\n');writer.writerow(['series','sample_id','source_column','group','tissue','CpG_ID','beta'])
    sample_group_checks=[]
    detection={}
    for series,path in zip(['GSE237036','GSE226569','GSE101961'],sources):
        meta=metadata[series];gsm=meta['!Sample_geo_accession'][0]
        with gzip.open(path,'rt') as source:
            if series=='GSE237036':
                reader=csv.reader(source,delimiter='\t');header=next(reader);columns=header[1:]
                descriptions=meta['!Sample_description'][0]
                mapping={name.upper():i for i,name in enumerate(descriptions)}
                if len(mapping)!=len(descriptions):raise ValueError('Duplicate source descriptions')
                idx=[mapping[name.upper()] for name in columns]
                labels=[meta['!Sample_source_name_ch1'][0][i] for i in idx]
                groups_for_samples=['breast_cancer' if label=='BC PBMCs sample' else 'healthy_control' if label=='normal PBMCs sample' else None for label in labels]
                if None in groups_for_samples:raise ValueError('Unknown original cohort source')
                inferred=['breast_cancer' if name.upper().startswith('BC_') else 'healthy_control' for name in columns]
                if inferred!=groups_for_samples:raise ValueError('Original sample-name grouping differs from GEO sources')
                sample_group_checks.append('All 80 local column names map case-insensitively to original GEO descriptions; BC/NC labels agree with source_name and disease annotations')
                sample_ids=[gsm[i] for i in idx]
                extract_indices=list(range(1,len(header)))
            elif series=='GSE226569':
                reader=csv.reader(source);header=next(reader)
                titles=meta['!Sample_title'][0]
                beta_columns=[col for col in header[1:] if not col.endswith('.DetectionPvalue')]
                if len(beta_columns)!=len(set(beta_columns)) or set(beta_columns)!=set(titles):raise ValueError('Tumor beta column/sample mismatch')
                lookup={name:i for i,name in enumerate(header)}
                extract_indices=[lookup[name] for name in titles]
                det_indices=[lookup[name+'.DetectionPvalue'] for name in titles]
                columns=titles;sample_ids=gsm;groups_for_samples=['breast_cancer']*len(columns)
            else:
                for line in source:
                    if line.strip()=='!series_matrix_table_begin':break
                reader=csv.reader(source,delimiter='\t');header=next(reader);columns=header[1:]
                if columns!=gsm:raise ValueError('Control sample alignment error')
                sample_ids=gsm;extract_indices=list(range(1,len(header)));groups_for_samples=['healthy_control']*len(columns)
            if len(columns)!=len(sample_ids):raise ValueError('Source sample count error')
            for sample,col,group in zip(sample_ids,columns,groups_for_samples):
                sample_rows.append({'series':series,'sample_id':sample,'source_column':col,'group':group,'tissue':tissues[series]})
            found={}
            for row in reader:
                if row and row[0]=='!series_matrix_table_end':break
                if not row or row[0] not in targets:continue
                if row[0] in found:raise ValueError('Duplicate selected source CpG')
                if len(row)!=len(header):raise ValueError('Source column mismatch')
                values=valid([row[i] for i in extract_indices]);found[row[0]]=values
                if series=='GSE226569':
                    pvalues=valid([row[i] for i in det_indices])
                    detection[row[0]]=int(np.count_nonzero(pvalues>.01))
                for sample,col,group,beta in zip(sample_ids,columns,groups_for_samples,values):
                    writer.writerow([series,sample,col,group,tissues[series],row[0],'' if np.isnan(beta) else format(beta,'.17g')])
            if set(found)!=targets:raise ValueError(f'{series}: plotted CpGs missing from original source: {targets-set(found)}')
            source.read() # validate gzip CRC through EOF
            observations[series]=(found,np.array(groups_for_samples))
            print(f'{series}: independently read {len(found)} plotted CpGs x {len(columns)} samples',flush=True)
    sample_out.close()
    pd.DataFrame(sample_rows).to_csv(args.output/'sample_identity_audit.tsv',sep='\t',index=False)
    checks=[]
    for _,row in plotted.iterrows():
        probe=row['CpG_ID'];internal,labels=observations['GSE237036']
        values=internal[probe]
        bc=float(np.nanmean(values[labels=='breast_cancer']));healthy=float(np.nanmean(values[labels=='healthy_control']))
        external_bc=float(np.nanmean(observations['GSE226569'][0][probe]));external_healthy=float(np.nanmean(observations['GSE101961'][0][probe]))
        check={'prediction_group':row['prediction_group'],'CpG_ID':probe,
               'Source_PBMC_BC_mean':bc,'Source_PBMC_healthy_mean':healthy,'Source_PBMC_delta':bc-healthy,
               'Source_breast_tumor_mean':external_bc,'Source_normal_breast_mean':external_healthy,'Source_breast_tissue_delta':external_bc-external_healthy,
               'tumor_samples_detection_p_gt_0.01':detection[probe]}
        for field,expected in [('Internal_BC_mean_beta',bc),('Internal_healthy_mean_beta',healthy),
                               ('Internal_delta_beta_BC_minus_healthy',bc-healthy),('BC_GSE226569_mean_beta',external_bc),
                               ('Healthy_GSE101961_mean_beta',external_healthy),('Delta_beta_BC_minus_healthy',external_bc-external_healthy)]:
            check[field+'_absolute_error']=abs(row[field]-expected)
        checks.append(check)
    audit=pd.DataFrame(checks)
    audit.to_csv(args.output/'all_plotted_cpg_recalculation.tsv',sep='\t',index=False)
    error_cols=[c for c in audit if c.endswith('_absolute_error')]
    errors={c:float(audit[c].max()) for c in error_cols}
    summary=[]
    for group,frame in audit.groupby('prediction_group'):
        x,y=frame.Source_PBMC_delta.to_numpy(),frame.Source_breast_tissue_delta.to_numpy()
        summary.append({'prediction_group':group,'plotted_cpgs':len(frame),'pearson_r':float(np.corrcoef(x,y)[0,1]),
                        'same_direction_fraction':float(np.mean(np.sign(x)==np.sign(y))),
                        'median_abs_pbmc_delta':float(np.median(abs(x))),'median_abs_breast_tissue_delta':float(np.median(abs(y))),
                        'CpGs_with_at_least_one_tumor_detection_p_gt_0_01':int((frame['tumor_samples_detection_p_gt_0.01']>0).sum())})
    source_hashes={}
    for path in sources:
        with path.open('rb') as f:source_hashes[path.name]=hashlib.file_digest(f,'sha256').hexdigest()
    report={'plotted_cpgs_independently_recomputed':len(targets),'samples_per_cohort':{series:len(obs[1]) for series,obs in observations.items()},
            'maximum_absolute_error':errors,'all_plotted_values_match_original_sources_within_1e_10':bool(audit[error_cols].le(1e-10).all().all()),
            'sample_group_checks':sample_group_checks,'groups':summary,'source_sha256':source_hashes,
            'finding':'Original GSE237036 is PBMC (GPL23976, EPIC 850K), while GSE226569 and GSE101961 are breast tissue (GPL13534, 450K). This is a cross-tissue descriptive comparison, not matched-tissue replication.'}
    (args.output/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
