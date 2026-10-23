#!/usr/bin/env python3
"""Reconstruct MeinteR UseCase1 GSE66695 DMSs from original GEO beta values.

Author code uses group-wise available-case means and strict |delta| > 0.3.
Welch p values and BH FDR below are independently calculated, not author tables.
"""
import argparse
import gzip
import json
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from prepare_gse66695 import ROOT, digest, read_header, select_samples

ARTICLE='https://www.nature.com/articles/s41598-019-55453-8'
COMMIT='861be904ca2dddf61bdaa7d7bd741488e7f4473d'
CODE=f'https://github.com/andigoni/MeinteR/blob/{COMMIT}/vignettes/UseCase1.Rmd'


def bh_adjust(p):
    p=np.asarray(p,dtype=float)
    order=np.argsort(p); ranked=p[order]
    adjusted=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty_like(p);result[order]=np.minimum(adjusted,1)
    return result


def select_dms(frame):
    return frame.loc[(frame.deltaBeta.abs()>.3)&(frame['P.Value']<.01)&(frame.FDR<.01)].copy()


def build(raw,annotation):
    ann=pd.read_csv(annotation,sep='\t',dtype=str,keep_default_na=False).set_index('ID',verify_integrity=True)
    parts=[]
    with gzip.open(raw/'source/GSE66695_series_matrix.txt.gz','rt') as source:
        samples,meta=read_header(source);indices,_,_=select_samples(samples,meta)
        author=pd.read_csv(raw/'source/meinter/GSE66695_annotation.csv')
        labels=dict(zip(author['sample'],author.status))
        assert len(labels)==120 and set(labels)==set(samples)
        assert all(labels[s]==meta['!Sample_source_name_ch1'][0][i] for i,s in enumerate(samples))
        for chunk in pd.read_csv(source,sep='\t',header=None,names=['ID_REF',*samples],comment='!',chunksize=5000):
            chunk=chunk.loc[chunk.ID_REF.str.fullmatch(r'cg\d+')]
            values=chunk.iloc[:,[i+1 for i in indices]].to_numpy(float)
            bc,nc=values[:,:80],values[:,80:]
            finite=values[np.isfinite(values)]
            if ((finite<0)|(finite>1)).any():raise ValueError('Invalid beta values')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                bm,nm=np.nanmean(bc,axis=1),np.nanmean(nc,axis=1)
                p=ttest_ind_from_stats(bm,np.nanstd(bc,axis=1,ddof=1),np.isfinite(bc).sum(axis=1),
                    nm,np.nanstd(nc,axis=1,ddof=1),np.isfinite(nc).sum(axis=1),equal_var=False).pvalue
            parts.append(pd.DataFrame({'probeID':chunk.ID_REF.to_numpy(),'mean_beta_BC':bm,'mean_beta_normal':nm,
                'deltaBeta':bm-nm,'P.Value':p,'n_BC_measured':np.isfinite(bc).sum(axis=1),
                'n_normal_measured':np.isfinite(nc).sum(axis=1)}))
    all_stats=pd.concat(parts,ignore_index=True)
    # Untestable CpGs are assigned p=1, retaining all source cg probes in the BH family.
    all_stats['P.Value']=all_stats['P.Value'].fillna(1.)
    all_stats['FDR']=bh_adjust(all_stats['P.Value'])
    threshold=all_stats.loc[all_stats.deltaBeta.abs()>.3]
    seeds=select_dms(all_stats)
    assert set(threshold.probeID)==set(seeds.probeID),'P/FDR filters change published effect-size list'
    counts=[int((seeds.deltaBeta>0).sum()),int((seeds.deltaBeta<0).sum())]
    assert counts==[293,62],f'Cannot reproduce published counts: {counts}'
    assert not seeds.probeID.duplicated().any()
    mapped=ann.reindex(seeds.probeID).fillna('')
    seeds['CHR']=mapped.CHR.to_numpy()
    seeds['UCSC_RefGene_Name']=[';'.join(dict.fromkeys(g for g in s.split(';') if g)) for s in mapped.UCSC_RefGene_Name]
    seeds['DMS_class']=np.where(seeds.deltaBeta>0,'DMS+','DMS-')
    complete=set(pd.read_csv(raw/'probe_genes.csv',sep='\t',usecols=['probeID']).probeID)
    seeds['present_in_complete_training_matrix']=seeds.probeID.isin(complete)
    # Keep the full reconstructed literature list; existing preprocessing reports exclusions.
    columns=['probeID','deltaBeta','P.Value','CHR','UCSC_RefGene_Name','FDR','DMS_class',
             'mean_beta_BC','mean_beta_normal','n_BC_measured','n_normal_measured','present_in_complete_training_matrix']
    seeds=seeds[columns].sort_values(['DMS_class','probeID']).reset_index(drop=True)
    backup=raw/'seed_probes.before_meinter.csv'
    if (raw/'seed_probes.csv').exists() and not backup.exists():shutil.copy2(raw/'seed_probes.csv',backup)
    seeds.to_csv(raw/'seed_probes.csv',sep='\t',index=False,float_format='%.17g')
    seeds.to_csv(raw/'meinter_gse66695_DMS.tsv',sep='\t',index=False,float_format='%.17g')
    report={'article':ARTICLE,'author_code':CODE,'author_commit':COMMIT,
        'identity_status':'Reconstructed from original GEO data using published author code; not an author-supplied CpG table',
        'supplements':'File 1 contains figures/tables; files 2 and 3 are liver-cancer tables, not the GSE66695 CpG list',
        'delta_definition':'mean(tumor available values) - mean(normal available values), without imputation',
        'selection':'abs(deltaBeta)>0.3; independently verified Welch P.Value<0.01 and BH FDR<0.01',
        'statistical_caveat':'Author UseCase1 and groupGEO implement effect-size filtering only; p-values/FDR here are independently computed, not reported author statistics',
        'bh_family':'All original cg probes, including untestable probes assigned p=1',
        'bh_test_count':len(all_stats),'DMS_positive':counts[0],'DMS_negative':counts[1],'total_seed_cpgs':len(seeds),
        'present_in_complete_training_matrix':int(seeds.present_in_complete_training_matrix.sum()),
        'absent_from_complete_training_matrix':int((~seeds.present_in_complete_training_matrix).sum()),
        'max_seed_p_value':float(seeds['P.Value'].max()),'max_seed_FDR':float(seeds.FDR.max()),
        'source_sha256':digest(raw/'source/GSE66695_series_matrix.txt.gz'),
        'seed_sha256':digest(raw/'seed_probes.csv'),'previous_seed_backup':str(backup.relative_to(ROOT))}
    (raw/'meinter_seed_report.json').write_text(json.dumps(report,indent=2)+'\n')
    preparation=raw/'preparation_report.json'
    if preparation.exists():
        old=json.loads(preparation.read_text());old['seed_policy']='Replaced with reconstructed MeinteR GSE66695 DMS list; see meinter_seed_report.json'
        old['output_sha256']['seed_probes.csv']=report['seed_sha256']
        old['current_seed_count']=len(seeds)
        old['legacy_seed_fields_note']='original_seed_count, measured_original_seeds and measured_seeds_p_le_0_05 describe the superseded identity selection; current counts are in meinter_seed_report.json'
        preparation.write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return seeds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'raw/GSE66695')
    parser.add_argument('--annotation',type=Path,default=ROOT/'evaluation/raw/GPL13534_probe_annotation.tsv')
    args=parser.parse_args();build(args.raw_dir,args.annotation)

if __name__=='__main__':main()
