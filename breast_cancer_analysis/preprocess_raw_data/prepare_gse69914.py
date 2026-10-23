#!/usr/bin/env python3
"""Prepare status-2 BC and status-0 healthy breast samples for existing preprocessing."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

ROOT=Path(__file__).resolve().parents[1]
URL='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE69nnn/GSE69914/matrix/GSE69914_series_matrix.txt.gz'
ANNOTATION_URL='https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/soft/GPL13534_family.soft.gz'


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def read_header(source):
    metadata={}
    for line in source:
        row=next(csv.reader([line],delimiter='\t'))
        if not row:continue
        if row[0]=='!series_matrix_table_begin':
            header=next(csv.reader([next(source)],delimiter='\t'))
            if header[0]!='ID_REF' or metadata.get('!Sample_geo_accession')!=[header[1:]]:
                raise ValueError('Matrix samples differ from GEO metadata')
            if len(set(header[1:]))!=len(header)-1:raise ValueError('Duplicate samples')
            return header[1:],metadata
        if row[0].startswith('!Sample_'):metadata.setdefault(row[0],[]).append(row[1:])
    raise ValueError('Series Matrix table absent')


def select_samples(samples,metadata,expected=(305,50)):
    if metadata.get('!Sample_platform_id')!=[['GPL16304']*len(samples)]:raise ValueError('Expected GPL16304')
    rows=metadata.get('!Sample_characteristics_ch1',[])
    statuses=[row for row in rows if row[0].startswith('status(')]
    if len(statuses)!=1 or len(statuses[0])!=len(samples):raise ValueError('Missing or ambiguous sample status')
    codes=[int(value.rsplit(':',1)[1].strip()) for value in statuses[0]]
    if not set(codes)<={0,1,2,3,4}:raise ValueError('Unknown sample status')
    bc=[i for i,status in enumerate(codes) if status==2]
    normal=[i for i,status in enumerate(codes) if status==0]
    if (len(bc),len(normal))!=expected:raise ValueError(f'Unexpected BC/healthy sample counts: {len(bc)}/{len(normal)}')
    indices=bc+normal
    aliases=[('BC_' if codes[i]==2 else 'NC_')+samples[i] for i in indices]
    return indices,aliases,codes


def differential(values,n_bc):
    cancer=values[:,:n_bc];normal=values[:,n_bc:]
    delta=cancer.mean(axis=1)-normal.mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        p=np.asarray(ttest_ind(cancer,normal,axis=1,equal_var=False).pvalue)
    # Welch's statistic is undefined when both groups are identical constants.
    constant=(cancer.var(axis=1)==0)&(normal.var(axis=1)==0)
    p=np.where(constant,np.where(delta==0,1.,0.),p)
    if not np.isfinite(p).all():raise ValueError('Undefined differential statistics')
    return delta,p


def prepare(source_path,out,annotation,seed_ids_path,expected=(305,50)):
    out.mkdir(parents=True,exist_ok=True)
    completed=out/'preparation_report.json';completed.unlink(missing_ok=True)
    ann=pd.read_csv(annotation,sep='\t',dtype=str,keep_default_na=False).set_index('ID',verify_integrity=True)
    required={'UCSC_RefGene_Name','CHR','MAPINFO'}
    if not required<=set(ann.columns):raise ValueError('Missing standard 450K annotation columns')
    seed_ids=pd.read_csv(seed_ids_path,sep='\t',dtype=str).probeID.tolist()
    if len(seed_ids)!=len(set(seed_ids)):raise ValueError('Duplicate original seed IDs')
    with gzip.open(source_path,'rt') as source:
        samples,meta=read_header(source)
        indices,aliases,codes=select_samples(samples,meta,expected)
        metadata_rows=[]
        for i,sample in enumerate(samples):
            selected=codes[i] in (0,2)
            metadata_rows.append({'sample_id':('BC_' if codes[i]==2 else 'NC_')+sample if selected else '',
                'geo_accession':sample,'series':'GSE69914','status':codes[i],
                'group':{0:'healthy_control',1:'excluded_normal_adjacent',2:'breast_cancer',3:'excluded_normal_BRCA1',4:'excluded_cancer_BRCA1'}[codes[i]],
                'label':1 if codes[i]==2 else 0 if codes[i]==0 else '',
                'selected':selected,'sample_title':meta['!Sample_title'][0][i],
                'source_name':meta['!Sample_source_name_ch1'][0][i],
                'platform':'GPL16304','tissue':'breast',
                'characteristics':' | '.join(row[i] for row in meta['!Sample_characteristics_ch1'])})
        pd.DataFrame([metadata_rows[i] for i in indices]).to_csv(out/'sample_metadata.tsv',sep='\t',index=False)
        pd.DataFrame(metadata_rows).to_csv(out/'all_sample_metadata.tsv',sep='\t',index=False)
        (out/'geo_sample_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        matrix=out/'GSE69914_matrix_processed.txt.gz'
        temporary=matrix.with_suffix('.gz.part')
        seen=set();seed_stats={};counts={'source_rows':0,'non_cpg_rows_excluded':0,'incomplete_cpg_rows_excluded':0,'retained_cpgs':0}
        with gzip.open(temporary,'wt',compresslevel=1) as target,(out/'probe_genes.csv').open('w') as mapper,(out/'excluded_cpgs.tsv').open('w') as exclusions:
            writer=csv.writer(target,delimiter='\t',lineterminator='\n');writer.writerow(['ID_REF',*aliases])
            mapwriter=csv.writer(mapper,delimiter='\t',lineterminator='\n');mapwriter.writerow(['probeID','deltaBeta','P.Value','UCSC_RefGene_Name'])
            excl=csv.writer(exclusions,delimiter='\t',lineterminator='\n');excl.writerow(['CpG_ID','reason'])
            for chunk in pd.read_csv(source,sep='\t',header=None,names=['ID_REF',*samples],dtype=str,
                                     keep_default_na=False,comment='!',chunksize=1000):
                counts['source_rows']+=len(chunk)
                probes=chunk.ID_REF.tolist()
                if len(set(probes))!=len(probes) or set(probes)&seen:raise ValueError('Duplicate source probes')
                seen.update(probes)
                cpg=chunk.ID_REF.str.fullmatch(r'cg\d+')
                counts['non_cpg_rows_excluded']+=int((~cpg).sum())
                chunk=chunk.loc[cpg]
                strings=chunk.iloc[:,[i+1 for i in indices]].to_numpy()
                missing=np.isin(np.char.lower(strings.astype(str)),['','na','nan','null','n/a'])
                numbers=np.where(missing,'nan',strings).astype(float)
                finite=numbers[~np.isnan(numbers)]
                if not np.isfinite(finite).all() or np.any((finite<0)|(finite>1)):raise ValueError('Invalid published beta value')
                complete=~np.isnan(numbers).any(axis=1)
                for probe in chunk.loc[~complete,'ID_REF']:excl.writerow([probe,'missing_beta_in_selected_samples'])
                counts['incomplete_cpg_rows_excluded']+=int((~complete).sum())
                retained=chunk.loc[complete,'ID_REF'].tolist();strings=strings[complete];numbers=numbers[complete]
                if not len(retained):continue
                delta,pvalues=differential(numbers,expected[0])
                annotated=ann.reindex(retained).fillna('')
                for j,probe in enumerate(retained):
                    writer.writerow([probe,*strings[j]])
                    genes=';'.join(dict.fromkeys(g.strip() for g in annotated.iloc[j].UCSC_RefGene_Name.split(';') if g.strip()))
                    mapwriter.writerow([probe,format(delta[j],'.17g'),format(pvalues[j],'.17g'),genes])
                    if probe in seed_ids:
                        seed_stats[probe]={'probeID':probe,'deltaBeta':float(delta[j]),'P.Value':float(pvalues[j]),
                                          'CHR':annotated.iloc[j].CHR,'UCSC_RefGene_Name':genes}
                counts['retained_cpgs']+=len(retained)
                if counts['source_rows']%100000==0:print(f'{counts["source_rows"]:,} source rows read; {counts["retained_cpgs"]:,} complete CpGs retained',flush=True)
        source.read()
        temporary.replace(matrix)
    if not counts['retained_cpgs']:raise ValueError('No complete CpGs')
    seeds=pd.DataFrame([seed_stats[s] for s in seed_ids if s in seed_stats])
    seeds.to_csv(out/'seed_probes.csv',sep='\t',index=False)
    seed_qc=pd.DataFrame({'probeID':seed_ids});seed_qc['present_in_selected_complete_matrix']=seed_qc.probeID.isin(seed_stats)
    seed_qc.to_csv(out/'original_seed_coverage.tsv',sep='\t',index=False)
    platform=ann.reset_index()[['ID','CHR','MAPINFO']]
    platform.to_csv(out/'GPL13534_platform.csv',index=False)
    report=dict(series='GSE69914',source_url=URL,platform_reported_by_GEO='GPL16304',
                selected_samples=sum(expected),breast_cancer_samples=expected[0],healthy_normal_samples=expected[1],
                excluded_status_counts={str(code):codes.count(code) for code in (1,3,4)},
                sample_order='All status-2 cancer samples, then all status-0 healthy samples; original order within each group',
                beta_policy='Published beta text preserved; cg probes only; complete cases across selected samples; no imputation or renormalization',
                delta_definition='mean(305 BC breast tumors) - mean(50 healthy normal breast samples)',
                p_value_method='Two-sided Welch t-test on selected samples; unadjusted P.Value for compatibility with existing cutoff_pval',
                seed_policy='Retain existing seed CpG identities when measured; recalculate deltaBeta and P.Value on GSE69914; no external evaluation data used',
                original_seed_count=len(seed_ids),measured_original_seeds=len(seeds),
                measured_seeds_p_le_0_05=int((seeds['P.Value']<=.05).sum()),
                annotation='GPL13534 manufacturer RefGene hg19 annotation joined by exact CpG ID, same as external evaluation; not GPL16304 closest-TSS annotation',
                annotation_source_url=ANNOTATION_URL,annotation_sha256=digest(annotation),original_seed_id_file_sha256=digest(seed_ids_path),
                source_sha256=digest(source_path),counts=counts)
    report['output_sha256']={p.name:digest(p) for p in [matrix,out/'probe_genes.csv',out/'seed_probes.csv',out/'sample_metadata.tsv',out/'GPL13534_platform.csv']}
    completed.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'raw/GSE69914')
    parser.add_argument('--annotation',type=Path,default=ROOT/'evaluation/raw/GPL13534_probe_annotation.tsv')
    parser.add_argument('--seed-ids',type=Path,default=ROOT/'raw/seed_probes.csv')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'preparation_report.json').exists() and not args.force:
        raise ValueError('Prepared inputs already exist; use --force to rebuild')
    source=args.output_dir/'source/GSE69914_series_matrix.txt.gz';source.parent.mkdir(exist_ok=True)
    if not source.exists():
        part=source.with_suffix('.gz.part')
        subprocess.run(['curl','-fL','--retry','3','--continue-at','-',URL,'-o',str(part)],check=True)
        source=part
    prepare(source,args.output_dir,args.annotation,args.seed_ids)
    if source.suffix=='.part':source.replace(args.output_dir/'source/GSE69914_series_matrix.txt.gz')
    for name in ['snp_7998probes.vh20151030.txt','cross_reactive_probes.csv']:
        shutil.copy2(ROOT/'raw'/name,args.output_dir/name)

if __name__=='__main__':main()
