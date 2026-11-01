#!/usr/bin/env python3
"""Independently compare every selected output beta to the original GEO matrix."""
import csv,gzip,json,math,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=ROOT/'raw/GSE66695';meta={};count=0;checked_values=0;seen=set();examples=[]
    mapper=pd.read_csv(raw/'probe_genes.csv',sep='\t').set_index('probeID',verify_integrity=True)
    seeds=pd.read_csv(raw/'seed_probes.csv',sep='\t').set_index('probeID',verify_integrity=True)
    with gzip.open(raw/'source/GSE66695_series_matrix.txt.gz','rt') as source,gzip.open(raw/'GSE66695_matrix_processed.txt.gz','rt') as target:
        for line in source:
            row=next(csv.reader([line],delimiter='\t'))
            if not row:continue
            if row[0]=='!series_matrix_table_begin':break
            if row[0].startswith('!Sample_'):meta.setdefault(row[0],[]).append(row[1:])
        source_reader=csv.reader(source,delimiter='\t');header=next(source_reader)
        labels=meta['!Sample_source_name_ch1'][0]
        cancer=[i for i,label in enumerate(labels) if label=='Tumor'];normal=[i for i,label in enumerate(labels) if label=='Normal']
        assert len(cancer)==80 and len(normal)==40
        order=cancer+normal
        expected=['ID_REF']+[('BC_' if labels[i]=='Tumor' else 'NC_')+header[i+1] for i in order]
        target_reader=csv.reader(target,delimiter='\t');assert next(target_reader)==expected
        for row in source_reader:
            if row[0]=='!series_matrix_table_end':break
            if not row[0].startswith('cg'):continue
            strings=[row[i+1] for i in order]
            if any(value.strip().lower() in ('','na','nan','null','n/a') for value in strings):continue
            actual=next(target_reader)
            assert actual==[row[0],*strings],f'Source text differs at {row[0]}'
            assert row[0] not in seen;seen.add(row[0])
            nums=np.array(strings,dtype=float);assert np.isfinite(nums).all() and ((nums>=0)&(nums<=1)).all()
            if count%100000==0 or row[0] in seeds.index:
                delta=float(nums[:80].mean()-nums[80:].mean())
                p=float(ttest_ind(nums[:80],nums[80:],equal_var=False).pvalue)
                assert math.isclose(delta,mapper.loc[row[0],'deltaBeta'],abs_tol=1e-12)
                assert math.isclose(p,mapper.loc[row[0],'P.Value'],abs_tol=1e-12)
                if row[0] in seeds.index:
                    assert math.isclose(delta,seeds.loc[row[0],'deltaBeta'],abs_tol=1e-12)
                    assert math.isclose(p,seeds.loc[row[0],'P.Value'],abs_tol=1e-12)
                if count%100000==0:examples.append(row[0])
            count+=1;checked_values+=len(strings)
        assert next(target_reader,None) is None
        source.read();target.read()
    assert count==len(mapper)
    samples=pd.read_csv(raw/'sample_metadata.tsv',sep='\t')
    assert samples.sample_id.tolist()==expected[1:]
    assert samples.group.tolist()==['breast_cancer']*80+['normal_control']*40
    # Check the actual filters used by the next preprocessing stage without creating outputs.
    ann=pd.read_csv(raw/'GPL13534_platform.csv',dtype=str)
    xy=set(ann.loc[ann.CHR.isin(['X','Y']),'ID'])
    snps=set(pd.read_csv(raw/'snp_7998probes.vh20151030.txt',header=None)[0].astype(str))
    cross=set(pd.read_csv(raw/'cross_reactive_probes.csv',header=None)[0].astype(str))
    eligible=mapper.loc[(mapper['P.Value']<=.05)&mapper.UCSC_RefGene_Name.notna()]
    eligible=eligible.loc[~eligible.index.isin(xy|snps|cross)]
    retained_seeds=seeds.loc[seeds.index.isin(eligible.index)&(seeds['P.Value']<=.05)]
    assert len(retained_seeds)>=7,'Too few seed CpGs for existing partitioning'
    negative_mappings=sum(len(set(str(genes).split(';'))) for probe,genes in eligible.UCSC_RefGene_Name.items() if probe not in retained_seeds.index)
    assert negative_mappings>=10000,'Insufficient candidates for configured negative count'
    report={'passed':True,'selected_samples':120,'tumor_samples':80,'normal_samples':40,'complete_cpgs':count,
            'source_beta_strings_verified_exactly':checked_values,'independently_checked_example_cpgs':examples,
            'seed_statistics_checked':len(seeds.index.intersection(mapper.index)),
            'seed_cpgs_absent_from_complete_matrix':len(seeds.index.difference(mapper.index)),'retained_seed_cpgs_after_existing_pvalue_and_probe_filters':len(retained_seeds),
            'available_negative_gene_mappings_after_existing_filters':negative_mappings,
            'original_geo_sample_count':len(labels)}
    if (raw/'meinter_seed_report.json').exists():
        provenance=json.loads((raw/'meinter_seed_report.json').read_text())
        assert hashlib.sha256((raw/'seed_probes.csv').read_bytes()).hexdigest()==provenance['seed_sha256']
        assert len(seeds)==355 and (seeds.deltaBeta>.3).sum()==293 and (seeds.deltaBeta<-.3).sum()==62
        assert (seeds['P.Value']<.01).all() and (seeds.FDR<.01).all()
        report['meinter_seed_count_and_hash_verified']=True
    (raw/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
