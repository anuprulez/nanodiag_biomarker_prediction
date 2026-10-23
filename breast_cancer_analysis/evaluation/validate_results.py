#!/usr/bin/env python3
"""Validate every shared row and independently recompute three raw CpGs."""
import argparse
import csv
import json
import math
import statistics
from geo_common import BASE,SERIES,rows,digest,annotations
from evaluate_datasets import HEALTHY,CANCER,DELTA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=__import__('pathlib').Path,default=BASE/'raw')
    parser.add_argument('--results-dir',type=__import__('pathlib').Path,default=BASE/'results')
    args=parser.parse_args(); out=args.results_dir
    report=json.loads((out/'analysis_report.json').read_text())
    for dataset in report['datasets'].values():
        for source in dataset['sources']:
            if digest(args.raw_dir/source['file'])!=source['sha256']:raise ValueError('Source checksum changed')
    assert digest(args.raw_dir/'GPL13534_probe_annotation.tsv')==report['annotation_sha256']
    pairs=set();probes=set();checks={}
    ann=annotations(args.raw_dir)
    with (out/'cpg_gene_beta_comparison.tsv').open() as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            key=row['CpG_Gene']; probe=row['CpG_ID']
            assert key==probe+'_'+row['Gene'] and key not in pairs
            assert row['Gene'] in ann[probe]['genes']
            pairs.add(key);probes.add(probe)
            h,c=row[HEALTHY+'_mean_beta'],row[CANCER+'_mean_beta']
            if h and c:
                assert math.isclose(float(c)-float(h),float(row[DELTA]),abs_tol=2e-12)
                if len(checks)<3:checks.setdefault(probe,row)
            else:assert row[DELTA]==''
            for series,prefix in zip(SERIES,(HEALTHY,CANCER)):
                n=int(row[prefix+'_n']);mean=row[prefix+'_mean_beta'];sd=row[prefix+'_SD']
                assert 0<=n<=report['datasets'][series]['samples']
                assert (n==0)==(mean=='') and (n<2)==(sd=='')
                if mean:assert 0<=float(mean)<=1
                if sd:assert float(sd)>=0
    assert len(pairs)==report['cpg_gene_rows'] and len(probes)==report['shared_gene_annotated_cpgs']
    shared=set()
    with (out/'shared_cpg_statistics.tsv').open() as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            assert row['CpG_ID'] not in shared
            shared.add(row['CpG_ID'])
    assert len(shared)==report['shared_cpgs'] and probes<=shared
    for series,prefix in zip(SERIES,(HEALTHY,CANCER)):
        remaining=set(checks)
        for probe,values in rows(series,args.raw_dir):
            if probe not in remaining:continue
            nums=[float(x) for x in values if x.strip().lower() not in ('','na','nan','null','n/a')]
            row=checks[probe]
            assert len(nums)==int(row[prefix+'_n'])
            assert math.isclose(statistics.mean(nums),float(row[prefix+'_mean_beta']),abs_tol=1e-12)
            assert math.isclose(statistics.stdev(nums),float(row[prefix+'_SD']),abs_tol=1e-12)
            remaining.remove(probe)
            if not remaining:break
        assert not remaining
    result={'passed':True,'shared_cpgs_checked':len(shared),'cpg_gene_rows_checked':len(pairs),
            'independently_recomputed_cpgs_per_dataset':sorted(checks)}
    (out/'validation_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
