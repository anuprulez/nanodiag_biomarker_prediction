#!/usr/bin/env python3
"""Map each cohort's CpGs to genes and compare exact shared sites; no model required."""
import argparse
import csv
import json
from pathlib import Path
import re
from geo_common import BASE, SERIES, SOURCES, PLATFORM_URL, digest, metadata, rows, statistics, annotations

HEALTHY = 'Healthy_GSE101961'
CANCER = 'BC_GSE226569'
DELTA = 'Delta_beta_BC_minus_healthy'
STAT_FIELDS = [f'{prefix}_{suffix}' for prefix in (HEALTHY,CANCER) for suffix in ('mean_beta','SD','n')]
FIELDS = ['CpG_ID','Gene','CpG_Gene',*STAT_FIELDS,DELTA,'CHR','MAPINFO_hg19']
LIMITATIONS = ('Descriptive cross-study breast tissue comparison. Disease and study are confounded; '
               'tumor samples are FFPE. No adjustment for batch, age, race, tumor purity or cell composition. '
               'Probe gene associations are legacy platform annotations, not confirmed regulatory targets.')


def formatted(values):
    return ['' if x is None else format(x,'.12g') if isinstance(x,float) else x for x in values]


def summarize(series, raw, output):
    samples, meta = metadata(series,raw)
    (output/f'{series}_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    with (output/f'{series}_samples.tsv').open('w') as out:
        writer=csv.writer(out,delimiter='\t',lineterminator='\n')
        writer.writerow(['sample_id','series','group','matrix_sample_title','source','characteristics'])
        for i,sample in enumerate(samples):
            writer.writerow([sample,series,'healthy_control' if series==SERIES[0] else 'breast_cancer',
                             meta['!Sample_title'][0][i],meta['!Sample_source_name_ch1'][0][i],
                             ' | '.join(row[i] for row in meta.get('!Sample_characteristics_ch1',[]))])
    result={}
    report={'samples':len(samples),'rows':0,'non_cpg_rows':0,'missing_values':0,'all_missing_cpgs':0}
    for probe,values in rows(series,raw):
        report['rows']+=1
        if not re.fullmatch(r'cg\d+',probe):
            report['non_cpg_rows']+=1
            continue
        mean,sd,n=statistics(values)
        result[probe]=(mean,sd,n)
        report['missing_values']+=len(values)-n
        report['all_missing_cpgs']+=n==0
        if report['rows']%100000==0:print(f'{series}: {report["rows"]:,} rows processed',flush=True)
    if not result:raise ValueError(f'{series}: empty beta matrix')
    report['cpgs']=len(result)
    report['sources']=[dict(file=name,url=url,sha256=digest(raw/name),bytes=(raw/name).stat().st_size)
                       for name,url in SOURCES.items() if name.startswith(series)]
    return result,report


def write_results(healthy,cancer,ann,output):
    # Independent mapping tables cover every measured CpG in each dataset,
    # including unmatched and unannotated probes; no mean is duplicated in a gene aggregate.
    for series,stats in zip(SERIES,(healthy,cancer)):
        with (output/f'{series}_cpg_gene_statistics.tsv').open('w') as out:
            writer=csv.writer(out,delimiter='\t',lineterminator='\n')
            writer.writerow(['CpG_ID','Gene','CpG_Gene','mean_beta','SD','n','CHR','MAPINFO_hg19','annotation_status'])
            for probe in sorted(stats):
                item=ann.get(probe,{})
                for gene in item.get('genes',[]) or ['']:
                    writer.writerow([probe,gene,probe+'_'+gene if gene else '',*formatted(stats[probe]),
                                     item.get('CHR',''),item.get('MAPINFO',''),'gene_associated' if gene else 'unannotated'])
    shared=sorted(healthy.keys() & cancer.keys())
    if not shared:raise ValueError('No common CpGs')
    count=unannotated=0
    with (output/'cpg_gene_beta_comparison.tsv').open('w') as out,(output/'shared_cpg_statistics.tsv').open('w') as all_out:
        writer=csv.writer(out,delimiter='\t',lineterminator='\n');writer.writerow(FIELDS)
        all_writer=csv.writer(all_out,delimiter='\t',lineterminator='\n')
        all_writer.writerow(['CpG_ID',*STAT_FIELDS,DELTA,'Genes','annotation_status'])
        for probe in shared:
            h,c=healthy[probe],cancer[probe]
            delta=c[0]-h[0] if h[0] is not None and c[0] is not None else None
            values=formatted([*h,*c,delta])
            item=ann.get(probe,{})
            genes=item.get('genes',[])
            unannotated+=not genes
            all_writer.writerow([probe,*values,';'.join(genes),'gene_associated' if genes else 'unannotated'])
            for gene in genes:
                writer.writerow([probe,gene,probe+'_'+gene,*values,item.get('CHR',''),item.get('MAPINFO','')])
                count+=1
    return {'shared_cpgs':len(shared),'shared_gene_annotated_cpgs':len(shared)-unannotated,
            'shared_unannotated_cpgs':unannotated,'cpg_gene_rows':count,
            'healthy_only_cpgs':len(healthy.keys()-cancer.keys()),'bc_only_cpgs':len(cancer.keys()-healthy.keys())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=BASE/'raw')
    parser.add_argument('--output-dir',type=Path,default=BASE/'results')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    # Remove completion marker before starting: failed reruns cannot appear complete.
    (args.output_dir/'analysis_report.json').unlink(missing_ok=True)
    healthy,hr=summarize(SERIES[0],args.raw_dir,args.output_dir)
    cancer,cr=summarize(SERIES[1],args.raw_dir,args.output_dir)
    ann=annotations(args.raw_dir)
    report=write_results(healthy,cancer,ann,args.output_dir)
    report.update(datasets={SERIES[0]:hr,SERIES[1]:cr},delta='mean(GSE226569) - mean(GSE101961)',
                  SD='sample standard deviation, denominator n-1',
                  missing_policy='omit missing per CpG; report valid n; no imputation; undefined values blank',
                  tumor_matrix='supplementary processed beta columns; DetectionPvalue columns excluded without p-value filtering',
                  matching='exact shared CpG IDs; common GPL13534 gene map expands unique CpG-gene pairs',
                  annotation_url=PLATFORM_URL,annotation_sha256=digest(args.raw_dir/'GPL13534_probe_annotation.tsv'),
                  limitations=LIMITATIONS)
    (args.output_dir/'analysis_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
