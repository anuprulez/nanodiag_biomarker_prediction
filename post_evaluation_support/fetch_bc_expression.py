"""Fetch matched public TCGA-BRCA inputs; callable and resumable from any dataset folder."""
from pathlib import Path
import argparse,json,requests,pandas as pd
from .expression_resources import cached_query,download


def main(base,offline=False):
    base=Path(base);out=base;src=out/'source'
    for folder in [src,out/'data',out/'plots']:folder.mkdir(parents=True,exist_ok=True)
    host='https://tcga.xenahubs.net'
    functions={
    'samples':'(fn [dataset limit] (map :value (query {:select [:value] :from [:dataset] :join [:field [:= :dataset.id :dataset_id] :code [:= :field.id :field_id]] :limit limit :where [:and [:= :dataset.name dataset] [:= :field.name "sampleID"]]})))',
    'fields':'(fn [dataset] (map :name (query {:select [:field.name] :from [:dataset] :join [:field [:= :dataset.id :dataset_id]] :where [:= :dataset.name dataset]})))',
    'fetch':'(fn [dataset samples probes] (fetch [{:table dataset :columns probes :samples samples}]))'}
    def query(kind,args,filename):
     p=src/filename
     def edn(x):return 'nil' if x is None else json.dumps(x)
     q='('+functions[kind]+' '+' '.join(edn(x) for x in args)+')'
     return cached_query(host,q,p,offline)

    ann=pd.read_csv(base.parent/'data/candidate_genomic_annotation.tsv',sep='\t',dtype=str)
    query('fields',['TCGA.BRCA.sampleMap/HiSeqV2'],'expression_fields.json')
    legacy=query('samples',['TCGA.BRCA.sampleMap/HiSeqV2',None],'expression_samples.json')
    host='https://gdc.xenahubs.net';md='TCGA-BRCA.methylation450.tsv'
    ms=query('samples',[md,None],'gdc_methylation_samples.json')
    from collections import defaultdict
    mapping=defaultdict(list)
    for s in ms:
     if len(s)>=16 and s[13:15] in ['01','11']:mapping[s[:15]].append(s)
    shared=sorted(s for s in set(legacy)&set(mapping) if len(mapping[s])==1)
    values=query('fetch',[md,[mapping[s][0] for s in shared],ann.CpG_ID.tolist()],'gdc_methylation_values.json')
    m=pd.DataFrame(values,index=ann.CpG_ID,columns=shared).apply(pd.to_numeric,errors='coerce');m.to_csv(out/'data/methylation_matched.tsv.gz',sep='\t')
    host='https://tcga.xenahubs.net';ed='TCGA.BRCA.sampleMap/HiSeqV2'
    genes=sorted(set(g.strip() for x in ann.UCSC_RefGene_Name.fillna('') for g in x.split(';') if g.strip()))
    values=query('fetch',[ed,shared,genes],'expression_gdc_matched_values.json')
    e=pd.DataFrame(values,index=genes,columns=shared).apply(pd.to_numeric,errors='coerce');e.to_csv(out/'data/expression_matched.tsv.gz',sep='\t')
    fields=['age_at_initial_pathologic_diagnosis','gender','PAM50Call_RNAseq','breast_carcinoma_estrogen_receptor_status']
    values=query('fetch',['TCGA.BRCA.sampleMap/BRCA_clinicalMatrix',shared,fields],'clinical_values.json')
    meta=pd.DataFrame(values,index=fields,columns=shared).T
    codes_path=src/'clinical_codes.json'
    q='(query {:select [:field.name :code.ordering :code.value] :from [:dataset] :join [:field [:= :dataset.id :dataset_id] :code [:= :field.id :field_id]] :where [:and [:= :dataset.name "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"] [:in :field.name '+json.dumps(fields)+']]})'
    codes_path.parent.mkdir(parents=True,exist_ok=True)
    cached_query(host,q,codes_path,offline)
    codes=json.loads(codes_path.read_text())
    for field in fields:
     decode={str(x['ordering']):x['value'] for x in codes if x['name']==field}
     if decode:meta[field]=meta[field].map(lambda v:decode.get(str(v),'')).replace('',None)
    meta['age']=pd.to_numeric(meta.age_at_initial_pathologic_diagnosis,errors='coerce')
    meta['patient_id']=[s[:12] for s in shared];meta['sample_type']=['Primary tumor' if s.endswith('-01') else 'Solid tissue normal' for s in shared]
    meta['methylation_sample_id']=[mapping[s][0] for s in shared];meta.index.name='sample_id';meta.to_csv(out/'data/sample_metadata.tsv',sep='\t')
    download('https://gdc.xenahubs.net/download/'+md+'.json',src/'gdc_methylation_metadata.json',offline)
    (src/'source_datasets.json').write_text(json.dumps({'methylation_hub':'https://gdc.xenahubs.net','methylation':md,'expression_hub':host,'expression':ed,'clinical':'TCGA.BRCA.sampleMap/BRCA_clinicalMatrix','matching':'15-character TCGA sample-type ID; require exactly one GDC vial per ID; one sample per patient within tissue type; distinct methylation and RNA aliquots of same biological sample, not same molecules','excluded_ambiguous_vials':{s:v for s,v in mapping.items() if len(v)>1 and s in legacy},'legacy_methylation_audit':'Initial legacy-hub audit found 13/33 CpGs entirely missing. GDC hub chosen for better coverage before correlation analysis; initial audit retained.','original_GEO':'GSE66695 methylation only; TCGA is separate'},indent=2))
    print('samples',meta.sample_type.value_counts().to_dict(),'subtypes',meta.loc[meta.sample_type.eq('Primary tumor'),'PAM50Call_RNAseq'].value_counts(dropna=False).to_dict())
    print('missing CpGs',m.index[m.isna().all(axis=1)].tolist(),'missing genes',e.index[e.isna().all(axis=1)].tolist())
    print('age',meta.age.notna().sum(),'ambiguous',sum(len(v)>1 for v in mapping.values()))

    meta=pd.read_csv(out/'data/sample_metadata.tsv',sep='\t',index_col=0);samples=meta.index.tolist()
    download('https://rest.genenames.org/search/prev_symbol/C7orf20',src/'HGNC_C7orf20_GET4.json',offline)
    lookup=json.loads((src/'HGNC_C7orf20_GET4.json').read_text());assert lookup['response']['docs'][0]['symbol']=='GET4'
    values=query('fetch',['TCGA.BRCA.sampleMap/HiSeqV2',samples,['GET4']],'GET4_expression_values.json')
    e=pd.read_csv(out/'data/expression_matched.tsv.gz',sep='\t',index_col=0)
    e.loc['GET4']=pd.to_numeric(pd.Series(values[0],index=samples),errors='coerce');e.to_csv(out/'data/expression_matched.tsv.gz',sep='\t')
    print('GET4',e.loc['GET4'].notna().sum())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();main(args.base,args.offline)
