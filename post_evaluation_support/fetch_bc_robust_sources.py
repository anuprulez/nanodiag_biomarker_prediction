"""Coordinate-consistent BC cis mappings, published purity, and gene copy number."""
from pathlib import Path
import json,requests
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'breast_cancer_analysis/data_corr_0.33_knn_n_3000_n_neg_10000/accumulated_results/post_evaluation/methylation_expression'
OUT=BASE/'robust_followup';SRC=OUT/'source'
URLS={'refGene.hg19.txt.gz':'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz',
      'HGNC_complete_set.tsv':'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt',
      'Tumor.purity.rda':'https://raw.githubusercontent.com/BioinformaticsFMRP/TCGAbiolinks/master/data/Tumor.purity.rda'}

def query(kind,dataset,args,name):
    p=SRC/name
    if p.exists():return json.loads(p.read_text())
    funcs={'fields':'(fn [dataset] (map :name (query {:select [:field.name] :from [:dataset] :join [:field [:= :dataset.id :dataset_id]] :where [:= :dataset.name dataset]})))',
           'samples':'(fn [dataset limit] (map :value (query {:select [:value] :from [:dataset] :join [:field [:= :dataset.id :dataset_id] :code [:= :field.id :field_id]] :limit limit :where [:and [:= :dataset.name dataset] [:= :field.name "sampleID"]]})))',
           'fetch':'(fn [dataset samples probes] (fetch [{:table dataset :columns probes :samples samples}]))'}
    q='('+funcs[kind]+' '+json.dumps(dataset)+' '+' '.join('nil' if v is None else json.dumps(v) for v in args)+')'
    r=requests.post('https://tcga.xenahubs.net/data/',data=q,timeout=90);r.raise_for_status();result=r.json()
    p.write_text(json.dumps(result));p.with_suffix('.query.txt').write_text(q);return result


def main():
    for folder in ['source','data','plots']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    for filename,url in URLS.items():
        if not (SRC/filename).exists():
            r=requests.get(url,timeout=90);r.raise_for_status();(SRC/filename).write_bytes(r.content)
    # Convert the package's published character-decimal table using the installed R.
    if not (SRC/'TCGA_purity.tsv').exists():
        import subprocess
        executable=ROOT.parents[2]/'.analysis_envs/bc-limma/bin/Rscript'
        code='load('+json.dumps(str(SRC/'Tumor.purity.rda'))+'); write.table(Tumor.purity,'+json.dumps(str(SRC/'TCGA_purity.tsv'))+',sep="\\t",quote=FALSE,row.names=FALSE)'
        subprocess.run([str(executable),'-e',code],check=True)
    hgnc=pd.read_csv(SRC/'HGNC_complete_set.tsv',sep='\t',dtype=str).fillna('')
    hgnc=hgnc.loc[hgnc.status.eq('Approved')]
    prev={};symbol_records={}
    for row in hgnc.itertuples():
        symbol_records[row.symbol]=row
        for old in row.prev_symbol.split('|'):
            if old:prev.setdefault(old,set()).add(row.symbol)
    def canonical(symbol):
        if symbol in symbol_records:return symbol
        choices=prev.get(symbol,set());return next(iter(choices)) if len(choices)==1 else symbol
    fields=set(json.loads((BASE/'source/expression_fields.json').read_text()))-{'sampleID'}
    approved_to_fields={}
    for gene in fields:approved_to_fields.setdefault(canonical(gene),[]).append(gene)
    def expression_gene(gene):
        symbol=canonical(gene);choices=approved_to_fields.get(symbol,[])
        if symbol in choices:return symbol
        return choices[0] if len(choices)==1 else None
    ref=pd.read_csv(SRC/'refGene.hg19.txt.gz',sep='\t',header=None,dtype={2:str,3:str,12:str})
    ref=ref.loc[ref[2].str.fullmatch('chr([0-9]+|X|Y)')].copy()
    ref['canonical_gene']=ref[12].map(canonical)
    ref['expression_gene']=ref['canonical_gene'].map(expression_gene)
    ref['tss0']=np.where(ref[3].eq('+'),ref[4],ref[5]-1)
    ann=pd.read_csv(BASE.parent/'data/candidate_genomic_annotation.tsv',sep='\t',dtype=str).fillna('')
    mapping=[];legacy=[]
    for row in ann.itertuples():
        pos=int(row.MAPINFO)-1;chrom='chr'+row.CHR;onchrom=ref.loc[ref[2].eq(chrom)].copy()
        onchrom['signed_distance']=((pos-onchrom.tss0)*np.where(onchrom[3].eq('+'),1,-1)).astype(int)
        near=onchrom.loc[onchrom.signed_distance.abs().le(250000)]
        for gene,group in near.groupby('canonical_gene'):
            nearest=group.loc[group.signed_distance.abs().idxmin()]
            mapping.append(dict(CpG_ID=row.CpG_ID,canonical_gene=gene,expression_gene=nearest.expression_gene,
                                chromosome=chrom,CpG_position_hg19_1based=pos+1,TSS_hg19_1based=int(nearest.tss0)+1,
                                strand=nearest[3],signed_distance_to_TSS=int(nearest.signed_distance),
                                gene_body_overlap=bool(((onchrom.canonical_gene==gene)&onchrom[4].le(pos)&onchrom[5].gt(pos)).any()),
                                transcript_count_with_TSS_in_window=len(group),mapping='TSS within ±250 kb',
                                expression_mapping_status='mapped' if pd.notna(nearest.expression_gene) else 'missing or ambiguous expression symbol'))
        tokens=row.UCSC_RefGene_Name.split(';');regions=row.UCSC_RefGene_Group.split(';')
        assert len(tokens)==len(regions)
        for old in sorted(set(tokens)-{''}):
            cg=canonical(old);matches=onchrom.loc[onchrom.canonical_gene.eq(cg)]
            closest=matches.loc[matches.signed_distance.abs().idxmin()] if len(matches) else None
            legacy.append(dict(CpG_ID=row.CpG_ID,annotated_gene=old,canonical_gene=cg,
                               expression_gene=expression_gene(cg),
                               legacy_gene_regions=';'.join(sorted({r for g,r in zip(tokens,regions) if g==old})),
                               legacy_promoter=any(g==old and r in ['TSS200','TSS1500'] for g,r in zip(tokens,regions)),
                               nearest_TSS_distance=int(closest.signed_distance) if closest is not None else np.nan,
                               within_250kb_of_any_TSS=bool(len(matches) and matches.signed_distance.abs().le(250000).any()),
                               hg19_gene_body_overlap=bool(len(matches) and (matches[4].le(pos)&matches[5].gt(pos)).any()),
                               hg19_transcript_found=bool(len(matches)),mapping='Legacy annotated gene'))
    cis=pd.DataFrame(mapping);leg=pd.DataFrame(legacy)
    assert not cis.duplicated(['CpG_ID','canonical_gene']).any()
    cis.to_csv(OUT/'data/cis_250kb_gene_mapping.tsv',sep='\t',index=False)
    leg.to_csv(OUT/'data/legacy_gene_mapping_audit.tsv',sep='\t',index=False)
    meta=pd.read_csv(BASE/'data/sample_metadata.tsv',sep='\t',index_col=0)
    purity=pd.read_csv(SRC/'TCGA_purity.tsv',sep='\t',dtype=str)
    purity=purity.loc[purity['Cancer.type'].eq('BRCA')].copy()
    for col in ['CPE','ABSOLUTE','ESTIMATE','LUMP','IHC']:
        purity[col]=pd.to_numeric(purity[col].str.replace(',','.',regex=False),errors='coerce')
        assert purity[col].dropna().between(0,1).all()
    purity['sample15']=purity['Sample.ID'].str[:15]
    for col in ['CPE','ABSOLUTE','ESTIMATE','LUMP','IHC']:meta[col]=np.nan
    meta['purity_match']='not applicable or unavailable'
    for sample,row in meta.iterrows():
        if row.sample_type!='Primary tumor':continue
        match=purity.loc[purity['Sample.ID'].eq(row.methylation_sample_id)]
        rule='exact methylation sample/vial'
        if len(match)==0:match=purity.loc[purity.sample15.eq(sample)];rule='unique same sample type'
        if len(match)==1:
            for col in ['CPE','ABSOLUTE','ESTIMATE','LUMP','IHC']:meta.loc[sample,col]=match.iloc[0][col]
            meta.loc[sample,'purity_match']=rule
    meta.to_csv(OUT/'data/sample_metadata_with_purity.tsv',sep='\t')
    genes=sorted(set(cis.expression_gene.dropna())|set(leg.expression_gene.dropna()))
    cn_dataset='TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_data_by_genes'
    cnfields=set(query('fields',cn_dataset,[],'copy_number_fields.json'))
    cnsamples=set(query('samples',cn_dataset,[None],'copy_number_samples.json'))
    probes={}
    for gene in genes:
        cg=canonical(gene)
        if gene in cnfields:probes[gene]=gene
        elif cg in cnfields:probes[gene]=cg
        else:
            matches=[g for g in cnfields if canonical(g)==cg]
            if len(matches)==1:probes[gene]=matches[0]
    samples=[s for s in meta.index if s in cnsamples]
    cnprobes=sorted(set(probes.values()));values=query('fetch',cn_dataset,[samples,cnprobes],'gene_copy_number_values.json')
    cn=pd.DataFrame(values,index=cnprobes,columns=samples).apply(pd.to_numeric,errors='coerce')
    aligned=pd.DataFrame(index=genes,columns=meta.index,dtype=float)
    for gene,probe in probes.items():aligned.loc[gene,samples]=cn.loc[probe,samples].values
    aligned.to_csv(OUT/'data/gene_copy_number.tsv.gz',sep='\t')
    pd.DataFrame([dict(expression_gene=g,copy_number_probe=probes.get(g,'')) for g in genes]).to_csv(OUT/'data/copy_number_gene_mapping.tsv',sep='\t',index=False)
    (SRC/'sources.json').write_text(json.dumps(URLS|{'purity_origin':'https://doi.org/10.1038/ncomms9971','CN_hub':'https://tcga.xenahubs.net','CN_dataset':cn_dataset,'coordinate_system':'hg19/GRCh37; GPL13534 MAPINFO is 1-based, refGene is 0-based half-open; TSS = txStart for + and txEnd-1 for -','window_bp':250000,'gene_mapping':'HGNC approved symbols and unambiguous previous-symbol mappings; no alias_symbol substitutions'},indent=2))
    print('Cis pairs',len(cis),'with expression',cis.expression_gene.notna().sum(),'genes',len(genes),'CN samples',len(samples))
    print('Matched tumor purity',meta.loc[meta.sample_type.eq('Primary tumor'),['CPE','ABSOLUTE']].notna().sum().to_dict())

if __name__=='__main__':main()
