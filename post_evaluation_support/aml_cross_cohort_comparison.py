"""AML fixed-panel cross-cohort effect comparison with audited GEO expression data."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,gzip,hashlib,json,subprocess,textwrap
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr,binomtest
from .publication import save_publication
from .expression_filter import expression_selected
from .cpg_scatter_style import scatter_context,unobstructed_labels


def geo_metadata(path):
    fields=defaultdict(list)
    with gzip.open(path,'rt') as handle:
        for line in handle:
            if line.startswith('!series_matrix_table_begin'):break
            if line.startswith('!Sample_'):
                row=next(csv.reader([line],delimiter='\t'))
                fields[row[0]].append(row[1:])
    ids=fields['!Sample_geo_accession'][0]
    records=[]
    for i,sample in enumerate(ids):
        d={'sample_id':sample}
        for key,rows in fields.items():
            if key=='!Sample_characteristics_ch1':
                for row in rows:
                    if ':' in row[i]:
                        k,v=row[i].split(':',1);d[k.strip().lower()]=v.strip()
            else:d[key.removeprefix('!Sample_')]=rows[0][i]
        records.append(d)
    return pd.DataFrame(records)


def symbol_mapper(hgnc_path):
    h=pd.read_csv(hgnc_path,sep='\t',dtype=str).fillna('')
    h=h.loc[h.status.eq('Approved')];approved=set(h.symbol);previous=defaultdict(set)
    for row in h.itertuples():
        for old in row.prev_symbol.split('|'):
            if old:previous[old].add(row.symbol)
    def canonical(symbol):
        if symbol in approved:return symbol
        if len(previous[symbol])==1:return next(iter(previous[symbol]))
        return symbol
    return canonical


def prepare_expression(out,canonical):
    src=out/'source';data=out/'data'
    meta=geo_metadata(src/'GSE15061_series_matrix.txt.gz')
    split=meta.title.str.split(',')
    meta['biological_sample_id']=split.str[0].str.strip()
    meta['normalization_variant']=split.str[-1].str.strip()
    meta['group']=meta['disease state'].replace({'none-of-the-targets':'Non-leukemia'})
    meta['center']=meta.source_name_ch1
    meta['selected']=meta.normalization_variant.eq('DS')&meta.group.isin(['AML','Non-leukemia'])
    meta['exclusion_reason']=np.where(meta.normalization_variant.ne('DS'),'Duplicate DQN3 representation; bounded rank signal',np.where(meta.group.eq('MDS'),'MDS excluded',''))
    meta.to_csv(data/'GSE15061_all_870_records_audit.tsv',sep='\t',index=False)
    selected=meta.loc[meta.selected].copy()
    assert len(selected)==271 and selected.biological_sample_id.is_unique
    assert selected.group.value_counts().to_dict()=={'AML':202,'Non-leukemia':69}
    selected.to_csv(data/'GSE15061_sample_metadata.tsv',sep='\t',index=False)
    pd.crosstab(selected.center,selected.group).to_csv(data/'group_by_center_counts.tsv',sep='\t')
    annotation=pd.read_csv(src/'GPL570_probe_annotation.tsv',sep='\t',dtype=str).fillna('')
    mapping=[]
    for row in annotation.to_dict('records'):
        symbols={canonical(s.strip()) for s in row['Gene Symbol'].split('///') if s.strip() not in {'','---'}}
        if len(symbols)==1 and not row['ID'].startswith('AFFX-'):
            mapping.append({'probe_id':row['ID'],'gene':next(iter(symbols))})
    mapping=pd.DataFrame(mapping);assert mapping.probe_id.is_unique
    matrix_path=data/'GSE15061_DS_probe_expression.tsv.gz'
    # Read only one representation of each biological sample, no DQN3 duplicates.
    chunks=pd.read_csv(src/'GSE15061_series_matrix.txt.gz',sep='\t',comment='!',
                       usecols=['ID_REF',*selected.sample_id],chunksize=1500,na_values=['null'])
    with gzip.open(matrix_path,'wt') as handle:
        first=True;used=[]
        wanted=set(mapping.probe_id)
        for chunk in chunks:
            chunk=chunk.loc[chunk.ID_REF.isin(wanted),['ID_REF',*selected.sample_id]]
            valid=np.isfinite(chunk.iloc[:,1:].to_numpy(float)).all(axis=1)&chunk.iloc[:,1:].gt(0).all(axis=1)
            chunk=chunk.loc[valid]
            used.extend(chunk.ID_REF)
            chunk.to_csv(handle,sep='\t',index=False,header=first);first=False
    mapping.loc[mapping.probe_id.isin(used)].to_csv(data/'GPL570_unambiguous_probe_gene_mapping.tsv',sep='\t',index=False)
    return selected


def summary(frame,x,y,label):
    f=frame.loc[np.isfinite(frame[[x,y]]).all(axis=1)]
    nz=f.loc[f[x].ne(0)&f[y].ne(0)];k=int(nz[x].mul(nz[y]).lt(0).sum())
    b=binomtest(k,len(nz),.5);ci=b.proportion_ci()
    r=pearsonr(f[x],f[y]) if len(f)>1 else SimpleNamespace(statistic=np.nan,pvalue=np.nan);s=spearmanr(f[x],f[y]) if len(f)>1 else SimpleNamespace(statistic=np.nan,pvalue=np.nan)
    return dict(comparison=label,n_CpGs=len(f),inverse_direction=k,fraction_inverse=k/len(nz),fraction_CI_low=ci.low,fraction_CI_high=ci.high,
                binomial_p_two_sided=b.pvalue,binomial_p_greater=binomtest(k,len(nz),.5,alternative='greater').pvalue,
                pearson_r=r.statistic,pearson_p=r.pvalue,spearman_rho=s.statistic,spearman_p=s.pvalue)


def make_plot(frame,stats,out):
    f=frame.loc[frame.expression_log2FC.notna()].copy()
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':14.3,'axes.labelsize':14.3,
                         'xtick.labelsize':14.3,'ytick.labelsize':14.3,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(12.5,8.5))
        handles,legend_labels=scatter_context(ax,f,out.parents[1]/'data/candidate_genomic_annotation.tsv','expression_log2FC','methylation_delta_beta',out)
        extent=max(f.expression_log2FC.abs().max()*1.25,.5)
        ax.set_xlim(-extent,extent);ax.margins(y=.22)
        ax.axvline(0,color='#aaaaaa',lw=1,zorder=0);ax.axhline(0,color='#aaaaaa',lw=1,zorder=0)
        y_span=float(f.methylation_delta_beta.max()-f.methylation_delta_beta.min())
        ax.set_ylim(f.methylation_delta_beta.min()-.25*y_span,f.methylation_delta_beta.max()+.25*y_span)
        ax.set_xlabel('GSE15061 expression log₂FC (AML / control)')
        ax.set_ylabel('Methylation Δβ (AML − control)')
        ax.set_title(f'AML: GSE15061 gene expression\nwith Δβ of {len(f)} CpGs | FDR < 0.01 and p < 0.01',fontsize=16.9,pad=12)
        ax.tick_params(direction='out',length=4,width=.8)
        ax.spines[['top','right']].set_visible(False)
        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.53,.008),ncol=3,frameon=False,fontsize=10.5,columnspacing=1.3,handletextpad=.5,labelspacing=.45)
        fig.tight_layout(rect=[0,.105,1,1],pad=.65)
        unobstructed_labels(fig,ax,f,'expression_log2FC','methylation_delta_beta',out,label_all=True)

    save_publication(fig,out,'AML_17_CpGs_methylation_vs_GSE15061_expression')


def main(base,rscript=None,prepare=False,hgnc_path=None):
    base=Path(base);out=base/'methylation_expression/cross_cohort_comparison'
    for folder in ['data','source','plots']:(out/folder).mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    hgnc=Path(hgnc_path) if hgnc_path else out/'source/HGNC_complete_set.tsv'
    canonical=symbol_mapper(hgnc)
    if prepare:
        prepare_expression(out,canonical)
        from .expression_resources import rscript_path
        subprocess.run([rscript_path(rscript),str(Path(__file__).with_name('aml_expression_limma.R')),str(out.resolve())],check=True)
    panel=pd.read_csv(base/'data/fixed_candidate_panel.tsv',sep='\t')
    frozen=pd.read_csv(Path(__file__).parent/'resources/fixed_AML_17_CpGs.tsv',sep='\t')
    assert len(panel)==17 and set(panel.CpG_ID)==set(frozen.CpG_ID)
    annotation=pd.read_csv(base/'data/candidate_genomic_annotation.tsv',sep='\t').fillna('').set_index('CpG_ID')
    models={name:pd.read_csv(out/'data'/f'all_genes_{name}_limma.tsv',sep='\t').set_index('gene') for name in ['center_adjusted','unadjusted']}
    rows=[];pairs=[]
    for row in panel.itertuples():
        ann=annotation.loc[row.CpG_ID];tokens=ann.UCSC_RefGene_Name.split(';');regions=ann.UCSC_RefGene_Group.split(';')
        genes=sorted(set(g for g in tokens if g));promoters={canonical(g) for g,r in zip(tokens,regions) if r in ['TSS200','TSS1500']}
        assert len(tokens)==len(regions)
        promoter=bool(set(regions)&{'TSS200','TSS1500'});body='Body' in regions
        region='Promoter + gene body' if promoter and body else 'Promoter' if promoter else 'Gene body' if body else 'Other / unannotated'
        c=dict(CpG_ID=row.CpG_ID,annotated_genes=';'.join(genes),gene_regions=ann.UCSC_RefGene_Group,
               genomic_region_class=region,methylation_delta_beta=row.Internal_delta_beta_AML_minus_healthy,
               external_methylation_delta_beta=row.Delta_beta_AML_minus_healthy,
               promoter_associated=promoter,CHR=ann.CHR,MAPINFO=ann.MAPINFO)
        for model,de in models.items():
            measured={};promoter_effects={}
            for gene in genes:
                mapped=canonical(gene)
                pair=dict(CpG_ID=row.CpG_ID,annotated_gene=gene,expression_gene=mapped,model=model,promoter_linked=mapped in promoters,
                          methylation_delta_beta=c['methylation_delta_beta'],mapping_method='exact' if mapped==gene else 'unambiguous HGNC previous symbol')
                if mapped in de.index:
                    v=de.loc[mapped]
                    passes=bool(expression_selected(v['P.Value'],v['adj.P.Val'])) and np.isfinite(v.logFC)
                    if passes:
                        measured[mapped]=v.logFC
                        if mapped in promoters:promoter_effects[mapped]=v.logFC
                    pair.update(status='measured',expression_selected=passes,expression_log2FC=v.logFC,expression_p=v['P.Value'],expression_FDR_genome=v['adj.P.Val'],CI_low=v['CI.L'],CI_high=v['CI.R'])
                else:pair.update(status='not measured by unambiguous probes',expression_log2FC=np.nan)
                pairs.append(pair)
            c[model+'_log2FC']=np.mean(list(measured.values())) if measured else np.nan
            c[model+'_genes_used']=';'.join(sorted(measured))
            c[model+'_promoter_log2FC']=np.mean(list(promoter_effects.values())) if promoter_effects else np.nan
            c[model+'_all_genes_measured']=all(canonical(g) in de.index for g in genes)
            c[model+'_all_genes_pass_expression_filter']=len(measured)==len({canonical(g) for g in genes})
        rows.append(c)
    frame=pd.DataFrame(rows)
    frame['expression_log2FC']=frame.center_adjusted_log2FC
    frame['expression_genes_used']=frame.center_adjusted_genes_used
    frame['direction']=np.where(frame.expression_log2FC.isna(),'unavailable',np.where(frame.expression_log2FC.mul(frame.methylation_delta_beta).lt(0),'inverse','same'))
    frame['plot_display_class']=frame.genomic_region_class.replace({'Other / unannotated':'Unannotated in plot'})
    frame.loc[frame.expression_log2FC.lt(0)&frame.methylation_delta_beta.lt(0),'plot_display_class']='Unannotated in plot'
    frame.to_csv(out/'data/all_17_cross_cohort_CpGs.tsv',sep='\t',index=False)
    pairs=pd.DataFrame(pairs);pairs['expression_selected']=pairs.expression_selected.fillna(False);pairs.to_csv(out/'data/all_CpG_gene_expression_results.tsv',sep='\t',index=False)
    pairs.loc[pairs.model.eq('center_adjusted')&pairs.expression_selected].to_csv(out/'data/selected_gene_expression_results.tsv',sep='\t',index=False)
    result=[summary(frame,'expression_log2FC','methylation_delta_beta','Primary: original methylation vs center-adjusted GSE15061 expression'),
            summary(frame,'unadjusted_log2FC','methylation_delta_beta','Sensitivity: unadjusted GSE15061 expression'),
            summary(frame,'expression_log2FC','external_methylation_delta_beta','Sensitivity: external GSE62298 minus GSE40279 methylation'),
            summary(frame,'center_adjusted_promoter_log2FC','methylation_delta_beta','Promoter-linked genes only')]
    pd.DataFrame(result).to_csv(out/'data/summary_statistics.tsv',sep='\t',index=False)
    make_plot(frame,result[0],out)
    meta=pd.read_csv(out/'data/GSE15061_sample_metadata.tsv',sep='\t')
    methyl_meta=pd.read_csv(base/'data/original_patient_annotations.tsv',sep='\t')
    overlap=set(meta.sample_id)&set(methyl_meta.sample_id);assert not overlap
    s=result[0];missing=frame.loc[frame.expression_log2FC.isna(),'CpG_ID'].tolist()
    inverse=pairs.loc[pairs.model.eq('center_adjusted')&pairs.expression_selected&pairs.expression_log2FC.mul(pairs.methylation_delta_beta).lt(0)].sort_values('expression_FDR_genome')
    inverse.to_csv(out/'data/inverse_direction_gene_pairs.tsv',sep='\t',index=False)
    report=f'''# AML: cross-cohort comparison of 17 fixed CpGs

## Data and analysis

- Expression: [GSE15061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15061), GPL570, **202 untreated AML and 69 non-leukemic bone marrow specimens**, from four centers. The 164 MDS specimens are excluded. Controls are non-leukemic, not necessarily healthy volunteers.
- The series has 870 GEO records but only 435 biological specimens, each represented as DS and DQN3. Only DS records are analyzed; DQN3 is a bounded rank-normalized signal and must not be treated as log2 expression. Duplicate records are never independent patients.
- DS signals are log2 transformed and quantile normalized across the 271 selected specimens. Nonpositive/nonfinite probes, AFFX controls, missing mappings, and probes mapping ambiguously to multiple distinct genes are excluded. Multiple unambiguous probes are averaged on the log2 scale per gene before fitting. Mapping and sample-selection audits are supplied.
- Primary differential expression: limma empirical Bayes with trend, expression ~ group + center. The AML coefficient is adjusted log2FC versus non-leukemia. Full-rank design checked. Unadjusted group-only sensitivity is supplied. Gene-level 95% intervals, raw p-values and BH FDR across all tested genes are retained.
- CpG mapping: existing GPL13534/hg19 legacy gene and transcript-context annotations, with unambiguous HGNC previous-symbol updates. No nearest-gene replacement or imputed expression is used. Expression genes must pass raw p < 0.01 AND genome-wide BH FDR < 0.01; FDR is not recalculated on the candidate panel. A point is the mean qualifying annotated-gene log2FC; all gene-level results are supplied separately. This panel has {len(frame)} CpGs, {int(frame.expression_log2FC.notna().sum())} retained after filtering; omitted (unmeasured or fails thresholds): {missing}.
- Methylation primary Y: the existing internal AML-minus-control delta beta: 62 GSE58477 AML samples minus 464 GSE50660 peripheral-blood controls. The values are preserved exactly from the fixed panel; external contrasts are sensitivity results only. Expression and methylation GSM overlap: {len(overlap)}.
- X = expression log2FC, Y = methylation delta beta; symmetric X limits center zero. Colors and symbols show promoter (TSS200/TSS1500), enhancer (legacy array Enhancer flag), gene body and overlapping contexts. Other contexts are gray unless signs are opposite; dark outlines mark opposite signs. Enhancer flags are not tissue-specific validation. Text placement checks all plotted points and previously placed labels. All plotted points receive CpG and gene labels, with arrows anchored to their actual coordinates. No explanatory footer.

## Findings

- Opposite signs: **{s['inverse_direction']}/{s['n_CpGs']} ({s['fraction_inverse']:.1%})**, exact 95% fraction interval {s['fraction_CI_low']:.1%}–{s['fraction_CI_high']:.1%}; two-sided binomial p={s['binomial_p_two_sided']:.4g}, one-sided p for excess above 50%={s['binomial_p_greater']:.4g}.
- Pearson r={s['pearson_r']:.3f}, p={s['pearson_p']:.4g}; Spearman rho={s['spearman_rho']:.3f}, p={s['spearman_p']:.4g}. See summary_statistics.tsv for promoter-specific and sensitivity results.
- Passing inverse-direction genes: {', '.join(sorted(inverse.expression_gene.unique()))}. These are exploratory priorities, not validated regulatory links. Sensitivity models apply the same thresholds separately.
- The plot is across CpG features, not a pooled patient-level methylation-expression correlation. No sample-wise correlation can be estimated by matching unrelated patients across these cohorts.
- Candidate signs and differential-expression significance are separate. Inverse promoter signs are consistent with a silencing hypothesis; inverse body signs do not establish promoter regulation. Positive gene-body associations are also biologically plausible. Legacy annotations alone do not establish active regulatory targets.

## Interpretation limits

The original AML methylation contrast completely confounds disease with study and compares AML blasts with peripheral blood of different cell composition (including varied smoking exposure in controls). Expression center adjustment does not repair methylation batch confounding. External methylation also uses separate case/control studies. Expression controls are non-leukemic marrow, not the same cell population as the original methylation controls. No purity, deconvolution, copy-number or age adjustment is possible with the available metadata used here; center adjustment is not cell-composition adjustment. GSE15061 includes varied AML subtypes, while GSE58477 is cytogenetically normal AML. Microarray values near background may be unreliable for low-expression genes; no detection-call filter was available in the downloaded series matrix. Genome annotation can miss true cis targets. The 17 CpGs were selected using methylation labels and are not independent random loci; feature-correlation and binomial p-values are nominal, not corrected for dependent CpGs or discovery selection. The operational 50% sign null is not guaranteed biologically. Significant effects do not imply methylation causes expression changes. These results prioritize hypotheses and require validation in independent, cell-comparable samples.

## Reproducibility

See all_870_records_audit, sample metadata, probe-gene mapping, full normalized gene expression, limma designs and genome-wide results, per-pair and per-CpG tables, summary statistics, R session info, and manifest.json. Frozen CpG IDs and zero methylation/expression sample overlap are checked. GSE9476 was assessed as an alternative but not selected for the main plot because of fewer patients, heterogeneous normal cell populations, and the smaller GPL96 platform; it is not required or downloaded by this workflow and was not combined with the primary expression dataset. Legacy local files, when present, are provenance only.
'''
    (out/'REPORT.md').write_text(report)
    source_urls={
        'GSE15061_series_matrix.txt.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE15nnn/GSE15061/matrix/GSE15061_series_matrix.txt.gz',
        'GPL570_probe_annotation.tsv':'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/soft/GPL570_family.soft.gz',
        'GSE9476_series_matrix.txt.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE9nnn/GSE9476/matrix/GSE9476_series_matrix.txt.gz',
        'GPL96.annot.gz':'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz'}
    inputs=[base/'data/fixed_candidate_panel.tsv',base/'data/candidate_genomic_annotation.tsv',hgnc,*[out/'source'/name for name in source_urls if (out/'source'/name).exists()]]
    manifest=dict(analysis='Cross-cohort disease effect comparison; not patient-level correlation',expression='GSE15061 DS only; AML202 vs non-leukemia69; center-adjusted limma',
                  methylation='GSE58477 AML62 minus GSE50660 controls464; inherited disease-study confounding',sources=source_urls,
                  annotation_download='GPL570 platform table extracted completely from start of gzip family stream; unused sample records and remaining stream not retained',
                  expression_filter=dict(raw_p_lt=0.01,genome_wide_BH_FDR_lt=0.01),fixed_CpGs=sorted(frame.CpG_ID),sample_overlap=sorted(overlap),unavailable_CpGs=missing,
                  hashes={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in inputs},statistics=result)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(result).to_string(index=False))
    print(frame[['CpG_ID','annotated_genes','expression_genes_used','methylation_delta_beta','expression_log2FC','genomic_region_class','direction']].to_string(index=False))
    print(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--rscript',default=None);ap.add_argument('--prepare',action='store_true');ap.add_argument('--hgnc',type=Path)
    args=ap.parse_args();main(args.base,args.rscript,args.prepare,args.hgnc)
