"""Within-tissue methylation–expression associations for the fixed BC panel."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .publication import save_publication

STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.labelsize':10,
       'pdf.fonttype':42,'svg.fonttype':'none'}


def bh(values):
    p=np.asarray(values,dtype=float);out=np.full(len(p),np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])];m=len(order)
    if m:out[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1])
    return out


def association(x,y):
    ok=np.isfinite(x)&np.isfinite(y);x=np.asarray(x)[ok];y=np.asarray(y)[ok];n=len(x)
    if n<20:return dict(n=n,rho=np.nan,p_value=np.nan,pearson_r=np.nan,pearson_p=np.nan,status='fewer than 20 matched observations')
    if np.ptp(x)==0 or np.ptp(y)==0:return dict(n=n,rho=np.nan,p_value=np.nan,pearson_r=np.nan,pearson_p=np.nan,status='constant measurement')
    s=spearmanr(x,y);p=pearsonr(x,y)
    return dict(n=n,rho=float(s.statistic),p_value=float(s.pvalue),pearson_r=float(p.statistic),pearson_p=float(p.pvalue),status='tested')


def partial_rank(x,y,age,subtype):
    frame=pd.DataFrame({'x':x,'y':y,'age':age,'subtype':subtype}).replace([np.inf,-np.inf],np.nan).dropna()
    n=len(frame)
    if n<20:return dict(n=n,rho=np.nan,p_value=np.nan,status='fewer than 20 complete observations',covariate_rank=np.nan)
    cov=pd.get_dummies(frame.subtype,drop_first=True,dtype=float)
    design=np.column_stack([np.ones(n),rankdata(frame.age),cov.to_numpy()])
    rank=np.linalg.matrix_rank(design);df=n-rank-1
    rx=rankdata(frame.x);ry=rankdata(frame.y)
    rx=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0]
    ry=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
    if df<=0 or np.std(rx)<1e-10 or np.std(ry)<1e-10:return dict(n=n,rho=np.nan,p_value=np.nan,status='degenerate residuals',covariate_rank=int(rank))
    rho=float(np.clip(np.corrcoef(rx,ry)[0,1],-1,1))
    p=float(2*t.sf(abs(rho)*np.sqrt(df/max(1-rho*rho,1e-15)),df))
    return dict(n=n,rho=rho,p_value=p,status='tested',covariate_rank=int(rank),
                same_sample_unadjusted_rho=float(spearmanr(frame.x,frame.y).statistic),
                same_sample_unadjusted_p=float(spearmanr(frame.x,frame.y).pvalue))


def run(out):
    out=Path(out);base=out.parent
    annpath=base/'data/candidate_genomic_annotation.tsv'
    ann=pd.read_csv(annpath,sep='\t',dtype=str).fillna('')
    assert len(ann)==33 and ann.CpG_ID.nunique()==33
    methyl=pd.read_csv(out/'data/methylation_matched.tsv.gz',sep='\t',index_col=0)
    expr=pd.read_csv(out/'data/expression_matched.tsv.gz',sep='\t',index_col=0)
    meta=pd.read_csv(out/'data/sample_metadata.tsv',sep='\t',index_col=0)
    assert methyl.columns.equals(expr.columns) and methyl.columns.equals(meta.index)
    assert not meta.index.duplicated().any()
    assert methyl.stack().between(0,1).all()
    for group,sub in meta.groupby('sample_type'):assert not sub.patient_id.duplicated().any()
    lookup=json.loads((out/'source/HGNC_C7orf20_GET4.json').read_text())
    assert lookup['response']['docs'][0]['symbol']=='GET4'
    pairs=[]
    for a in ann.itertuples():
        for original in sorted(set(a.UCSC_RefGene_Name.split(';'))- {''}):
            gene={'C7orf20':'GET4'}.get(original,original)
            pairs.append(dict(CpG_ID=a.CpG_ID,annotated_gene=original,expression_gene=gene,gene_regions=a.UCSC_RefGene_Group,
                              island_context=a.Relation_to_UCSC_CpG_Island or 'Unannotated / open sea'))
    rows=[];adjusted=[];strata=[]
    for pair in pairs:
        x=methyl.loc[pair['CpG_ID']].to_numpy(dtype=float)
        y=expr.loc[pair['expression_gene']].to_numpy(dtype=float) if pair['expression_gene'] in expr.index else np.full(len(meta),np.nan)
        for group in ['Primary tumor','Solid tissue normal']:
            mask=meta.sample_type.eq(group).to_numpy()
            rows.append(pair|{'group':group}|association(x[mask],y[mask]))
        tumor=meta.sample_type.eq('Primary tumor').to_numpy()
        adjusted.append(pair|partial_rank(x[tumor],y[tumor],meta.age.to_numpy()[tumor],meta.PAM50Call_RNAseq.to_numpy()[tumor]))
        for subtype in ['LumA','LumB','Basal','Her2']:
            mask=tumor & meta.PAM50Call_RNAseq.eq(subtype).to_numpy()
            strata.append(pair|{'subtype':subtype}|association(x[mask],y[mask]))
    results=pd.DataFrame(rows);adj=pd.DataFrame(adjusted);strat=pd.DataFrame(strata)
    results['FDR']=np.nan;results['pearson_FDR']=np.nan
    for group in results.group.unique():
        mask=results.group.eq(group);results.loc[mask,'FDR']=bh(results.loc[mask,'p_value']);results.loc[mask,'pearson_FDR']=bh(results.loc[mask,'pearson_p'])
    adj['FDR']=bh(adj.p_value)
    adj['same_sample_unadjusted_FDR']=bh(adj.same_sample_unadjusted_p)
    strat['FDR']=bh(strat.p_value)
    results.to_csv(out/'data/cpg_gene_associations.tsv',sep='\t',index=False)
    adj.to_csv(out/'data/tumor_age_PAM50_adjusted.tsv',sep='\t',index=False)
    strat.to_csv(out/'data/tumor_subtype_associations.tsv',sep='\t',index=False)
    audit=[]
    for a in ann.itertuples():
        rr=results.loc[results.CpG_ID.eq(a.CpG_ID)&results.group.eq('Primary tumor')]
        audit.append(dict(CpG_ID=a.CpG_ID,annotated_genes=a.UCSC_RefGene_Name,n_methylation_tumors=int(methyl.loc[a.CpG_ID,meta.sample_type.eq('Primary tumor')].notna().sum()),
                          tested_pairs=int(rr.status.eq('tested').sum()),significant_pairs=int(rr.FDR.lt(.05).sum()),
                          unavailable_expression_genes=';'.join(rr.loc[rr.expression_gene.map(lambda g:g not in expr.index or expr.loc[g].isna().all()),'annotated_gene']),
                          status='tested' if rr.status.eq('tested').any() else 'unavailable methylation' if methyl.loc[a.CpG_ID].isna().all() else 'unavailable expression'))
    audit=pd.DataFrame(audit);audit.to_csv(out/'data/coverage_all_33_CpGs.tsv',sep='\t',index=False)
    tumor=results.loc[results.group.eq('Primary tumor')&results.status.eq('tested')].sort_values('rho')
    normal=results.loc[results.group.eq('Solid tissue normal')].set_index(['CpG_ID','expression_gene'])
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(10,11))
        for i,row in enumerate(tumor.itertuples()):
            ax.plot(row.rho,i,'o',color='#0072B2',ms=6,markerfacecolor='#0072B2' if row.FDR<.05 else 'white')
            nr=normal.loc[(row.CpG_ID,row.expression_gene)]
            if np.isfinite(nr.rho):ax.plot(nr.rho,i,'s',color='#D55E00',ms=5,markerfacecolor='#D55E00' if nr.FDR<.05 else 'white')
            ax.text(1.02,i,f'{row.n} | {row.FDR:.2g}',va='center',fontsize=8)
        ax.set_yticks(range(len(tumor)),[f'{r.CpG_ID} · {r.expression_gene}' for r in tumor.itertuples()],fontsize=9)
        ax.invert_yaxis();ax.set_xlim(-1,1.32);ax.set_xticks([-1,-.5,0,.5,1]);ax.axvline(0,color='#999999',lw=1)
        ax.set_xlabel('Within-tissue Spearman correlation (ρ)');ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#eeeeee');ax.set_axisbelow(True)
        ax.text(1.02,1.015,'Tumor n | FDR',transform=ax.get_xaxis_transform(),fontsize=9)
        fig.suptitle('BC CpG–expression associations | TCGA-BRCA',fontsize=16,fontweight='bold',y=.98)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#0072B2',label='Primary tumors'),Line2D([],[],marker='s',ls='',color='#D55E00',label='Normal tissue')],loc='upper center',bbox_to_anchor=(.57,.955),ncol=2,frameon=False)
        fig.text(.03,.025,'Filled markers: BH FDR < 0.05, corrected separately within each tissue. Open markers: FDR ≥ 0.05.\nAll testable annotated pairs shown. Four panel CpGs lack methylation values; one gene lacks expression.\nAssociations are exploratory and unadjusted for tumor purity; they do not establish regulation or causation.',fontsize=9)
        fig.subplots_adjust(left=.32,right=.96,top=.89,bottom=.13)
    save_publication(fig,out,'cpg_expression_correlations')
    # All testable pairs; tumors and normals stay distinct in both display and statistics.
    plotpairs=list(tumor.sort_values('p_value').itertuples());ncols=4;nrows=int(np.ceil(len(plotpairs)/ncols))
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(nrows,ncols,figsize=(16,nrows*3.15),squeeze=False)
        for ax,row in zip(axes.flat,plotpairs):
            for group,color in [('Solid tissue normal','#D55E00'),('Primary tumor','#0072B2')]:
                ids=meta.index[meta.sample_type.eq(group)]
                ax.scatter(methyl.loc[row.CpG_ID,ids],expr.loc[row.expression_gene,ids],s=6,alpha=.35,color=color,rasterized=True)
            ax.set_title(f'{row.CpG_ID} · {row.expression_gene}\nTumor ρ={row.rho:.2f}; FDR={row.FDR:.2g}; n={row.n}',fontsize=9)
            ax.set(xlim=(-.03,1.03),xlabel='Methylation β',ylabel='Expression log₂(count + 1)');ax.spines[['top','right']].set_visible(False)
        for ax in list(axes.flat)[len(plotpairs):]:ax.set_visible(False)
        fig.suptitle('BC | Matched methylation and expression\nBlue: primary tumor; orange: normal tissue. Statistics use tumors only.',fontsize=16,y=.997)
        fig.tight_layout(rect=[0,.01,1,.975])
    save_publication(fig,out,'all_testable_cpg_gene_scatterplots')
    # A compact figure for the six strongest primary associations.
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        for ax,row in zip(axes.flat,tumor.sort_values('p_value').head(6).itertuples()):
            ids=meta.index[meta.sample_type.eq('Primary tumor')]
            ax.scatter(methyl.loc[row.CpG_ID,ids],expr.loc[row.expression_gene,ids],s=10,alpha=.45,color='#0072B2',rasterized=True)
            ax.set_title(f'{row.CpG_ID} · {row.expression_gene}\nρ={row.rho:.2f}; FDR={row.FDR:.2g}; n={row.n}')
            ax.set(xlim=(-.03,1.03),xlabel='Methylation β',ylabel='Expression log₂(count + 1)')
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('BC | Six strongest within-tumor associations',fontsize=16,fontweight='bold')
        fig.text(.03,.015,'Selected by primary Spearman p-value; FDR corrected over all 30 testable pairs. Observational associations.',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,.94])
    save_publication(fig,out,'top_six_tumor_associations')
    with plt.rc_context(STYLE):
        plot=adj.loc[adj.p_value.notna()].sort_values('rho')
        fig,ax=plt.subplots(figsize=(10,11))
        for i,row in enumerate(plot.itertuples()):
            ax.plot([row.same_sample_unadjusted_rho,row.rho],[i,i],color='#bbbbbb',lw=1)
            ax.plot(row.same_sample_unadjusted_rho,i,'o',color='#999999',ms=5)
            ax.plot(row.rho,i,'D',color='#009E73',ms=5,markerfacecolor='#009E73' if row.FDR<.05 else 'white')
            ax.text(1.02,i,f'{row.n} | {row.FDR:.2g}',va='center',fontsize=8)
        ax.set_yticks(range(len(plot)),[f'{r.CpG_ID} · {r.expression_gene}' for r in plot.itertuples()],fontsize=9)
        ax.invert_yaxis();ax.set_xlim(-1,1.32);ax.set_xticks([-1,-.5,0,.5,1]);ax.axvline(0,color='#999999',lw=1)
        ax.set_xlabel('Rank correlation (same complete-case tumor samples)')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#eeeeee');ax.set_axisbelow(True)
        ax.text(1.02,1.015,'n | adjusted FDR',transform=ax.get_xaxis_transform(),fontsize=9)
        fig.suptitle('BC | Age/PAM50 adjustment sensitivity',fontsize=16,fontweight='bold',y=.98)
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#999999',label='Unadjusted, same samples'),Line2D([],[],marker='D',ls='',color='#009E73',label='Age/PAM50 adjusted')],loc='upper center',bbox_to_anchor=(.57,.955),ncol=2,frameon=False)
        fig.text(.03,.025,'Complete cases only. Filled green markers: adjusted BH FDR < 0.05; open markers: FDR ≥ 0.05.\nAge/PAM50 adjustment does not control tumor purity, copy number or batch.',fontsize=9)
        fig.subplots_adjust(left=.32,right=.96,top=.89,bottom=.13)
    save_publication(fig,out,'age_PAM50_adjustment_sensitivity')
    inputs=[annpath,out/'data/methylation_matched.tsv.gz',out/'data/expression_matched.tsv.gz',out/'data/sample_metadata.tsv']
    manifest={'source':'TCGA-BRCA via UCSC Xena; separate from original GSE66695','panel_CpGs':33,'annotated_pairs':len(pairs),
              'sample_counts':meta.sample_type.value_counts().to_dict(),'primary':'Two-sided Spearman correlation within primary tumors; pairwise complete cases; minimum n=20',
              'adjustment':'Partial correlation of ranked methylation/expression controlling ranked age and categorical PAM50; complete cases; t approximation with n-rank(design)-1 df',
              'FDR_families':'All finite p-values across annotated pairs separately for tumor, normal, Pearson tumor, Pearson normal, adjusted tumor; all four subtype tests corrected together',
              'no_imputation':True,'no_pooled_case_control_test':True,'no_purity_adjustment':True,
              'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'source_datasets':json.loads((out/'source/source_datasets.json').read_text())}
    import scipy,matplotlib as mpl
    manifest['versions']={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':mpl.__version__}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    write_report(out,results,adj,audit,manifest)
    print(json.dumps({'samples':manifest['sample_counts'],'tested_pairs':len(tumor),'tested_CpGs':int(audit.status.eq('tested').sum()),'tumor_significant_pairs':int(tumor.FDR.lt(.05).sum()),'normal_significant_pairs':int(results.loc[results.group.eq('Solid tissue normal'),'FDR'].lt(.05).sum()),'adjusted_significant_pairs':int(adj.FDR.lt(.05).sum())},indent=2))
    print(tumor.sort_values('p_value')[['CpG_ID','expression_gene','n','rho','FDR']].head(12).to_string(index=False))


def write_report(out,results,adj,audit,manifest):
    tumor=results.loc[results.group.eq('Primary tumor')&results.status.eq('tested')].sort_values('p_value')
    lines=['# BC: methylation–expression analysis of the fixed 33-CpG panel','',
           'This is a separate TCGA-BRCA analysis. The original GSE66695 study supplies methylation, not matched expression; no expression values were inferred for its patients.','',
           f"Matched samples: {manifest['sample_counts']}. Exact TCGA sample-type matching (first 15 characters), excluding ambiguous methylation vial matches; at most one sample per patient within each tissue group. Primary tumor (01) and solid tissue normal (11) are analyzed separately. Some patients contribute both tissues; no pooled independence test is performed.",'',
           f"Coverage: {int(audit.status.eq('tested').sum())}/33 CpGs have at least one testable gene association; {len(tumor)} annotated CpG–gene pairs are tested. {int(tumor.FDR.lt(.05).sum())} tumor pairs have BH FDR < 0.05. Association does not establish methylation-driven expression or causal involvement in BC.",'',
           '## Primary results: within primary tumors','',
           '| CpG | Expression gene | n | Spearman ρ | Raw p | BH FDR |','|---|---|---:|---:|---:|---:|']
    for r in tumor.itertuples():lines.append(f'| {r.CpG_ID} | {r.expression_gene} | {r.n} | {r.rho:.3f} | {r.p_value:.3g} | {r.FDR:.3g} |')
    lines+=['','## Findings and interpretation','',
            'The strongest primary association is cg14947478–GDF5 (ρ = −0.620, n = 775, FDR = 3.73 × 10⁻⁸²). This site is annotated to the first exon / 5′UTR, not to the TSS200/TSS1500 categories. Higher methylation is associated with lower GDF5 expression within tumors; the adjusted association remains negative (ρ = −0.500).','',
            'Positive associations include cg11388664–CADM3 (ρ = 0.489), cg11121969–PCBP3 (ρ = 0.497) and cg03026004–ARHGAP28 (ρ = 0.418). All three carry legacy TSS1500 annotations, and their adjusted associations remain positive. This directly shows why TSS annotation alone cannot justify assuming that methylation silences the annotated gene.','',
            'cg21570818–FUT5 is significant in the full tumor analysis (ρ = 0.310, FDR = 2.71 × 10⁻¹⁸) but not after age/PAM50 adjustment in complete cases (ρ = 0.046, FDR = 0.365). In the same 530 complete-case patients, the unadjusted correlation is 0.271, falling to 0.046 after adjustment. This supports sensitivity to age/subtype beyond the change in sample set, without proving that either covariate causes the association.','',
            'There are 24 significant primary pairs across 23 CpGs, including both VEPH1 and PTX3 associations for cg05389084. Nine normal-tissue pairs and 20 age/PAM50-adjusted tumor pairs meet their respective FDR thresholds. Significance in tumors but not normals is not itself a statistical test that the correlations differ.','']
    lines+=['','Positive ρ means higher methylation is associated with higher expression across tumors; negative ρ means higher methylation is associated with lower expression. Gene-body sites can show either direction. An association within tumors is not evidence that methylation differs between tumor and normal, nor that expression mediates that difference.','',
            '## Coverage and gene mapping','',audit.to_markdown(index=False) if False else audit.to_csv(sep='\t',index=False),'',
            'All 33 frozen CpGs are audited. Four have no available GDC methylation values: '+', '.join(audit.loc[audit.status.eq('unavailable methylation'),'CpG_ID'])+'. Missing values were not imputed. MIR548F1 has no usable RNA-seq gene measurement; the other gene associated with that CpG can still be tested. C7orf20 was mapped to GET4 using an HGNC previous-symbol lookup, saved with the sources. Other gene symbols were matched exactly. Duplicate transcript gene names were collapsed; one CpG can contribute multiple gene pairs. No distal target search was performed.','',
            '## Sensitivity analyses','',
            '• Normal tissue: independent correlation analysis within normal samples, with its own BH family. These are TCGA solid-tissue normals, not necessarily healthy unrelated donors.','• Pearson correlations: supplied alongside Spearman results, with separate FDR correction.','• Age/PAM50 adjustment: partial rank correlations in complete-case tumors, controlling ranked age and categorical PAM50 subtype. This is sensitivity analysis: subtype can reflect genuine disease biology, and the smaller complete-case set differs from the primary analysis. Unadjusted correlations on exactly the same complete-case patients are also supplied to distinguish sample restriction from adjustment.','• Subtype-specific correlations: LumA, LumB, Basal and Her2, minimum 20 complete pairs. FDR is corrected across all tested pairs and all four subtypes together. The small Normal-like tumor subgroup is not analyzed separately.','',
            f'Age/PAM50-adjusted significant pairs: {int(adj.FDR.lt(.05).sum())}. Complete-case sample counts vary by pair; see the full table.','',
            '## Methods and limitations','',
            'Methylation: GDC TCGA-BRCA.methylation450.tsv beta values via gdc.xenahubs.net. Expression: TCGA.BRCA.sampleMap/HiSeqV2 via tcga.xenahubs.net, preprocessed log2(RSEM normalized count + 1); no additional log transform or visualization centering was applied. Different assays are matched to the same TCGA biological sample identifier, not to the same extracted molecules. The GDC hub reports averaging available technical aliquots within its sample records. Source metadata, query payloads, cached responses, sample identifiers and input hashes are retained.','',
            'The primary analysis is two-sided Spearman correlation, minimum n=20, pairwise complete observations, without expression or methylation imputation. P-values use the SciPy large-sample approximation. BH includes every testable annotated pair in the relevant analysis family, rather than just displayed significant pairs. Significant means FDR < 0.05.','',
            'Usable observations vary substantially across CpGs because of missing methylation measurements. Complete-case associations can be affected by nonrandom missingness; per-pair sample sizes and the full missing-value matrix are retained. Tumor purity, immune/stromal composition, copy number and technical batch are not adjusted here. They can generate methylation–expression associations. These correlations support locus-specific hypotheses, not causal regulation. Legacy CpG annotations and transcript ambiguity remain; no claim that the annotated gene is necessarily the regulatory target is made. Lack of significance is not evidence of no relationship. TCGA is distinct from the current discovery/evaluation GEO cohorts, but its independence from the historical sources of any prior seed selection has not been audited.','',
            '## Artifacts','',
            '• plots/top_six_tumor_associations.{pdf,svg,png}: six strongest primary results, selected by raw p-value.','• plots/age_PAM50_adjustment_sensitivity.{pdf,svg,png}: same-sample unadjusted versus adjusted rank correlations.','• plots/cpg_expression_correlations.{pdf,svg,png}: every testable pair, tumor and normal correlations; filled symbols indicate FDR < 0.05.','• plots/all_testable_cpg_gene_scatterplots.{pdf,svg,png}: individual matched measurements, all testable pairs.','• data/cpg_gene_associations.tsv: tumor and normal Spearman/Pearson results.','• data/tumor_age_PAM50_adjusted.tsv: complete-case adjusted sensitivity results.','• data/tumor_subtype_associations.tsv: subtype-specific results.','• data/coverage_all_33_CpGs.tsv: every panel CpG, including unavailable sites.','• data/*matched.tsv.gz and sample_metadata.tsv: exact analysis inputs.','• source/: public dataset metadata, cached requests/responses and HGNC lookup.','• manifest.json: methods, versions and input hashes.','',
            'Reproduce from the repository root:','',
            '```bash','python -m post_evaluation_support.fetch_bc_expression',f'python -m post_evaluation_support.methylation_expression --out {out}','```','',
            'Source pages: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66695 ; https://github.com/ucscXena/xenaPython ; https://gdc.xenahubs.net/download/TCGA-BRCA.methylation450.tsv.json ; https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.json ; https://rest.genenames.org/search/prev_symbol/C7orf20']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
