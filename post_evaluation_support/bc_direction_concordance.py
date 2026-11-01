"""One-point-per-CpG comparison of BC differential methylation and expression."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp,binomtest,spearmanr,pearsonr
import matplotlib.pyplot as plt
from .publication import save_publication
from .methylation_expression import bh


def summarize(frame,beta,lfc,subset):
    f=frame.loc[frame.promoter] if subset=='TSS' else frame
    used_lfc=lfc.replace('_logFC','_promoter_logFC') if subset=='TSS' else lfc
    f=f.copy();f[lfc]=f[used_lfc]
    f=f.loc[np.isfinite(f[beta])&np.isfinite(f[lfc])].copy()
    f=f.loc[f[beta].abs().gt(1e-12)&f[lfc].abs().gt(1e-12)]
    n=len(f);k=int((f[beta]*f[lfc]<0).sum())
    result=dict(subset=subset,beta_source=beta,expression_model=lfc,n_CpGs=n,inverse_direction=k,
                fraction=k/n if n else np.nan,binomial_p_greater=binomtest(k,n,.5,alternative='greater').pvalue if n else np.nan,
                binomial_p_two_sided=binomtest(k,n,.5).pvalue if n else np.nan)
    if n:
        ci=binomtest(k,n).proportion_ci();result.update(fraction_CI_low=ci.low,fraction_CI_high=ci.high)
    if n>=3 and f[beta].nunique()>1 and f[lfc].nunique()>1:
        sr=spearmanr(f[beta],f[lfc]);pr=pearsonr(f[beta],f[lfc]);result.update(spearman_rho=sr.statistic,spearman_p=sr.pvalue,pearson_r=pr.statistic,pearson_p=pr.pvalue)
    return result


def main(base):
    base=Path(base);out=base/'direction_concordance';panelroot=base.parent
    annotation=pd.read_csv(panelroot/'data/candidate_genomic_annotation.tsv',sep='\t').fillna('')
    panel=pd.read_csv(panelroot/'data/fixed_candidate_panel.tsv',sep='\t').set_index('CpG_ID')
    meta=pd.read_csv(base/'data/sample_metadata.tsv',sep='\t',index_col=0)
    methyl=pd.read_csv(base/'data/methylation_matched.tsv.gz',sep='\t',index_col=0)
    de={name:pd.read_csv(out/'data'/f'all_genes_{name}_limma.tsv',sep='\t').set_index('gene') for name in ['paired','blocked']}
    tumor=meta.loc[meta.sample_type.eq('Primary tumor')].set_index('patient_id')
    normal=meta.loc[meta.sample_type.eq('Solid tissue normal')].set_index('patient_id')
    paired=sorted(set(tumor.index)&set(normal.index));assert len(paired)==77
    tids=[meta.index[(meta.patient_id==p)&meta.sample_type.eq('Primary tumor')][0] for p in paired]
    nids=[meta.index[(meta.patient_id==p)&meta.sample_type.eq('Solid tissue normal')][0] for p in paired]
    pairs=[];cpgs=[]
    for row in annotation.itertuples():
        genes=sorted(set(row.UCSC_RefGene_Name.split(';'))-{''})
        gene_tokens=row.UCSC_RefGene_Name.split(';');region_tokens=row.UCSC_RefGene_Group.split(';')
        assert len(gene_tokens)==len(region_tokens),f'Unaligned gene/region annotation: {row.CpG_ID}'
        promoter_genes={g for g,r in zip(gene_tokens,region_tokens) if r in {'TSS200','TSS1500'}}
        promoter=bool(promoter_genes)
        diffs=methyl.loc[row.CpG_ID,tids].to_numpy()-methyl.loc[row.CpG_ID,nids].to_numpy()
        diffs=diffs[np.isfinite(diffs)]
        delta=float(diffs.mean()) if len(diffs)>=10 else np.nan
        dp=float(ttest_1samp(diffs,0).pvalue) if len(diffs)>=10 and np.std(diffs)>0 else np.nan
        c=dict(CpG_ID=row.CpG_ID,promoter=promoter,gene_regions=row.UCSC_RefGene_Group,annotated_genes=';'.join(genes),
               original_delta_beta=panel.loc[row.CpG_ID,'Internal_delta_beta_BC_minus_healthy'],
               external_delta_beta=panel.loc[row.CpG_ID,'Delta_beta_BC_minus_healthy'],
               TCGA_paired_delta_beta=delta,TCGA_methylation_paired_n=len(diffs),TCGA_methylation_p=dp,
               original_methylation_p=np.nan)
        c['promoter_annotated_genes']=';'.join(sorted(promoter_genes))
        for model,table in de.items():
            lfcs=[];usable=[];padjs=[];prom_lfcs=[];prom_used=[]
            for original in genes:
                gene={'C7orf20':'GET4'}.get(original,original)
                if gene in table.index:
                    v=table.loc[gene];lfcs.append(v.logFC);usable.append(gene);padjs.append(v['adj.P.Val'])
                    if original in promoter_genes:prom_lfcs.append(v.logFC);prom_used.append(gene)
                    pairs.append(c|dict(model=model,gene_has_TSS_annotation=original in promoter_genes,annotated_gene=original,expression_gene=gene,logFC=v.logFC,
                                       expression_p=v['P.Value'],expression_FDR_genome=v['adj.P.Val'],CI_low=v['CI.L'],CI_high=v['CI.R'],
                                       status='tested',original_inverse_direction=bool(c['original_delta_beta']*v.logFC<0)))
                else:pairs.append(c|dict(model=model,gene_has_TSS_annotation=original in promoter_genes,annotated_gene=original,expression_gene=gene,status='unavailable or expression-filtered'))
            # Fixed aggregation: equal mean of available gene logFCs, one point per CpG.
            c[model+'_logFC']=float(np.mean(lfcs)) if lfcs else np.nan
            c[model+'_genes']=';'.join(usable)
            c[model+'_gene_count']=len(lfcs)
            c[model+'_mixed_gene_directions']=bool(any(x>0 for x in lfcs) and any(x<0 for x in lfcs))
            c[model+'_all_genes_DE_FDR_lt_0_05']=bool(padjs and all(x<.05 for x in padjs))
            c[model+'_all_genes_covered']=len(lfcs)==len(genes)
            c[model+'_promoter_logFC']=float(np.mean(prom_lfcs)) if prom_lfcs else np.nan
            c[model+'_promoter_genes']=';'.join(prom_used)
            c[model+'_promoter_all_genes_covered']=len(prom_lfcs)==len(promoter_genes)
            c[model+'_promoter_mixed_directions']=bool(any(x>0 for x in prom_lfcs) and any(x<0 for x in prom_lfcs))
        cpgs.append(c)
    frame=pd.DataFrame(cpgs);frame['TCGA_methylation_FDR']=bh(frame.TCGA_methylation_p)
    for beta in ['original_delta_beta','external_delta_beta','TCGA_paired_delta_beta']:
        for model in ['paired','blocked']:
            col=f'{beta}_{model}_inverse';frame[col]=pd.Series(pd.NA,index=frame.index,dtype='boolean')
            mask=frame[beta].notna()&frame[model+'_logFC'].notna()
            frame.loc[mask,col]=(frame.loc[mask,beta]*frame.loc[mask,model+'_logFC']<0).to_numpy()
    for beta in ['original_delta_beta','external_delta_beta','TCGA_paired_delta_beta']:
        for model in ['paired','blocked']:
            col=f'{beta}_{model}_promoter_inverse'
            frame[col]=pd.Series(pd.NA,index=frame.index,dtype='boolean')
            mask=frame[beta].notna()&frame[model+'_promoter_logFC'].notna()
            frame.loc[mask,col]=(frame.loc[mask,beta]*frame.loc[mask,model+'_promoter_logFC']<0).to_numpy()
    assert len(frame)==33 and frame.CpG_ID.is_unique and frame.promoter.sum()==11
    records=[]
    for beta in ['original_delta_beta','external_delta_beta','TCGA_paired_delta_beta']:
        for model in ['paired','blocked']:
            for subset in ['TSS','All']:records.append(summarize(frame,beta,model+'_logFC',subset))
    summary=pd.DataFrame(records)
    summary['binomial_FDR_across_12']=bh(summary.binomial_p_greater)
    summary['spearman_FDR_across_12']=bh(summary.spearman_p)
    summary['pearson_FDR_across_12']=bh(summary.pearson_p)
    # Strict sensitivity: require every annotated gene measured, directions unanimous.
    strict=frame.loc[frame.paired_all_genes_covered & ~frame.paired_mixed_gene_directions]
    strict_promoter=frame.loc[frame.paired_promoter_all_genes_covered & ~frame.paired_promoter_mixed_directions]
    strict_summary=pd.DataFrame([summarize(strict_promoter if subset=='TSS' else strict,b,'paired_logFC',subset) for b in ['original_delta_beta','TCGA_paired_delta_beta'] for subset in ['TSS','All']])
    pd.DataFrame(pairs).to_csv(out/'data/all_annotated_gene_DE_and_methylation.tsv',sep='\t',index=False)
    frame.to_csv(out/'data/all_33_CpG_concordance.tsv',sep='\t',index=False)
    summary.to_csv(out/'data/direction_concordance_summary.tsv',sep='\t',index=False)
    strict_summary.to_csv(out/'data/strict_gene_mapping_sensitivity.tsv',sep='\t',index=False)
    selected=sorted({r['expression_gene'] for r in pairs})
    for model,table in de.items():
        sel=table.reindex(selected).copy();sel['panel_only_FDR']=bh(sel['P.Value']);sel.to_csv(out/'data'/f'panel_gene_{model}_DE.tsv',sep='\t')
    figures(out,frame,summary)
    report(out,frame,summary,strict_summary,de)
    inputs=[panelroot/'data/candidate_genomic_annotation.tsv',panelroot/'data/fixed_candidate_panel.tsv',base/'data/sample_metadata.tsv',base/'data/methylation_matched.tsv.gz',out/'data/expression_genome.tsv.gz']
    inputs += [out/'data'/f'all_genes_{name}_limma.tsv' for name in de]
    manifest=dict(primary_expression='limma paired patient fixed effects: 77 patients / 154 samples; eBayes trend=TRUE, robust=TRUE; all expressed genes for variance moderation and genome-wide BH',
                  sensitivity_expression='775 tumors / 84 normals; duplicateCorrelation patient block; limma eBayes trend and robust',
                  primary_methylation='Original GSE66695 tumor-normal delta beta; cross-cohort comparison with TCGA expression',
                  primary_direction_test='TSS200/TSS1500 CpGs; one-sided exact binomial against 0.5, inverse sign alternative',
                  cpg_aggregation='One point per CpG: unweighted mean logFC over available annotated genes for all-region summaries; only aligned promoter-linked genes for TSS summaries; strict complete/unanimous mapping sensitivity supplied',
                  TCGA_methylation='Mean of available within-patient beta differences over up to 77 matched pairs; minimum 10 complete pairs; paired t test and BH across testable CpGs',
                  dependence='Binomial and feature-correlation p-values are nominal; CpGs can be correlated. Multi-gene mappings and feature selection limit inference.',
                  input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False));print(frame[['CpG_ID','annotated_genes','promoter','original_delta_beta','TCGA_paired_delta_beta','paired_logFC','paired_mixed_gene_directions']].to_string(index=False))


def figures(out,f,summary):
    style={'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        usable=f.copy()
        # Use the promoter-linked target for promoter sites, not another gene body target.
        usable['plot_log2FC']=np.where(usable.promoter,usable.blocked_promoter_logFC,usable.blocked_logFC)
        usable['plot_gene']=np.where(usable.promoter,usable.blocked_promoter_genes,usable.blocked_genes)
        de=pd.read_csv(out/'data/panel_gene_blocked_DE.tsv',sep='\t').set_index('gene')
        def target_fdr(genes):
            values=de.reindex(str(genes).split(';'))['adj.P.Val']
            return float(values.max()) if values.notna().all() and len(values) else np.nan
        usable['target_expression_FDR']=usable.plot_gene.map(target_fdr)
        usable=usable.loc[usable.original_delta_beta.notna()&usable.plot_log2FC.notna()]
        highlight=(usable.promoter.eq(True)
                   & (usable.original_delta_beta*usable.plot_log2FC).lt(0)
                   & usable.target_expression_FDR.lt(.05))
        usable['prioritized']=highlight
        other=usable.loc[~highlight];selected=usable.loc[highlight]
        fig,ax=plt.subplots(figsize=(9,7))
        ax.scatter(other.original_delta_beta,other.plot_log2FC,s=45,color='#bdbdbd',
                   edgecolor='white',linewidth=.5,alpha=.65,zorder=2,label='Other candidates')
        ax.scatter(selected.original_delta_beta,selected.plot_log2FC,s=85,color='#D55E00',
                   edgecolor='white',linewidth=.6,zorder=3,
                   label='Promoter CpGs: inverse signs, expression FDR < 0.05')
        offsets={'RUNX1':(7,7),'TCHH':(7,7),'TTYH1':(7,6),'CA3':(-98,10)}
        for row in selected.itertuples():
            ax.annotate(f'{row.plot_gene}\n{row.CpG_ID}',
                        (row.original_delta_beta,row.plot_log2FC),
                        xytext=offsets.get(row.plot_gene,(7,7)),textcoords='offset points',fontsize=9)
        ax.axvline(0,color='#aaaaaa',lw=1,zorder=1)
        ax.axhline(0,color='#aaaaaa',lw=1,zorder=1)
        ax.set(xlabel='Methylation Δβ (tumor − normal)',
               ylabel='Expression log₂FC (tumor / normal)',
               title=f'Breast cancer | {len(usable)} CpG sites')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out')
        ax.legend(loc='upper right',bbox_to_anchor=(1,1.12),frameon=False,fontsize=8)
        fig.text(.04,.015,'Priority: promoter-linked CpGs with opposite Δβ / log₂FC signs and expression BH FDR < 0.05.\nOriginal GSE66695 Δβ versus TCGA full-cohort expression. Promoter sites use their promoter-linked gene.',fontsize=9)
        fig.tight_layout(rect=[0,.07,1,1])
        columns=['CpG_ID','plot_gene','original_delta_beta','plot_log2FC','target_expression_FDR','prioritized']
        usable[columns].rename(columns={'original_delta_beta':'delta_beta','plot_log2FC':'log2FC',
                                       'plot_gene':'expression_gene'}).to_csv(
                out/'data/delta_beta_log2FC_plotted.tsv',sep='\t',index=False)
        selected[columns].to_csv(out/'data/prioritized_promoter_CpGs.tsv',sep='\t',index=False)
    save_publication(fig,out,'delta_beta_vs_expression_logFC')
    with plt.rc_context(style):
        show=summary.loc[(summary.expression_model.eq('paired_logFC')&~summary.beta_source.eq('external_delta_beta')) | (summary.expression_model.eq('blocked_logFC')&summary.beta_source.eq('original_delta_beta'))]
        fig,ax=plt.subplots(figsize=(10,6))
        for i,row in enumerate(show.itertuples()):
            ax.barh(i,row.fraction,color='#0072B2' if row.subset=='TSS' else '#888888',height=.6)
            ax.errorbar(row.fraction,i,xerr=[[row.fraction-row.fraction_CI_low],[row.fraction_CI_high-row.fraction]],fmt='none',ecolor='black',capsize=3)
            ax.text(1.03,i,f'{row.inverse_direction}/{row.n_CpGs}; p={row.binomial_p_greater:.3g}',va='center')
        ax.set_yticks(range(len(show)),[('Original BC' if r.beta_source=='original_delta_beta' else 'Paired TCGA')+' / '+('paired DE' if r.expression_model=='paired_logFC' else 'full DE')+' · '+('TSS only' if r.subset=='TSS' else 'All regions') for r in show.itertuples()])
        ax.axvline(.5,ls='--',color='#999999');ax.set_xlim(0,1.45);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Fraction with inverse Δβ / logFC signs')
        ax.spines[['top','right']].set_visible(False)
        ax.set_title('BC | Inverse-direction concordance',fontweight='bold')
        fig.text(.03,.02,'Exact binomial 95% intervals and one-sided nominal p-values vs 50%.\nCpGs may be correlated; the inverse-direction hypothesis is primarily for promoter-associated sites.',fontsize=9)
        fig.tight_layout(rect=[0,.12,1,1])
    save_publication(fig,out,'inverse_direction_concordance')


def report(out,f,summary,strict,de):
    lines=['# BC: tumor–normal differential expression and CpG direction concordance','',
           '## Analysis','',
           'Expression was modeled using limma on the existing log2(RSEM normalized count + 1) values. DESeq2 was not applied to transformed data. The primary comparison uses 77 matched TCGA tumor/normal patient pairs with patient fixed effects. All expressed genes enter empirical-Bayes moderation (trend=TRUE, robust=TRUE), not only the panel genes. Genome-wide BH-adjusted p-values and 95% logFC intervals are supplied. Positive logFC means higher expression in tumor. Because of the supplied +1 transformation, this coefficient is the difference of mean log2(normalized count+1), not the log ratio of arithmetic mean raw counts.','',
           f'The primary model tests {len(de["paired"]):,} expressed genes. A full-cohort sensitivity model uses 775 tumors and 84 normals with duplicateCorrelation patient blocking (consensus correlation estimated using the 77 complete tumor/normal pairs) and tests {len(de["blocked"]):,} genes. The supplied expression filter requires finite values, nonzero variance and log2 expression ≥1 in at least 10% of the normal-group size (8 samples in paired model, 9 in full model).','',
           'Original GSE66695 Δβ is compared with TCGA expression across all 33 selected CpGs. This is a cross-cohort comparison, not matched multi-omics within the original cohort. External discovery/evaluation Δβ and available paired TCGA Δβ are supplied separately. TCGA Δβ is the average within-patient tumor–normal beta difference using at least 10 complete pairs; its contributing patients can differ by CpG.','',
           'Promoter genes are identified from the aligned semicolon-separated gene and gene-region annotation lists. In the TSS-only analysis, only genes with a TSS200/TSS1500 relation to that CpG enter the expression aggregate. For example, cg24516399 is promoter-associated with SCARNA6 but gene-body-associated with ATG16L1; cg05389084 is promoter-associated with PTX3 but gene-body-associated with VEPH1. The primary biological test concerns the 11 CpGs with a TSS200 or TSS1500 annotation. These are legacy transcript-relative annotations, not proof of a functional promoter. An inverse direction is Δβ × logFC < 0. Other regions are summarized descriptively using the same rule; gene-body sites are not universally expected to show inverse effects. No expression-significance filtering is applied before the primary sign test.','',
           '## Summary','',
           '| Δβ source | Expression model | Sites | Inverse / evaluable | Fraction | One-sided binomial p | Spearman ρ | Spearman p |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.beta_source} | {r.expression_model} | {r.subset} | {r.inverse_direction}/{r.n_CpGs} | {r.fraction:.1%} | {r.binomial_p_greater:.4g} | {getattr(r,"spearman_rho",np.nan):.3f} | {getattr(r,"spearman_p",np.nan):.4g} |')
    primary=summary.loc[summary.beta_source.eq('original_delta_beta')&summary.expression_model.eq('paired_logFC')&summary.subset.eq('TSS')].iloc[0]
    full=summary.loc[summary.beta_source.eq('original_delta_beta')&summary.expression_model.eq('blocked_logFC')&summary.subset.eq('All')].iloc[0]
    lines+=['','## Main findings','',
            f'The primary promoter-specific result is {primary.inverse_direction}/{primary.n_CpGs} inverse-direction sites ({primary.fraction:.1%}); one-sided exact binomial p = {primary.binomial_p_greater:.4f}, two-sided p = {primary.binomial_p_two_sided:.4f}. There is no evidence for more inverse-direction promoter sites than the 50% reference. Original promoter Δβ versus paired expression has Spearman ρ = {primary.spearman_rho:.3f}, p = {primary.spearman_p:.3f}.','',
            f'The full-cohort expression sensitivity covers all 33 sites: {full.inverse_direction}/{full.n_CpGs} ({full.fraction:.1%}) have inverse signs, one-sided p = {full.binomial_p_greater:.3f}. Across these 33 features, Spearman ρ = {full.spearman_rho:.3f} (p = {full.spearman_p:.3f}), while Pearson r = {full.pearson_r:.3f} (nominal p = {full.pearson_p:.4f}). These statistics answer different questions; a negative across-feature slope does not require opposite signs relative to zero.','',
            'The four inverse-direction promoter sites in the primary comparison are cg20592707–TCHH, cg09435929–CA3, cg15242225–RUNX1 and cg02623991–TTYH1. Examples of same-direction promoter changes include hypomethylation with lower CADM3 or PCBP3 expression. SCARNA6, not ATG16L1, is the promoter-linked gene for cg24516399; its small nonsignificant expression change must not be treated as evidence of regulation.','',
            'KCNG4 and MEG8 fail the paired-model expression filter, so the all-region paired summary has 31 evaluable sites. Both pass the full-cohort filter, enabling the 33-site sensitivity result. These rows remain present in the audit and gene tables.','',
            'Some external-Δβ feature correlations are significant nominally, despite low opposite-sign fractions. The original and external Δβ signs agree by panel construction, so their sign-test counts are identical; their magnitudes differ and therefore their correlations differ. These external comparisons are secondary and were affected by the original panel selection. All feature-correlation p-values and BH-over-12 sensitivity values are in the summary table.','']
    lines+=['','Each correlation has one data point per CpG. Tests are nominal: CpGs can co-vary, and the selected panel is not a random sample of independent loci. The binomial null of 50% is an operational reference, not a guaranteed biological null; marginal biases in methylation and expression directions can affect it. Two-sided binomial p-values, exact intervals and BH over the 12 directional summaries are also supplied. The prespecified primary test is original Δβ / paired expression / TSS sites.','',
            '## Multi-gene handling and coverage','',
            'Gene-level DE results are retained separately. To prevent one CpG contributing multiple independent points, the across-CpG summary uses the equal mean of available annotated-gene logFCs in the all-region analysis; the TSS-only analysis uses only the promoter-linked gene(s). This represents an aggregate expression change and does not identify a causal target. Mixed gene directions and incomplete gene coverage are flagged. A stricter sensitivity excludes CpGs unless every annotated gene has a DE estimate and all gene directions agree. C7orf20 maps to GET4 using the saved HGNC lookup. MIR548F1 is absent from the expression matrix; no missing gene or CpG values are imputed.','',
            f'All 33 sites are included in the audit. {int(f.paired_logFC.notna().sum())} have a paired expression aggregate; {int(f.TCGA_paired_delta_beta.notna().sum())} have paired TCGA Δβ; {int(f.paired_mixed_gene_directions.sum())} have mixed expression directions among mapped genes.','',
            '## Site-level results','',
            '| CpG | Genes used (all) | TSS gene(s) | Original Δβ | Paired TCGA Δβ | All-gene paired logFC | TSS-gene paired logFC |','|---|---|---|---:|---:|---:|---:|']
    for r in f.itertuples():lines.append(f'| {r.CpG_ID} | {r.paired_genes} | {r.paired_promoter_genes} | {r.original_delta_beta:.3f} | {r.TCGA_paired_delta_beta:.3f} | {r.paired_logFC:.3f} | {r.paired_promoter_logFC:.3f} |')
    lines+=['','## Interpretation limits','',
            'An inverse methylation/expression direction supports a regulatory hypothesis but does not establish that methylation causes expression change or mediates BC. Concordance across features is distinct from the earlier within-tumor patient correlation. A site can show inverse tumor–normal changes yet a positive correlation within tumors. Tumor purity, cell populations, copy number and subtype composition can influence both measurements. Normal tissue is TCGA solid-tissue normal, not necessarily unrelated healthy donor tissue. The primary paired design controls stable patient characteristics but does not remove tumor–normal tissue-composition differences.','',
            'Do not label all gene-body same-direction pairs as biologically discordant. Opposite-direction results are a hypothesis for promoter sites; they are not a universal expectation at every CpG. Selection used methylation labels in original and external cohorts, so the sign test describes this selected panel.','',
            '## Files','',
            '• data/panel_gene_paired_DE.tsv and panel_gene_blocked_DE.tsv: each annotated gene, logFC, raw p, genome-wide FDR and panel-only FDR.','• data/all_annotated_gene_DE_and_methylation.tsv: every annotated CpG–gene pair, including unavailable mappings.','• data/all_33_CpG_concordance.tsv: exactly one row per CpG.','• data/direction_concordance_summary.tsv: sign tests, intervals and feature-level correlations.','• data/strict_gene_mapping_sensitivity.tsv: complete/unanimous mapping results.','• data/all_genes_*_limma.tsv: complete genome-wide DE tables.','• plots/: publication PDF/SVG/PNG scatterplots and concordance summary, plus limma mean–variance diagnostic.','• source/: cached full-expression rows and R session information.','• manifest.json: methods and input hashes.','',
            'Reproduce with post_evaluation_support/fetch_bc_expression_genome.py, then bc_differential_expression.R <methylation_expression_directory>, then python -m post_evaluation_support.bc_direction_concordance --base <methylation_expression_directory>.','',
            'Method sources: https://bioconductor.org/packages/release/bioc/html/limma.html ; https://github.com/ucscXena/xenaPython']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True,type=Path);main(parser.parse_args().base)
