"""Exploratory GSE66695 CpG delta-beta versus TCGA-BRCA gene-expression logFC."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest,spearmanr,pearsonr
import matplotlib.pyplot as plt
from .publication import save_publication
from .point_labels import label_cpg_points


def statistics(frame,x,y,label):
    use=frame.loc[frame[x].notna()&frame[y].notna()].copy()
    nonzero=use.loc[use[x].ne(0)&use[y].ne(0)]
    n=len(nonzero);k=int((nonzero[x]*nonzero[y]<0).sum());bt=binomtest(k,n,.5);ci=bt.proportion_ci()
    sr=spearmanr(use[x],use[y]);pr=pearsonr(use[x],use[y])
    return dict(comparison=label,n_CpGs=len(use),nonzero_CpGs=n,inverse_direction=k,fraction_inverse=k/n,
                fraction_CI_low=ci.low,fraction_CI_high=ci.high,binomial_p_two_sided=bt.pvalue,
                binomial_p_greater=binomtest(k,n,.5,alternative='greater').pvalue,
                spearman_rho=float(sr.statistic),spearman_p=float(sr.pvalue),pearson_r=float(pr.statistic),pearson_p=float(pr.pvalue))


def main(base):
    base=Path(base);src=base/'direction_concordance/data';out=base/'cross_cohort_comparison'
    for name in ['data','plots']:(out/name).mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(src/'all_33_CpG_concordance.tsv',sep='\t')
    assert len(frame)==33 and frame.CpG_ID.is_unique
    plotted=frame[['CpG_ID','annotated_genes','blocked_genes','original_delta_beta','blocked_logFC',
                   'blocked_mixed_gene_directions','blocked_all_genes_covered','blocked_all_genes_DE_FDR_lt_0_05']].rename(columns={
                       'blocked_genes':'expression_genes_used','original_delta_beta':'GSE66695_delta_beta',
                       'blocked_logFC':'TCGA_BRCA_log2FC','blocked_mixed_gene_directions':'mixed_gene_expression_directions',
                       'blocked_all_genes_covered':'all_annotated_genes_measured','blocked_all_genes_DE_FDR_lt_0_05':'all_measured_genes_DE_FDR_lt_0_05'})
    def region_class(value):
        regions=set(str(value).split(';')) if pd.notna(value) else set()
        promoter=bool(regions & {'TSS200','TSS1500'});body='Body' in regions
        return 'Promoter + gene body' if promoter and body else 'Promoter' if promoter else 'Gene body' if body else 'Other / unannotated'
    plotted['genomic_region_class']=frame.gene_regions.map(region_class)
    plotted['plot_display_class']=plotted.genomic_region_class.replace({'Other / unannotated':'Unannotated in plot'})
    plotted.loc[plotted.GSE66695_delta_beta.lt(0)&plotted.TCGA_BRCA_log2FC.lt(0),'plot_display_class']='Unannotated in plot'
    plotted['direction']=np.where(plotted.GSE66695_delta_beta*plotted.TCGA_BRCA_log2FC<0,'inverse','same')
    plotted['methylation_cohort']='GSE66695: 80 tumors / 40 normals'
    plotted['expression_cohort']='TCGA-BRCA: 775 tumors / 84 solid-tissue normals'
    plotted.to_csv(out/'data/all_33_cross_cohort_CpGs.tsv',sep='\t',index=False)
    genes=pd.read_csv(src/'all_annotated_gene_DE_and_methylation.tsv',sep='\t')
    genes=genes.loc[genes.model.eq('blocked'),['CpG_ID','annotated_gene','expression_gene','gene_has_TSS_annotation',
                                             'original_delta_beta','logFC','expression_p','expression_FDR_genome','CI_low','CI_high','status']].copy()
    genes['direction']=np.where(genes.logFC.isna(),'unavailable',np.where(genes.original_delta_beta*genes.logFC<0,'inverse','same'))
    genes.to_csv(out/'data/gene_level_cross_cohort_results.tsv',sep='\t',index=False)
    promoter=frame.loc[frame.promoter].copy()
    promoter['inverse_direction']=promoter.original_delta_beta*promoter.blocked_promoter_logFC<0
    promoter[['CpG_ID','promoter_annotated_genes','blocked_promoter_genes','original_delta_beta','blocked_promoter_logFC','inverse_direction']].to_csv(out/'data/promoter_specific_comparison.tsv',sep='\t',index=False)
    summary=pd.DataFrame([
        statistics(frame,'original_delta_beta','blocked_logFC','All 33 CpGs: primary cross-cohort comparison'),
        statistics(promoter,'original_delta_beta','blocked_promoter_logFC','11 legacy promoter-associated CpGs: promoter-linked genes only'),
        statistics(frame,'original_delta_beta','paired_logFC','Sensitivity: TCGA paired-expression model, 31 testable CpGs')])
    summary.to_csv(out/'data/summary_statistics.tsv',sep='\t',index=False)
    primary=summary.iloc[0];prom=summary.iloc[1]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':14.3,'axes.labelsize':14.3,'xtick.labelsize':14.3,'ytick.labelsize':14.3,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(11,8.5))
        inverse=plotted.TCGA_BRCA_log2FC.mul(plotted.GSE66695_delta_beta).lt(0)
        for mask,color,label in [(~inverse,'#B8BEC5','Other measured CpGs'),
                                 (inverse,'#D55E00','Opposite Δβ / log₂FC directions')]:
            part=plotted.loc[mask]
            if len(part):
                ax.scatter(part.TCGA_BRCA_log2FC,part.GSE66695_delta_beta,s=75 if mask.equals(inverse) else 55,
                           color=color,marker='o',edgecolor='white',linewidth=.7,
                           alpha=.95,label=f'{label} (n={len(part)})',zorder=3)
        # Symmetric expression limits put the biological zero at the panel center.
        x_extent = float(plotted['TCGA_BRCA_log2FC'].abs().max()) * 1.15
        ax.set_xlim(-x_extent, x_extent)
        ax.margins(y=.18)
        ax.tick_params(direction='out', length=4, width=.8)
        ax.axvline(0,color='#aaaaaa',lw=1,zorder=0);ax.axhline(0,color='#aaaaaa',lw=1,zorder=0)
        ax.set(xlabel='TCGA-BRCA expression log₂FC (tumor / normal)',ylabel='GSE66695 methylation Δβ (tumor − normal)')
        ax.set_title('Breast cancer: TGCA-BRCA gene expression\nwith Δβ of 33 CpGs',fontsize=16.9,pad=12)
        ax.spines[['top','right']].set_visible(False)
        handles,legend_labels=ax.get_legend_handles_labels()
        fig.legend(handles[::-1],legend_labels[::-1],loc='lower center',bbox_to_anchor=(.53,.015),ncol=1,frameon=False,fontsize=11.7)
        fig.tight_layout(rect=[0,.105,1,1],pad=.7)
        # Explicit label positions spread the near-zero cluster across both sides.
        labels=plotted.loc[plotted.TCGA_BRCA_log2FC.mul(plotted.GSE66695_delta_beta).lt(0),
                           ['CpG_ID','expression_genes_used','TCGA_BRCA_log2FC','GSE66695_delta_beta']].copy()
        upper=labels.loc[labels.TCGA_BRCA_log2FC.lt(0)]
        for row in upper.itertuples(index=False):
            below = row.CpG_ID in {'cg14947478','cg02623991'}
            ax.annotate(row.CpG_ID+'\n'+row.expression_genes_used.replace(';',' / '),
                        xy=(row.TCGA_BRCA_log2FC,row.GSE66695_delta_beta),
                        xytext=(0,-10 if below else 10),textcoords='offset points',ha='center',va='top' if below else 'bottom',
                        fontsize=11.05,zorder=5)
        lower=labels.loc[labels.TCGA_BRCA_log2FC.gt(0)].sort_values(
            ['GSE66695_delta_beta','TCGA_BRCA_log2FC'],ascending=[False,False])
        positions = {
            'cg03235603': (-.12, -.045, 'right'),
            'cg27133038': (-.12, -.135, 'right'),
            'cg24516399': (-.12, -.200, 'right'),
            'cg21570818': (-.12, -.260, 'right'),
            'cg25961292': (.24, -.045, 'left'),
            'cg23237765': (.88, -.100, 'left'),
            'cg20592707': (.73, -.156, 'left'),
            'cg15350780': (.50, -.220, 'left'),
            'cg15242225': (.50, -.306, 'left'),
        }
        for row in lower.itertuples(index=False):
            label_x,label_y,alignment=positions[row.CpG_ID]
            ax.annotate(row.CpG_ID+'\n'+row.expression_genes_used.replace(';',' / '),
                        xy=(row.TCGA_BRCA_log2FC,row.GSE66695_delta_beta),
                        xytext=(label_x,label_y),textcoords='data',
                        ha=alignment,va='center',fontsize=11.05,zorder=5,
                        arrowprops=dict(arrowstyle='->',color='#666666',lw=.8,
                                        shrinkA=4,shrinkB=5,
                                        relpos=(1 if alignment=='right' else 0,.5),
                                        connectionstyle='arc3,rad=0'),
                        bbox=dict(facecolor='white',edgecolor='none',pad=.4))
        labels.to_csv(out/'data/displayed_CpG_gene_labels.tsv',sep='\t',index=False)

    save_publication(fig,out,'GSE66695_delta_beta_vs_TCGA_expression_log2FC')
    text=f'''# BC: cross-cohort methylation–expression comparison

Methylation cohort A: GSE66695, 80 breast tumors and 40 normal breast samples. Y is the existing fixed-panel tumor-minus-normal methylation Δβ.

Expression cohort B: TCGA-BRCA, 775 primary tumors and 84 solid-tissue normal samples. X is the existing patient-blocked limma tumor–normal log₂FC; within-patient tissue dependence is modeled with duplicateCorrelation. The expression values supplied by UCSC Xena are log₂(RSEM normalized count + 1), so log₂FC is a model contrast on that transformed scale, not a ratio of arithmetic mean raw counts. Genome-wide limma FDR and coefficient intervals are retained per gene.

This compares disease-associated changes across two cohorts. No patient from cohort A is paired with an unrelated expression observation from cohort B. It is separate from the matched TCGA within-tissue correlation analysis and purity/CN sensitivity analyses.

## Results

All 33 CpGs have an expression aggregate in the full TCGA expression model. {int(primary.inverse_direction)}/33 ({primary.fraction_inverse:.1%}) have opposite Δβ/log₂FC signs. Exact 95% interval for this fraction: [{primary.fraction_CI_low:.1%}, {primary.fraction_CI_high:.1%}]. One-sided binomial p for more than 50% inverse directions: {primary.binomial_p_greater:.3f}; two-sided p: {primary.binomial_p_two_sided:.3f}.

Across the 33 features, Pearson r = {primary.pearson_r:.3f}, nominal p = {primary.pearson_p:.4f}; Spearman ρ = {primary.spearman_rho:.3f}, p = {primary.spearman_p:.3f}. A negative slope can coexist with many same-sign points because correlation concerns relative covariation, while the direction test concerns signs relative to zero. The Pearson result is not evidence that most CpGs show inverse changes; the weak Spearman result also cautions against a general monotonic relationship.

Among the 11 legacy promoter-associated sites, using only the gene(s) linked through the promoter annotation, {int(prom.inverse_direction)}/11 ({prom.fraction_inverse:.1%}) have inverse signs; one-sided binomial p = {prom.binomial_p_greater:.3f}. There is no excess inverse-direction evidence for the promoter panel as a whole. Expression-significance thresholds were not used to define the denominator. Individual inverse-direction sites include TCHH, CA3, RUNX1 and TTYH1; these are sign-concordant hypotheses, not four established regulatory links.

## Mapping and aggregation

Display-only rule: points with negative Δβ and negative log₂FC are gray and unannotated in the plot. Their original genomic-region class is preserved in the table; plot_display_class records the visual override. Gray also includes other/unannotated contexts. All points outside the both-negative quadrant are labeled with CpG ID and the expression gene(s) used, with automatic overlap reduction and no arrows. Both-negative points remain unlabeled. This does not change statistical results.

The plot has one point per CpG. When a CpG maps to multiple genes, the primary all-site plot uses the unweighted mean of available annotated-gene coefficients. Gene-level results are retained separately; averaging is a display/summary rule, not an assertion that all mapped genes are targets. cg24516399 has opposing ATG16L1/SCARNA6 expression directions, so its average should be interpreted cautiously. MIR548F1 is unavailable in the expression matrix; cg09411231 uses HMCN1 and its incomplete mapping is flagged. C7orf20 is mapped to GET4 using the previously verified HGNC mapping. The promoter-only table uses promoter-linked genes and therefore differs from the all-gene average for some CpGs.

## Interpretation

Inverse Δβ/log₂FC signs can support an exploratory regulatory hypothesis for an appropriately mapped promoter-associated site. Gene-body sites can show either sign. Directional agreement does not establish cis regulation, expression mediation or causality, and it can reflect differences in subtype, purity, cell composition, copy number or other cohort characteristics. Within-cohort contrasts reduce platform-scale problems but do not eliminate these differences.

These are selected CpGs, not randomly sampled independent loci. Binomial and feature-correlation p-values are nominal because CpGs may co-vary. The 50% sign-test reference is an operational null, not a guaranteed biological null when the marginal directions are imbalanced. The primary plot uses original GSE66695 Δβ, not the much larger external-cohort Δβ.

## Relationship to robust matched-sample follow-up

The separate robust_followup analysis found negative within-tumor associations for GDF5, TTYH1 and HOXA4 that persist under purity/CN and age/PAM50 sensitivity analyses, with transcript-TSS proximity checked. RUNX1 has no clear adjusted within-tumor association; CA3 lacks usable TCGA methylation; TCHH shows a positive within-tumor association and its updated transcript annotation differs from its legacy promoter label. Thus this cross-cohort plot does not replace that evidence.

## Files

- plots/GSE66695_delta_beta_vs_TCGA_expression_log2FC.pdf, .svg and 300-dpi .png: 33-point plot with expression log₂FC on X and methylation Δβ on Y, highlighting inverse-sign CpGs in orange and other measured CpGs in gray; inverse-quadrant labels use arrows to identify crowded points.
- data/all_33_cross_cohort_CpGs.tsv: exactly one row per candidate, with gene-coverage and mixed-direction flags.
- data/gene_level_cross_cohort_results.tsv: every annotated CpG–gene pair, limma log₂FC, raw p, genome-wide FDR and coefficient interval.
- data/promoter_specific_comparison.tsv: correct promoter-linked targets.
- data/summary_statistics.tsv: primary, promoter-specific and paired-expression sensitivity summaries.
- manifest.json: exact source tables, cohort definitions, aggregation rule and hashes.

Reproduce: python -m post_evaluation_support.bc_cross_cohort_comparison --base <methylation_expression_directory>
'''
    (out/'REPORT.md').write_text(text)
    inputs=[src/'all_33_CpG_concordance.tsv',src/'all_annotated_gene_DE_and_methylation.tsv',src/'panel_gene_blocked_DE.tsv']
    manifest=dict(methylation_cohort='GSE66695: 80 tumors, 40 normals',expression_cohort='TCGA-BRCA: 775 primary tumors, 84 solid-tissue normals',
                  expression_model='Previously fitted limma patient-blocked full-cohort model; log2(RSEM normalized count+1)',
                  sample_level_correlation=False,one_point_per_CpG=True,aggregation='Mean available annotated-gene coefficients, with mixed-direction and coverage flags',
                  statistics_recomputed='Feature-level sign tests and correlations only; gene expression DE and methylation effects reused from validated saved results',
                  input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False));print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);main(p.parse_args().base)
