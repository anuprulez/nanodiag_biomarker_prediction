"""Exploratory GSE66695 CpG delta-beta versus TCGA-BRCA gene-expression logFC."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest,spearmanr,pearsonr
import matplotlib.pyplot as plt
from .publication import save_publication
from .expression_filter import expression_selected
from .cpg_scatter_style import scatter_context,unobstructed_labels
from .point_labels import label_cpg_points


def statistics(frame,x,y,label):
    use=frame.loc[frame[x].notna()&frame[y].notna()].copy()
    nonzero=use.loc[use[x].ne(0)&use[y].ne(0)]
    n=len(nonzero);k=int((nonzero[x]*nonzero[y]<0).sum());bt=binomtest(k,n,.5);ci=bt.proportion_ci()
    sr=spearmanr(use[x],use[y]) if len(use)>1 else SimpleNamespace(statistic=np.nan,pvalue=np.nan);pr=pearsonr(use[x],use[y]) if len(use)>1 else SimpleNamespace(statistic=np.nan,pvalue=np.nan)
    return dict(comparison=label,n_CpGs=len(use),nonzero_CpGs=n,inverse_direction=k,fraction_inverse=k/n,
                fraction_CI_low=ci.low,fraction_CI_high=ci.high,binomial_p_two_sided=bt.pvalue,
                binomial_p_greater=binomtest(k,n,.5,alternative='greater').pvalue,
                spearman_rho=float(sr.statistic),spearman_p=float(sr.pvalue),pearson_r=float(pr.statistic),pearson_p=float(pr.pvalue))


def main(base):
    base=Path(base);src=base/'direction_concordance/data';out=base/'cross_cohort_comparison'
    for name in ['data','plots']:(out/name).mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(src/'all_33_CpG_concordance.tsv',sep='\t')
    assert len(frame)==33 and frame.CpG_ID.is_unique
    genes=pd.read_csv(src/'all_annotated_gene_DE_and_methylation.tsv',sep='\t')
    genes=genes.loc[genes.model.eq('blocked'),['CpG_ID','annotated_gene','expression_gene','gene_has_TSS_annotation',
                                             'original_delta_beta','logFC','expression_p','expression_FDR_genome','CI_low','CI_high','status']].copy()
    genes['expression_selected']=expression_selected(genes.expression_p,genes.expression_FDR_genome)&genes.logFC.notna()
    selected=genes.loc[genes.expression_selected].drop_duplicates(['CpG_ID','expression_gene'])
    for prefix, subset in [('blocked',selected),('blocked_promoter',selected.loc[selected.gene_has_TSS_annotation.eq(True)])]:
        frame[prefix+'_logFC']=frame.CpG_ID.map(subset.groupby('CpG_ID').logFC.mean())
        frame[prefix+'_genes']=frame.CpG_ID.map(subset.groupby('CpG_ID').expression_gene.agg(lambda x:';'.join(sorted(set(x))))).fillna('')
    frame['blocked_mixed_gene_directions']=frame.CpG_ID.map(selected.groupby('CpG_ID').logFC.agg(lambda x:bool((x>0).any() and (x<0).any()))).fillna(False)
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
    plotted['direction']=np.where(plotted.TCGA_BRCA_log2FC.isna(),'unavailable',np.where(plotted.GSE66695_delta_beta*plotted.TCGA_BRCA_log2FC<0,'inverse','same'))
    plotted['methylation_cohort']='GSE66695: 80 tumors / 40 normals'
    plotted['expression_cohort']='TCGA-BRCA: 775 tumors / 84 solid-tissue normals'
    plotted.to_csv(out/'data/all_33_cross_cohort_CpGs.tsv',sep='\t',index=False)
    genes['direction']=np.where(genes.logFC.isna(),'unavailable',np.where(genes.original_delta_beta*genes.logFC<0,'inverse','same'))
    genes.to_csv(out/'data/gene_level_cross_cohort_results.tsv',sep='\t',index=False)
    genes.loc[genes.expression_selected].to_csv(out/'data/selected_gene_expression_results.tsv',sep='\t',index=False)
    plotted=plotted.loc[plotted.TCGA_BRCA_log2FC.notna()].copy()
    promoter=frame.loc[frame.promoter].copy()
    promoter['inverse_direction']=promoter.original_delta_beta*promoter.blocked_promoter_logFC<0
    promoter[['CpG_ID','promoter_annotated_genes','blocked_promoter_genes','original_delta_beta','blocked_promoter_logFC','inverse_direction']].to_csv(out/'data/promoter_specific_comparison.tsv',sep='\t',index=False)
    summary=pd.DataFrame([
        statistics(frame,'original_delta_beta','blocked_logFC','Primary: FDR < 0.01 and p < 0.01'),
        statistics(promoter,'original_delta_beta','blocked_promoter_logFC','Selected promoter-linked genes only'),
        statistics(frame,'original_delta_beta','paired_logFC','Unfiltered sensitivity: TCGA paired-expression model')])
    summary.to_csv(out/'data/summary_statistics.tsv',sep='\t',index=False)
    primary=summary.iloc[0];prom=summary.iloc[1]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':14.3,'axes.labelsize':14.3,'xtick.labelsize':14.3,'ytick.labelsize':14.3,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(12.5,8.5))
        handles,legend_labels=scatter_context(ax,plotted,base.parent/'data/candidate_genomic_annotation.tsv','TCGA_BRCA_log2FC','GSE66695_delta_beta',out)
        # Symmetric expression limits put the biological zero at the panel center.
        x_extent = float(plotted['TCGA_BRCA_log2FC'].abs().max()) * 1.15
        ax.set_xlim(-x_extent, x_extent)
        ax.margins(y=.18)
        ax.tick_params(direction='out', length=4, width=.8)
        ax.axvline(0,color='#aaaaaa',lw=1,zorder=0);ax.axhline(0,color='#aaaaaa',lw=1,zorder=0)
        ax.set(xlabel='TCGA-BRCA expression log₂FC (breast cancer / normal)',ylabel='GSE66695 methylation Δβ (breast cancer − normal)')
        ax.set_title(f'Breast cancer: TCGA-BRCA gene expression\nwith Δβ of {len(plotted)} CpGs | FDR < 0.01 and p < 0.01',fontsize=16.9,pad=12)
        ax.spines[['top','right']].set_visible(False)
        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.53,.008),ncol=3,frameon=False,fontsize=10.5,columnspacing=1.3,handletextpad=.5,labelspacing=.45)
        fig.tight_layout(rect=[0,.105,1,1],pad=.65)
        unobstructed_labels(fig,ax,plotted,'TCGA_BRCA_log2FC','GSE66695_delta_beta',out,label_all=True)

    save_publication(fig,out,'GSE66695_delta_beta_vs_TCGA_expression_log2FC')
    text=f'''# BC: filtered cross-cohort methylation–expression comparison

Expression genes must have raw p < 0.01 AND genome-wide BH FDR < 0.01 in the existing patient-blocked TCGA-BRCA limma model. FDR is not recalculated on the candidate panel. No minimum absolute log2FC is imposed.

Each point averages only qualifying annotated-gene coefficients. CpGs with no qualifying gene are omitted from the plot, but all 33 fixed candidates remain in all_33_cross_cohort_CpGs.tsv with missing filtered expression where appropriate. All gene results and selection flags remain in gene_level_cross_cohort_results.tsv; selected_gene_expression_results.tsv contains passing pairs. Promoter summaries use only qualifying promoter-linked genes. The paired-model sensitivity is explicitly unfiltered.

Retained: {len(plotted)}/33 CpGs; inverse signs: {int(primary.inverse_direction)}/{int(primary.n_CpGs)}. Pearson r={primary.pearson_r:.3f}, p={primary.pearson_p:.4g}; Spearman rho={primary.spearman_rho:.3f}, p={primary.spearman_p:.4g}. Feature-level statistics are exploratory after candidate and expression selection.

Methylation: GSE66695, 80 tumors / 40 normals. Expression: TCGA-BRCA, 775 tumors / 84 normals, patient-blocked limma on log2(RSEM normalized count + 1). This compares effects across independent cohorts, not patient-level methylation–expression correlations. Inverse signs alone do not establish regulation or causality. Legacy gene contexts do not establish active targets.

All plotted CpGs are labeled, using genes that pass both thresholds. See summary_statistics.tsv for primary, promoter and unfiltered paired sensitivity results.
'''
    (out/'REPORT.md').write_text(text)
    inputs=[src/'all_33_CpG_concordance.tsv',src/'all_annotated_gene_DE_and_methylation.tsv',src/'panel_gene_blocked_DE.tsv']
    manifest=dict(methylation_cohort='GSE66695: 80 tumors, 40 normals',expression_cohort='TCGA-BRCA: 775 primary tumors, 84 solid-tissue normals',
                  expression_model='Previously fitted limma patient-blocked full-cohort model; log2(RSEM normalized count+1)',
                  sample_level_correlation=False,one_point_per_CpG=True,expression_filter=dict(raw_p_lt=0.01,genome_wide_BH_FDR_lt=0.01),aggregation='Mean qualifying annotated-gene coefficients only',
                  statistics_recomputed='Feature-level sign tests and correlations only; gene expression DE and methylation effects reused from validated saved results',
                  input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False));print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);main(p.parse_args().base)
