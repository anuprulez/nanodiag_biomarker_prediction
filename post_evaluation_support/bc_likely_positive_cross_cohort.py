"""Cross-cohort BC comparison for the unthresholded 189 common likely-positive CpGs."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .publication import save_publication
from .point_labels import label_cpg_points
from .bc_cross_cohort_comparison import statistics


def main(base):
    base=Path(base);disease=base.parents[3];dataset=base.parents[2]
    source=dataset/'accumulated_results/evaluation'/('likely_positive_common_unique_cpgs_PNA_'+dataset.name.removeprefix('data_')+'.tsv')
    panel=pd.read_csv(source,sep='\t')
    assert len(panel)==189 and panel.CpG_ID.is_unique and panel.n_runs.eq(5).all()
    assert np.isfinite(panel[['Internal_delta_beta_BC_minus_healthy','Delta_beta_BC_minus_healthy']]).all().all()
    annotation_path=disease/'evaluation/raw/GPL13534_probe_annotation.tsv'
    annotation=pd.read_csv(annotation_path,sep='\t',dtype=str).drop_duplicates('ID').set_index('ID')
    context_path=Path(__file__).parent/'resources/GPL13534_context.tsv.gz'
    context=pd.read_csv(context_path,sep='\t',dtype=str).drop_duplicates('ID').set_index('ID')
    de_paths={m:base/'direction_concordance/data'/f'all_genes_{m}_limma.tsv' for m in ['blocked','paired']}
    de={m:pd.read_csv(p,sep='\t').set_index('gene') for m,p in de_paths.items()}
    hgnc_path=base/'robust_followup/source/HGNC_complete_set.tsv'
    hgnc=pd.read_csv(hgnc_path,sep='\t',dtype=str).fillna('');hgnc=hgnc.loc[hgnc.status.eq('Approved')]
    previous={}
    for row in hgnc.itertuples():
        for old in row.prev_symbol.split('|'):
            if old:previous.setdefault(old,set()).add(row.symbol)
    def map_gene(gene,table):
        if gene in table.index:return gene,'exact symbol'
        options=previous.get(gene,set())
        if len(options)==1:
            symbol=next(iter(options))
            if symbol in table.index:return symbol,'HGNC unambiguous previous symbol'
        return None,'unavailable or expression-filtered'
    out=base/'likely_positive_cross_cohort_comparison'
    for folder in ['plots','data']:(out/folder).mkdir(parents=True,exist_ok=True)
    cpg_rows=[];gene_rows=[]
    for row in panel.itertuples():
        if row.CpG_ID not in annotation.index:raise ValueError(f'Missing platform annotation: {row.CpG_ID}')
        ann=annotation.loc[row.CpG_ID];tokens=str(ann.UCSC_RefGene_Name).split(';') if pd.notna(ann.UCSC_RefGene_Name) else []
        genes=sorted(set(g.strip() for g in tokens if g.strip()))
        region=context.loc[row.CpG_ID,'UCSC_RefGene_Group'] if row.CpG_ID in context.index else np.nan
        region_tokens=str(region).split(';') if pd.notna(region) else []
        aligned=len(tokens)==len(region_tokens) and len(tokens)>0
        promoter_genes={g.strip() for g,r in zip(tokens,region_tokens) if r in ['TSS200','TSS1500']} if aligned else set()
        c=dict(CpG_ID=row.CpG_ID,annotated_genes=';'.join(genes),gene_regions=region,
               promoter=bool(promoter_genes),gene_region_alignment_valid=aligned,promoter_annotated_genes=';'.join(sorted(promoter_genes)),
               GSE66695_delta_beta=row.Internal_delta_beta_BC_minus_healthy,
               external_delta_beta=row.Delta_beta_BC_minus_healthy,n_runs=row.n_runs)
        for model,table in de.items():
            effects={};mapped={};fdrs={};promoter_effects={}
            for gene in genes:
                expression_gene,method=map_gene(gene,table)
                record=dict(CpG_ID=row.CpG_ID,annotated_gene=gene,expression_gene=expression_gene,
                            mapping_status=method,expression_model=model,promoter_linked=gene in promoter_genes,
                            GSE66695_delta_beta=row.Internal_delta_beta_BC_minus_healthy)
                if expression_gene is not None:
                    v=table.loc[expression_gene]
                    effects[expression_gene]=v.logFC;mapped[gene]=expression_gene;fdrs[expression_gene]=v['adj.P.Val']
                    if gene in promoter_genes:promoter_effects[expression_gene]=v.logFC
                    record.update(TCGA_BRCA_log2FC=v.logFC,expression_p=v['P.Value'],expression_FDR_genome=v['adj.P.Val'],CI_low=v['CI.L'],CI_high=v['CI.R'],direction='inverse' if row.Internal_delta_beta_BC_minus_healthy*v.logFC<0 else 'same')
                gene_rows.append(record)
            c[model+'_log2FC']=float(np.mean(list(effects.values()))) if effects else np.nan
            c[model+'_expression_genes_used']=';'.join(sorted(effects))
            c[model+'_annotated_genes_covered']=len(mapped)
            c[model+'_unique_expression_genes']=len(effects)
            c[model+'_all_genes_measured']=len(mapped)==len(genes) and len(genes)>0
            c[model+'_mixed_gene_directions']=bool(any(v>0 for v in effects.values()) and any(v<0 for v in effects.values()))
            c[model+'_promoter_log2FC']=float(np.mean(list(promoter_effects.values()))) if promoter_effects else np.nan
            c[model+'_promoter_expression_genes_used']=';'.join(sorted(promoter_effects))
            c[model+'_all_measured_genes_DE_FDR_lt_0_05']=bool(fdrs and all(v<.05 for v in fdrs.values()))
        cpg_rows.append(c)
    frame=pd.DataFrame(cpg_rows);genes=pd.DataFrame(gene_rows)
    assert len(frame)==189 and frame.CpG_ID.is_unique
    def region_class(value):
        regions=set(str(value).split(';')) if pd.notna(value) else set()
        promoter=bool(regions & {'TSS200','TSS1500'});body='Body' in regions
        return 'Promoter + gene body' if promoter and body else 'Promoter' if promoter else 'Gene body' if body else 'Other / unannotated'
    frame['genomic_region_class']=frame.gene_regions.map(region_class)
    frame['plot_display_class']=frame.genomic_region_class.replace({'Other / unannotated':'Unannotated in plot'})
    frame.loc[frame.GSE66695_delta_beta.lt(0)&frame.blocked_log2FC.lt(0),'plot_display_class']='Unannotated in plot'
    frame['direction']='unavailable'
    mask=frame.blocked_log2FC.notna()&frame.GSE66695_delta_beta.ne(0)&frame.blocked_log2FC.ne(0)
    frame.loc[mask,'direction']=np.where(frame.loc[mask,'GSE66695_delta_beta']*frame.loc[mask,'blocked_log2FC']<0,'inverse','same')
    frame.to_csv(out/'data/all_189_likely_positive_CpGs.tsv',sep='\t',index=False)
    genes.to_csv(out/'data/all_annotated_gene_expression_results.tsv',sep='\t',index=False)
    records=[statistics(frame,'GSE66695_delta_beta','blocked_log2FC','Full 189 likely-positive set; evaluable CpGs only'),
             statistics(frame.loc[frame.promoter],'GSE66695_delta_beta','blocked_promoter_log2FC','Promoter-linked genes only'),
             statistics(frame,'GSE66695_delta_beta','paired_log2FC','Paired TCGA expression sensitivity'),
             statistics(frame.loc[frame.blocked_all_genes_measured&~frame.blocked_mixed_gene_directions],
                        'GSE66695_delta_beta','blocked_log2FC','Complete gene coverage and unanimous expression directions')]
    summary=pd.DataFrame(records);summary.to_csv(out/'data/summary_statistics.tsv',sep='\t',index=False)
    # Validate that the expanded comparison reproduces all 33 earlier points.
    oldpath=base/'cross_cohort_comparison/data/all_33_cross_cohort_CpGs.tsv'
    old=pd.read_csv(oldpath,sep='\t');overlap=old.merge(frame,on='CpG_ID',validate='one_to_one',suffixes=('_old','_new'))
    assert len(overlap)==33
    np.testing.assert_allclose(overlap.GSE66695_delta_beta_old,overlap.GSE66695_delta_beta_new,atol=1e-12)
    np.testing.assert_allclose(overlap.TCGA_BRCA_log2FC,overlap.blocked_log2FC,atol=1e-12)
    primary=summary.iloc[0];plotted=frame.loc[frame.blocked_log2FC.notna()]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(18,13))
        for category,color,marker in [('Unannotated in plot','#aaaaaa','o'),('Gene body','#0072B2','o'),('Promoter','#D55E00','^'),('Promoter + gene body','#7B2CBF','D')]:
            subset=plotted.loc[plotted.plot_display_class.eq(category)]
            ax.scatter(subset.blocked_log2FC,subset.GSE66695_delta_beta,s=45,color=color,marker=marker,edgecolor='white',linewidth=.4,alpha=.85,label=f'{category} (n={len(subset)})')
        # Symmetric expression limits put the biological zero at the panel center.
        x_extent = float(plotted['blocked_log2FC'].abs().max()) * 1.15
        ax.set_xlim(-x_extent, x_extent)
        ax.margins(y=.18)
        ax.tick_params(direction='out', length=4, width=.8)
        ax.axvline(0,color='#aaaaaa',lw=1,zorder=0);ax.axhline(0,color='#aaaaaa',lw=1,zorder=0)
        ax.set(xlabel='TCGA-BRCA expression log₂FC (tumor / normal)',ylabel='GSE66695 methylation Δβ (tumor − normal)',
               title=f'BC | Likely-positive CpGs · {len(plotted)}/189 evaluable')
        ax.set_title(ax.get_title()+f'\nPearson r = {primary.pearson_r:.3f}, p = {primary.pearson_p:.3g}  |  Spearman ρ = {primary.spearman_rho:.3f}, p = {primary.spearman_p:.3g}',fontsize=14)
        ax.spines[['top','right']].set_visible(False)
        handles,labels=ax.get_legend_handles_labels()
        order=[2,1,3,0]
        fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower center',bbox_to_anchor=(.53,.085),ncol=2,frameon=False,fontsize=9)
        fig.tight_layout(rect=[0,.20,1,1])
        labels=label_cpg_points(ax,plotted,'blocked_log2FC','GSE66695_delta_beta','blocked_expression_genes_used',fontsize=7.5)
        labels.to_csv(out/'data/displayed_CpG_gene_labels.tsv',sep='\t',index=False)

    save_publication(fig,out,'likely_positive_GSE66695_delta_beta_vs_TCGA_log2FC')
    unavailable=frame.loc[frame.blocked_log2FC.isna(),'CpG_ID'].tolist()
    promoter=summary.iloc[1]
    report=f'''# BC likely-positive CpGs: cross-cohort comparison

This extends the previous 33-CpG comparison to all **189 unique likely-positive CpGs common to five PNA runs**. No external |Δβ| ≥ 0.30, p-value, FDR or direction-consistency filter was applied to define this set. “Likely positive” refers to model predictions for CpG nodes, not patient diagnoses.

- Cohort A: original GSE66695 methylation, 80 tumors and 40 normals. Y = tumor-minus-normal methylation Δβ.
- Cohort B: TCGA-BRCA expression, 775 primary tumors and 84 solid-tissue normals. X = saved limma patient-blocked tumor–normal expression coefficient. Gene-level genome-wide FDR and coefficient intervals are retained.
- Expression was supplied as log₂(RSEM normalized count + 1), so log₂FC is a contrast on that scale rather than a ratio of arithmetic mean raw counts.

## Results

{int(primary.n_CpGs)}/189 CpGs have at least one usable gene-expression result. Unavailable sites: {', '.join(unavailable) if unavailable else 'none'}.

{int(primary.inverse_direction)}/{int(primary.nonzero_CpGs)} ({primary.fraction_inverse:.1%}) have opposite Δβ/log₂FC signs. Exact 95% interval: [{primary.fraction_CI_low:.1%}, {primary.fraction_CI_high:.1%}]. Two-sided binomial p versus 50% = {primary.binomial_p_two_sided:.4g}; one-sided p for an excess of inverse directions = {primary.binomial_p_greater:.4g}.

Across evaluable CpGs, Pearson r = {primary.pearson_r:.3f}, p = {primary.pearson_p:.4g}; Spearman ρ = {primary.spearman_rho:.3f}, p = {primary.spearman_p:.4g}.

For promoter-linked gene targets only, {int(promoter.inverse_direction)}/{int(promoter.nonzero_CpGs)} ({promoter.fraction_inverse:.1%}) have inverse signs; one-sided binomial p = {promoter.binomial_p_greater:.4g}. The plot distinguishes promoter-only, gene-body-only, overlapping promoter/body, and other/unannotated contexts using the platform labels. This region classification is descriptive; the promoter-specific statistics use only promoter-linked genes with usable expression results. The plot has no arrows.

The full likely-positive set shows no overall negative association between methylation and expression effects. The fraction of inverse signs is below 50% (the two-sided binomial result), rather than enriched above 50%. This must not be interpreted as failure of all candidates: gene-body and other sites are not universally expected to show inverse effects, and the promoter subset is approximately 50/50.

## Mapping and coverage

Display-only rule: points with negative Δβ and negative log₂FC are gray and unannotated in the plot. Their original genomic-region class is preserved in the table; plot_display_class records the visual override. Gray also includes other/unannotated contexts. All points outside the both-negative quadrant are labeled with CpG ID and the expression gene(s) used, with automatic overlap reduction and no arrows. Both-negative points remain unlabeled. This does not change statistical results.

GPL13534 annotation provides CpG-to-gene names; the full platform context provides aligned gene-region annotation. An expression symbol is matched exactly when available; otherwise an unambiguous HGNC previous-symbol mapping is allowed. Gene names are deduplicated before aggregation. Plot colors use any TSS200/TSS1500 tag for promoter and any Body tag for gene body; sites carrying both labels are shown separately because transcript annotations can overlap. Other/unannotated includes UTR/exon-only labels and missing labels. These are legacy platform annotations, not proof of active promoters or regulatory targets. All CpG–gene results, including missing/filtered expression genes, appear in the gene table. Promoter-specific targets are assigned only where the semicolon-separated gene and region lists align.

Each plotted CpG contributes one point: the unweighted mean of available distinct annotated-gene log₂FC coefficients. {int(frame.blocked_mixed_gene_directions.sum())} CpGs have mixed expression directions among their mapped genes, and {int((~frame.blocked_all_genes_measured).sum())} have incomplete gene-expression coverage. These flags prevent averages from being mistaken for uniform target-gene behavior. Complete/unanimous-mapping and paired-expression sensitivity analyses are supplied. The 33 previous CpG points were independently checked and reproduce exactly in this expanded table.

## Interpretation and limits

Inverse directions can support a regulatory hypothesis for an appropriately mapped promoter CpG. They are not a universal expectation for gene-body sites. Neither correlation across CpG effects nor direction agreement establishes cis regulation or methylation-driven expression. The plot compares disease-associated changes between separate cohorts; it does not correlate unrelated patients or pool tumor/normal samples into a sample-level correlation.

P-values for feature-level correlations and sign tests are nominal: CpGs can co-vary, multiple CpGs can map to the same gene, and the sites were selected by a model. The 50% sign reference is operational and can be affected by imbalanced marginal directions. Purity, subtype and copy-number differences can affect tumor–normal contrasts; the separate robust within-tumor analysis of the selected 33 sites does not automatically validate all 189 likely-positive sites. No new within-tumor correlations or purity/CN models were fitted for the expanded set in this task.

## Files

- plots/likely_positive_GSE66695_delta_beta_vs_TCGA_log2FC.pdf, .svg and 300-dpi .png.
- data/all_189_likely_positive_CpGs.tsv: every likely-positive CpG, expression aggregates, direction and mapping flags.
- data/all_annotated_gene_expression_results.tsv: individual gene log₂FC, significance, FDR and intervals.
- data/summary_statistics.tsv: all-site, promoter-target, paired-expression and strict-mapping sensitivities.
- manifest.json: source definitions and hashes.

Reproduce: python -m post_evaluation_support.bc_likely_positive_cross_cohort --base <methylation_expression_directory>
'''
    (out/'REPORT.md').write_text(report)
    inputs=[source,annotation_path,context_path,hgnc_path,*de_paths.values(),oldpath]
    (out/'manifest.json').write_text(json.dumps(dict(panel='189 BC likely-positive CpGs common to five PNA runs; no 33-site selection thresholds',
        methylation='GSE66695 internal tumor-minus-normal delta beta',expression='TCGA-BRCA full-cohort patient-blocked limma log2FC',
        one_point_per_CpG=True,aggregation='Mean available distinct annotated-gene coefficients; exact then unambiguous HGNC previous-symbol mapping',
        new_patient_level_tests=False,previous_33_exactly_reproduced=True,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
    print(summary.to_string(index=False));print('Unavailable:',unavailable,'mixed directions:',int(frame.blocked_mixed_gene_directions.sum()))
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);main(p.parse_args().base)
