"""Candidate-panel publication figures from validated methylation/expression results.

Adaptations of Liu et al., Bioscience Reports 2020 (PMC7263199). These plots
use the selected CpGs, not the paper's genome-wide or multi-cancer analyses.
"""
from pathlib import Path
import argparse,json,hashlib,shutil,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage,leaves_list
from .publication import save_publication

STYLE={'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.labelsize':12,
       'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}
COLORS={'All candidates':'#777777','Hypermethylated':'#D55E00','Hypomethylated':'#0072B2'}
REGIONS=['TSS1500','TSS200',"5'UTR",'1stExon','Body',"3'UTR",'Unannotated']
ISLANDS=['Island','N_Shore','S_Shore','N_Shelf','S_Shelf','Unspecified / open sea']


def gene_label(value):
    return ' / '.join(dict.fromkeys(g.strip() for g in str(value).split(';') if g.strip() and g.strip() != 'nan'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(out,disease):
    base=out/'methylation_expression';cross=base/'cross_cohort_comparison/data'
    bc=disease=='BC';n=33 if bc else 17
    ann=pd.read_csv(out/'data/candidate_genomic_annotation.tsv',sep='\t').set_index('CpG_ID')
    effects=pd.read_csv(cross/('all_33_cross_cohort_CpGs.tsv' if bc else 'all_17_cross_cohort_CpGs.tsv'),sep='\t').set_index('CpG_ID')
    effects=effects.rename(columns={'GSE66695_delta_beta':'methylation_delta_beta','TCGA_BRCA_log2FC':'expression_log2FC'})
    assert len(ann)==n and ann.index.is_unique and set(ann.index)==set(effects.index)
    ann=ann.loc[effects.index].copy()
    ann['island']=ann.Relation_to_UCSC_CpG_Island.fillna('Unspecified / open sea').replace('','Unspecified / open sea')
    genes=pd.read_csv(cross/('gene_level_cross_cohort_results.tsv' if bc else 'all_CpG_gene_expression_results.tsv'),sep='\t')
    if not bc:genes=genes.loc[genes.model.eq('center_adjusted')].copy()
    genes=genes.rename(columns={'logFC':'expression_log2FC'})
    return ann,effects,genes


def context(ann,effects,out,name):
    records=[];members=[]
    for cpg,row in ann.iterrows():
        genes=str(row.UCSC_RefGene_Name).split(';') if pd.notna(row.UCSC_RefGene_Name) else []
        regions=str(row.UCSC_RefGene_Group).split(';') if pd.notna(row.UCSC_RefGene_Group) else ['Unannotated']
        for region in set(regions):members.append(dict(CpG_ID=cpg,region=region))
        # Gene-specific legacy contexts are paired only where vector lengths agree.
        for i,gene in enumerate(genes):
            if gene:records.append(dict(CpG_ID=cpg,gene=gene,island=row['island'],region=regions[i] if len(regions)==len(genes) else 'Unannotated'))
    membership=pd.DataFrame(members);gene_ann=pd.DataFrame(records).drop_duplicates()
    masks={'All candidates':np.ones(len(effects),bool),'Hypermethylated':effects.methylation_delta_beta.gt(0).to_numpy(),
           'Hypomethylated':effects.methylation_delta_beta.lt(0).to_numpy()}
    data=[]
    with plt.rc_context({**STYLE, 'font.size':14, 'axes.titlesize':17,
                         'axes.labelsize':15, 'xtick.labelsize':14, 'ytick.labelsize':14}):
        fig,axes=plt.subplots(2,2,figsize=(16,11))
        for axis,(kind,unit,categories) in enumerate([('island','CpG',ISLANDS),('region','CpG',REGIONS),('island','gene',ISLANDS),('region','gene',REGIONS)]):
            ax=axes.flat[axis];xx=np.arange(len(categories));width=.25
            for j,(label,mask) in enumerate(masks.items()):
                ids=set(effects.index[mask]);values=[]
                for cat in categories:
                    if unit=='gene':count=gene_ann.loc[gene_ann.CpG_ID.isin(ids)&gene_ann[kind].eq(cat),'gene'].nunique()
                    elif kind=='region':count=membership.loc[membership.CpG_ID.isin(ids)&membership.region.eq(cat),'CpG_ID'].nunique()
                    else:count=int(ann.loc[list(ids),'island'].eq(cat).sum())
                    value=100*count/len(ids) if unit=='CpG' and ids else count
                    values.append(value);data.append(dict(panel='ABCD'[axis],unit=unit,context=kind,category=cat,group=label,n_CpGs=len(ids),count=count,plotted_value=value))
                ax.bar(xx+(j-1)*width,values,width,color=COLORS[label],label=f'{label} (n={len(ids)})')
            ax.set_xticks(xx,[c.replace('Unspecified / open sea','Unspecified /\nopen sea') for c in categories],rotation=30,ha='right');ax.set_ylabel('Selected CpGs (%)' if unit=='CpG' else 'Distinct annotated genes')
            ax.set_title(f'{"ABCD"[axis]}   {"CpG" if unit=="CpG" else "Gene"} distribution by {"island context" if kind=="island" else "gene region"}',loc='left')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
        fig.suptitle(f'{name}: genomic context of {len(ann)} candidate CpGs',fontsize=23,y=.99)
        fig.subplots_adjust(left=.075,right=.985,top=.90,bottom=.18,hspace=.75,wspace=.30)
        save_publication(fig,out,'reference_figure2_genomic_context')
    pd.DataFrame(data).to_csv(out/'data/reference_figure2_counts.tsv',sep='\t',index=False)
    gene_ann.to_csv(out/'data/candidate_gene_context_memberships.tsv',sep='\t',index=False)


def heatmaps(source,ann,out,name):
    panels={}
    for cohort in ['original','external']:
        beta=pd.read_csv(source/'data'/f'{cohort}_candidate_beta.tsv.gz',sep='\t',index_col=0)
        meta=pd.read_csv(source/'data'/f'{cohort}_patient_annotations.tsv',sep='\t',index_col=0).loc[beta.index]
        assert beta.index.is_unique and set(ann.index)==set(beta.columns)
        assert set(meta.label)=={0,1}
        panels[cohort]=(beta.loc[:,ann.index],meta)
    original=panels['original'][0].to_numpy(float).T
    if not np.isfinite(original).all():raise ValueError('Candidate heatmaps require complete beta values')
    order=leaves_list(linkage(original,method='average',metric='euclidean',optimal_ordering=True))
    ids=ann.index[order].tolist()
    pd.DataFrame({'row_order':np.arange(len(ids)), 'CpG_ID':ids}).to_csv(out/'data/heatmap_CpG_order.tsv',sep='\t',index=False)
    labels=[c+'  '+gene_label(ann.loc[c,'UCSC_RefGene_Name']) for c in ids]
    with plt.rc_context({**STYLE, 'font.size':14, 'axes.titlesize':17,
                         'axes.labelsize':15, 'xtick.labelsize':14, 'ytick.labelsize':14}):
        fig,axes=plt.subplots(1,2,figsize=(20,max(8,.34*len(ids)+2.2)))
        for i,(cohort,(beta,meta)) in enumerate(panels.items()):
            columns=[]
            for group in [0,1]:
                group_ids=meta.index[meta.label.eq(group)]
                x=beta.loc[group_ids,ids].to_numpy(float)
                if not np.isfinite(x).all():raise ValueError('Nonfinite beta values in heatmap')
                group_order=leaves_list(linkage(x,method='average',metric='euclidean')) if len(x)>1 else np.arange(len(x))
                columns.extend(group_ids[group_order].tolist())
            ordered=meta.loc[columns].copy();ordered['plot_column']=np.arange(len(columns))
            ordered.to_csv(out/'data'/f'{cohort}_heatmap_sample_order.tsv',sep='\t')
            ax=axes[i];im=ax.imshow(beta.loc[columns,ids].to_numpy(float).T,aspect='auto',cmap='RdBu_r',vmin=0,vmax=1,interpolation='nearest',rasterized=True)
            n0=int(meta.label.eq(0).sum());n1=int(meta.label.eq(1).sum());ax.axvline(n0-.5,color='white',lw=1.5)
            ax.set_xticks([(n0-1)/2,n0+(n1-1)/2],[f'Control (n={n0})',f'Disease (n={n1})'])
            ax.tick_params(axis='x',length=0,pad=10,labelsize=14);ax.set_yticks(np.arange(len(ids)),labels if i==0 else [],fontsize=11)
            ax.set_title(f'{"BC"[i]}   {cohort.capitalize()} methylation cohort',loc='left',pad=18,fontsize=18)
            ax.tick_params(axis='y',length=0,pad=6)
            for spine in ax.spines.values():spine.set_visible(False)
            for a,b,color in [(0,n0,'#0072B2'),(n0,len(columns),'#D55E00')]:ax.plot([a-.5,b-.5],[-.9,-.9],color=color,lw=4,clip_on=False)
        cax=fig.add_axes([.94,.30,.012,.4])
        colorbar=fig.colorbar(im,cax=cax,label='Methylation β')
        colorbar.ax.tick_params(labelsize=13)
        colorbar.set_label('Methylation β',fontsize=16,labelpad=12)
        fig.suptitle(f'{name}: methylation of {len(ids)} candidate CpGs',fontsize=24,y=.99)
        fig.subplots_adjust(left=.28,right=.925,bottom=.07,top=.88,wspace=.08)
        save_publication(fig,out,'reference_figure4_BC_cohort_heatmaps')


def effects_map(ann,effects,genes,out,name):
    """Observed differential effects, never an invented sample-level correlation."""
    ids=effects.sort_values('methylation_delta_beta',ascending=False).index.tolist()
    labels=[c+'  '+gene_label(ann.loc[c,'UCSC_RefGene_Name']) for c in ids]
    gene_names=sorted(set(genes.expression_gene.dropna().astype(str)))
    matrix=pd.DataFrame(np.nan,index=ids,columns=gene_names)
    for (cpg,gene),part in genes.dropna(subset=['expression_gene']).groupby(['CpG_ID','expression_gene']):
        values=part.expression_log2FC.dropna().unique()
        if len(values)>1:raise ValueError('Conflicting expression coefficients for one annotated gene')
        if len(values):matrix.loc[cpg,gene]=values[0]
    matrix.to_csv(out/'data/annotated_gene_expression_effect_matrix.tsv',sep='\t')
    effects.loc[ids,['methylation_delta_beta','expression_log2FC']].to_csv(out/'data/CpG_effect_heatmap_values.tsv',sep='\t')
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(max(16,.30*len(gene_names)+7),max(8,.24*len(ids)+3)))
        grid=fig.add_gridspec(1,3,width_ratios=[1,1,max(7,len(gene_names)*.35)],wspace=.12)
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e5e5e5')
        for i,(column,label) in enumerate([('methylation_delta_beta','Methylation Δβ'),('expression_log2FC','Expression log₂FC')]):
            ax=fig.add_subplot(grid[i]);values=effects.loc[ids,[column]].to_numpy(float)
            limit=max(float(np.nanmax(np.abs(values))),.01)
            im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set_xticks([0],[label],rotation=60,ha='right');ax.set_yticks(np.arange(len(ids)),labels if i==0 else [],fontsize=8)
            ax.tick_params(axis='y',length=0);fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.22,ticks=[-limit,limit],format='%.2g')
            if i==0:ax.set_title('A   Disease effects',loc='left',fontsize=13,pad=16)
        ax=fig.add_subplot(grid[2]);limit=max(float(np.nanmax(np.abs(matrix.to_numpy()))),.01)
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_xticks(np.arange(len(gene_names)),gene_names,rotation=60,ha='right',fontsize=8)
        ax.set_yticks([]);ax.set_title('B   Annotated-gene expression effects',loc='left',pad=16)
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.22,ticks=[-limit,limit],format='%.2g');cb.set_label('Gene expression log₂FC')
        fig.suptitle(f'{name}: candidate methylation and expression changes',fontsize=18,y=.98)
        fig.text(.54,.015,'Cross-cohort effect comparison, not cis/trans correlation. Gray: no annotated pair or unavailable expression.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.20,right=.985,top=.88,bottom=.11)
        save_publication(fig,out,'reference_figure3_effect_maps')


def locus_map(ann,effects,out,name):
    """Coordinate annotation tracks, explicitly not measured ATAC/ChIP tracks."""
    frame=ann.copy();frame['chr_sort']=pd.to_numeric(frame.CHR,errors='coerce').fillna(99)
    frame['coordinate']=pd.to_numeric(frame.MAPINFO,errors='coerce')
    if frame.coordinate.isna().any():raise ValueError('Candidate CpG has no coordinate for locus plot')
    frame=frame.sort_values(['chr_sort','coordinate']);ids=frame.index.tolist();y=np.arange(len(ids))
    context=[]
    for text in frame.UCSC_RefGene_Group.fillna(''):
        regions=set(text.split(';'));p=bool(regions&{'TSS200','TSS1500'});b='Body' in regions
        context.append('Promoter + gene body' if p and b else 'Promoter' if p else 'Gene body' if b else 'Other')
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(17,max(7,.25*len(ids)+2)),gridspec_kw={'width_ratios':[2,1,1]},sharey=True)
        for i,c in enumerate(context):
            color={'Promoter':'#D55E00','Gene body':'#0072B2','Promoter + gene body':'#7B2CBF','Other':'#999999'}[c]
            axes[0].scatter(frame.coordinate.iloc[i]/1e6,i,c=color,s=50,zorder=3)
            axes[0].text(1.02,i,'chr'+str(frame.CHR.iloc[i]).replace('.0',''),transform=axes[0].get_yaxis_transform(),va='center',fontsize=8)
        axes[0].set_yticks(y,[c+'  '+gene_label(frame.loc[c,'UCSC_RefGene_Name']) for c in ids],fontsize=8)
        axes[0].set_xlabel('Position on annotated chromosome (Mb, hg19)');axes[0].set_title('CpG genomic positions')
        for ax,key,label in [(axes[1],'methylation_delta_beta','Original methylation Δβ'),(axes[2],'expression_log2FC','Expression log₂FC')]:
            values=effects.loc[ids,key].to_numpy(float);ax.barh(y,values,color=np.where(values>=0,'#D55E00','#0072B2'),height=.65)
            for row in np.flatnonzero(~np.isfinite(values)):
                ax.text(0,row,'NA',ha='center',va='center',color='#777777',fontsize=9,fontstyle='italic')
            ax.axvline(0,color='#555555',lw=.7);ax.set_xlabel(label);ax.tick_params(axis='y',left=False,labelleft=False)
        for ax in axes:
            ax.set_ylim(len(ids)-.3,-.8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        legend=[Patch(color=color,label=label) for label,color in [('Promoter','#D55E00'),('Gene body','#0072B2'),('Promoter + gene body','#7B2CBF'),('Other','#999999')]]
        fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.57,.025),fontsize=10)
        fig.suptitle(f'{name}: candidate loci and disease-associated changes',fontsize=18,y=.985)
        fig.text(.57,.005,'Annotation/effect tracks only. No ATAC-seq or ChIP-seq signal is represented.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.22,right=.98,bottom=.15,top=.89,wspace=.37)
        save_publication(fig,out,'reference_figure6_candidate_locus_map')
    frame.assign(genomic_context=context).to_csv(out/'data/candidate_locus_tracks.tsv',sep='\t')


def run(source,disease):
    source=Path(source).resolve();out=source/'methylation_expression/publication_figures'
    for folder in ['data','plots']:(out/folder).mkdir(parents=True,exist_ok=True)
    name='Breast cancer' if disease=='BC' else 'AML'
    ann,effects,genes=inputs(source,disease)
    context(ann,effects,out,name);effects_map(ann,effects,genes,out,name)
    heatmaps(source,ann,out,name);locus_map(ann,effects,out,name)
    report=f'''# {name}: candidate-panel publication figures

Reference: [Liu et al., Bioscience Reports 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7263199/).

These are explicitly adapted candidate-panel figures, not reproductions of the paper's full analysis. They use {len(ann)} frozen CpGs. No new candidate selection, expression fitting, enrichment testing, patient correlation, or chromatin-signal inference is performed. All inputs are saved outputs of this disease's post-evaluation workflow.

| Reference | Export | Scope and limitation |
| --- | --- | --- |
| Figure 2 | reference_figure2_genomic_context | Island/gene-region distributions of all selected, hypermethylated and hypomethylated CpGs; distinct annotated-gene counts. Denominators are selected CpGs, not the full array. Gene-region categories can overlap. Blank island context is explicitly unspecified/open sea. Gene-specific regions are assigned only when legacy gene/region vectors align. |
| Figure 3 | reference_figure3_effect_maps | Between-cohort Δβ and log₂FC heatmap and a sparse matrix of annotated genes' differential expression. Neither panel is a cis/trans correlation matrix. Non-annotated or unavailable entries are gray; no unmeasured CpG–gene association is filled in. |
| Figure 4 B/C | reference_figure4_BC_cohort_heatmaps | Original and external patient-level methylation β heatmaps. Panel C shows the external cohort, not other cancer types. These figures do not establish cancer specificity. Samples cluster within known disease groups; CpG rows cluster using original-cohort methylation and use identical order externally. No row z scaling is applied. |
| Figure 6 | reference_figure6_candidate_locus_map | Candidate CpG coordinate annotations and observed methylation/expression effects, without invented gene boundaries, ATAC-seq or H3K27ac ChIP-seq signals. It is not a chromatin-accessibility analysis. |

The paper's actual Figure 3 requires matched patient methylation/expression, Figure 4C requires measurements across additional cancer types, and Figure 6 requires chromatin datasets. The saved cross-cohort disease-effect comparison alone cannot supply those measurements. These adaptations preserve the data's meaning and should be captioned accordingly in a manuscript. A study reproducing the original analysis types would require separately curated inputs and methods.

Methylation β values describe the original and external cohorts. Δβ uses the original cohort. Expression uses the existing TCGA-BRCA patient-blocked model for BC or GSE15061 center-adjusted model for AML. Gene effects are genome-wide moderated limma results from the existing workflow. AML expression is available for 16/17 CpGs; the unavailable site remains gray in effect heatmaps. All 17 are present in methylation and annotation figures. Multiple gene annotations can share a CpG and are not independent features.

The existing main cross-cohort scatterplot has the requested two-line title with Δβ and omits Pearson/Spearman coefficients and their P values. Numerical statistics remain in saved tables. CpG labels remain restricted to inverse-sign quadrants and avoid all plotted points. Colors and symbols indicate promoter, enhancer, gene body and overlapping contexts; dark outlines indicate inverse signs. CpGs outside those contexts are gray unless signs are inverse. Enhancer membership comes from the legacy array annotation, not tissue-specific enhancer validation. BC's requested title spelling TGCA-BRCA is retained; source metadata correctly identify TCGA-BRCA.

Exports: vector PDF, editable SVG and 300-dpi PNG. The five-page candidate_methylation_expression_figures.pdf combines the updated main scatterplot with the four adapted figures. Each of these five figures is also exported as an individual PDF directly in the parent methylation_expression folder. Data tables retain memberships, effect matrices and heatmap sample/CpG orders for review. External-label-based selection and study/cell-composition confounding remain relevant; these figures are exploratory.

## Recreate

From the disease directory:

```bash
python run_pipeline.py --stage post-evaluation --folder data_corr_0.33_knn_n_3000_n_neg_10000 --post-evaluation-analysis cross-cohort
```

Standalone from the repository root after the existing cross-cohort results are available:

```bash
python -m post_evaluation_support.expression_publication --out {source} --disease {disease}
```
'''
    (out/'README.md').write_text(report)
    data=out/'data';checks={}
    for cohort in ['original','external']:
        samples=pd.read_csv(data/f'{cohort}_heatmap_sample_order.tsv',sep='\t')
        checks[f'{cohort}_samples']=len(samples)
        assert samples.sample_id.is_unique and set(samples.label)=={0,1}
    assert len(pd.read_csv(data/'heatmap_CpG_order.tsv',sep='\t'))==len(ann)
    stems=['reference_figure2_genomic_context','reference_figure3_effect_maps','reference_figure4_BC_cohort_heatmaps','reference_figure6_candidate_locus_map']
    exports=[out/'plots'/f'{stem}.{ext}' for stem in stems for ext in ['pdf','svg','png']]
    for path in exports:assert path.stat().st_size>1000
    from pypdf import PdfWriter
    main_stem='GSE66695_delta_beta_vs_TCGA_expression_log2FC' if disease=='BC' else 'AML_17_CpGs_methylation_vs_GSE15061_expression'
    main_pdf=source/'methylation_expression/cross_cohort_comparison/plots'/f'{main_stem}.pdf'
    bundle=out/'candidate_methylation_expression_figures.pdf'
    writer=PdfWriter()
    for path in [main_pdf,*[out/'plots'/f'{stem}.pdf' for stem in stems]]:writer.append(str(path))
    with bundle.open('wb') as f:writer.write(f)
    writer.close();exports.append(bundle)
    # Also expose the five figures individually at the requested analysis root.
    individual_dir=source/'methylation_expression'
    individual_exports=[]
    for path in [main_pdf,*[out/'plots'/f'{stem}.pdf' for stem in stems]]:
        target=individual_dir/path.name
        shutil.copy2(path,target)
        individual_exports.append(target)
    exports.extend(individual_exports)
    inputs_list=[source/'data/candidate_genomic_annotation.tsv',
                 source/'methylation_expression/cross_cohort_comparison/data'/('all_33_cross_cohort_CpGs.tsv' if disease=='BC' else 'all_17_cross_cohort_CpGs.tsv'),
                 source/'methylation_expression/cross_cohort_comparison/data'/('gene_level_cross_cohort_results.tsv' if disease=='BC' else 'all_CpG_gene_expression_results.tsv'),
                 *[source/'data'/f'{cohort}_{kind}' for cohort in ['original','external'] for kind in ['candidate_beta.tsv.gz','patient_annotations.tsv']]]
    manifest=dict(status='complete',disease=disease,n_CpGs=len(ann),expression_CpGs=int(effects.expression_log2FC.notna().sum()),
                  reference='PMC7263199',scope='Candidate-panel adaptations; no matched-sample correlation, pan-cancer specificity or chromatin-accessibility claims',
                  checks=checks,input_sha256={str(p):digest(p) for p in inputs_list},script_sha256=digest(Path(__file__)),outputs={os.path.relpath(p,out):digest(p) for p in exports})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'{name}: {len(exports)} publication exports -> {out}',flush=True)
    return exports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='Disease dataset accumulated_results/post_evaluation directory')
    parser.add_argument('--disease',choices=['BC','AML'],required=True)
    args=parser.parse_args();run(args.out,args.disease)
