"""BC within-tissue associations, confidence intervals, purity/CN sensitivity and cis search."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr,t,norm
import matplotlib.pyplot as plt
from .methylation_expression import bh
from .publication import save_publication

SEED=20261031


def bootstrap_spearman(x,y,repetitions=2000,seed=SEED):
    """Paired patient resampling, reranking each bootstrap sample (ties retained)."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
    if len(x)<20 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan,0
    rng=np.random.default_rng(seed);values=[]
    for first in range(0,repetitions,100):
        ids=rng.integers(0,len(x),size=(min(100,repetitions-first),len(x)))
        rx=rankdata(x[ids],axis=1);ry=rankdata(y[ids],axis=1)
        rx-=rx.mean(axis=1,keepdims=True);ry-=ry.mean(axis=1,keepdims=True)
        denominator=np.sqrt((rx*rx).sum(axis=1)*(ry*ry).sum(axis=1))
        with np.errstate(invalid='ignore',divide='ignore'):values.extend(((rx*ry).sum(axis=1)/denominator).tolist())
    values=np.asarray(values);values=values[np.isfinite(values)]
    if len(values)<repetitions*.9:return np.nan,np.nan,len(values)
    return *np.quantile(values,[.025,.975]),len(values)


def partial_rank_model(x,y,covariates):
    """Partial Spearman via ranked continuous variables and categorical dummies."""
    frame=pd.DataFrame({'x':x,'y':y}).join(covariates.reset_index(drop=True)).replace([np.inf,-np.inf],np.nan).dropna()
    n=len(frame);base=dict(n=n,rho=np.nan,p_value=np.nan,CI_low=np.nan,CI_high=np.nan,
                          same_sample_rho=np.nan,design_rank=np.nan,df=np.nan,status='insufficient complete data')
    if n<20:return base
    columns=[np.ones(n)]
    for c in covariates.columns:
        if pd.api.types.is_numeric_dtype(covariates[c]):columns.append(rankdata(frame[c]))
        else:columns.extend(pd.get_dummies(frame[c],drop_first=True,dtype=float).to_numpy().T)
    design=np.column_stack(columns);rank=np.linalg.matrix_rank(design);df=n-rank-1
    base.update(design_rank=int(rank),df=int(df))
    if np.ptp(frame.x)==0 or np.ptp(frame.y)==0 or df<10:return base|{'status':'constant input or insufficient degrees of freedom'}
    rx=rankdata(frame.x);ry=rankdata(frame.y)
    residual_x=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0]
    residual_y=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
    if np.std(residual_x)<1e-10 or np.std(residual_y)<1e-10:return base|{'status':'degenerate residuals'}
    rho=float(np.clip(np.corrcoef(residual_x,residual_y)[0,1],-1,1))
    p=float(2*t.sf(abs(rho)*np.sqrt(df/max(1-rho*rho,1e-15)),df))
    # Approximate partial-correlation CI, explicitly distinguished from bootstrap raw CIs.
    se=1/np.sqrt(n-rank-2);z=np.arctanh(np.clip(rho,-1+1e-12,1-1e-12));lo,hi=np.tanh(z+np.array([-1,1])*norm.ppf(.975)*se)
    return base|dict(rho=rho,p_value=p,CI_low=float(lo),CI_high=float(hi),same_sample_rho=float(spearmanr(frame.x,frame.y).statistic),status='tested')


def unadjusted(x,y,reps,key):
    valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
    base=dict(n=len(x),rho=np.nan,p_value=np.nan,CI_low=np.nan,CI_high=np.nan,bootstrap_valid=0,status='insufficient complete data')
    if len(x)<20:return base
    if np.ptp(x)==0 or np.ptp(y)==0:return base|{'status':'constant input'}
    r=spearmanr(x,y);seed=SEED+int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    low,high,count=bootstrap_spearman(x,y,reps,seed)
    return base|dict(rho=float(r.statistic),p_value=float(r.pvalue),CI_low=low,CI_high=high,bootstrap_valid=count,status='tested')


def build_union(legacy,cis):
    rows={}
    for family,table in [('annotated',legacy),('cis_250kb',cis)]:
        for row in table.to_dict('records'):
            key=(row['CpG_ID'],row['canonical_gene'])
            if key not in rows:rows[key]={'CpG_ID':key[0],'canonical_gene':key[1],
                                       'expression_gene':row['expression_gene'],'annotated':False,'cis_250kb':False}
            rows[key][family]=True
            if family=='annotated':rows[key].update(annotated_gene=row['annotated_gene'],legacy_gene_regions=row['legacy_gene_regions'],legacy_promoter=row['legacy_promoter'])
            else:rows[key].update(signed_distance_to_TSS=row['signed_distance_to_TSS'],strand=row['strand'])
    return pd.DataFrame(rows.values())


def main(base):
    base=Path(base);out=base/'robust_followup'
    legacy=pd.read_csv(out/'data/legacy_gene_mapping_audit.tsv',sep='\t')
    cis=pd.read_csv(out/'data/cis_250kb_gene_mapping.tsv',sep='\t')
    pairs=build_union(legacy,cis);pairs.to_csv(out/'data/all_distinct_tested_targets.tsv',sep='\t',index=False)
    meta=pd.read_csv(out/'data/sample_metadata_with_purity.tsv',sep='\t',index_col=0)
    methyl=pd.read_csv(base/'data/methylation_matched.tsv.gz',sep='\t',index_col=0)
    cn=pd.read_csv(out/'data/gene_copy_number.tsv.gz',sep='\t',index_col=0)
    genes=set(pairs.expression_gene.dropna());chunks=[]
    for chunk in pd.read_csv(base/'direction_concordance/data/expression_genome.tsv.gz',sep='\t',index_col=0,chunksize=3000):
        chunks.append(chunk.loc[chunk.index.isin(genes)])
    expr=pd.concat(chunks)
    assert methyl.columns.equals(meta.index) and expr.columns.equals(meta.index) and cn.columns.equals(meta.index)
    assert len(methyl)==33
    for _,m in meta.groupby('sample_type'):assert m.patient_id.is_unique
    raw=[];adjusted=[]
    for i,pair in enumerate(pairs.to_dict('records')):
        gene=pair['expression_gene'];cpg=pair['CpG_ID']
        x=methyl.loc[cpg].to_numpy(dtype=float)
        y=expr.loc[gene].to_numpy(dtype=float) if gene in expr.index else np.full(len(meta),np.nan)
        copies=cn.loc[gene].to_numpy(dtype=float) if gene in cn.index else np.full(len(meta),np.nan)
        for tissue in ['Primary tumor','Solid tissue normal']:
            mask=meta.sample_type.eq(tissue).to_numpy();reps=2000 if pair['annotated'] else 1000
            raw.append(pair|{'tissue':tissue,'bootstrap_requested':reps,'CI_method':'paired patient bootstrap percentile'}|unadjusted(x[mask],y[mask],reps,cpg+'|'+str(gene)+'|'+tissue))
        mask=meta.sample_type.eq('Primary tumor').to_numpy();m=meta.loc[mask].reset_index(drop=True)
        for model,purity,extended in [('CPE_CN','CPE',False),('ABSOLUTE_CN','ABSOLUTE',False),('CPE_CN_age_PAM50','CPE',True)]:
            cov=pd.DataFrame({'purity':m[purity].to_numpy(dtype=float),'copy_number':copies[mask]})
            if extended:cov['age']=m.age.to_numpy(dtype=float);cov['PAM50']=m.PAM50Call_RNAseq
            fit=partial_rank_model(x[mask],y[mask],cov)
            adjusted.append(pair|{'model':model,'CI_method':'approximate Fisher-z partial-rank CI'}|fit)
        if (i+1)%40==0:print(f'Analyzed {i+1}/{len(pairs)} distinct CpG–gene targets',flush=True)
    raw=pd.DataFrame(raw);adjusted=pd.DataFrame(adjusted)
    raw['FDR']=np.nan
    for tissue in raw.tissue.unique():
        mask=raw.tissue.eq(tissue);raw.loc[mask,'FDR']=bh(raw.loc[mask,'p_value'])
    raw['FDR_all_tissues']=bh(raw.p_value)
    adjusted['FDR']=np.nan
    for model in adjusted.model.unique():
        mask=adjusted.model.eq(model);adjusted.loc[mask,'FDR']=bh(adjusted.loc[mask,'p_value'])
    adjusted['FDR_all_adjustment_models']=bh(adjusted.p_value)
    raw.to_csv(out/'data/within_tissue_correlations_with_CI.tsv',sep='\t',index=False)
    adjusted.to_csv(out/'data/tumor_purity_copy_number_adjusted.tsv',sep='\t',index=False)
    # Keep group-wise differential expression separate from sample-level associations.
    for model in ['paired','blocked']:
        de=pd.read_csv(base/'direction_concordance/data'/f'all_genes_{model}_limma.tsv',sep='\t').set_index('gene')
        target_de=pairs.merge(de.reset_index(),left_on='expression_gene',right_on='gene',how='left',validate='many_to_one')
        target_de.to_csv(out/'data'/f'separate_tumor_vs_normal_expression_{model}.tsv',sep='\t',index=False)
    core=raw.loc[raw.annotated & raw.tissue.eq('Primary tumor') & raw.status.eq('tested')]
    primary=adjusted.loc[adjusted.model.eq('CPE_CN')]
    priority=primary.loc[primary.status.eq('tested')].sort_values(['FDR','p_value'])
    priority.to_csv(out/'data/priority_associations_after_adjustment.tsv',sep='\t',index=False)
    coverage=[]
    for cpg in methyl.index:
        for family,flag in [('annotated','annotated'),('cis_250kb','cis_250kb')]:
            u=raw.loc[raw.CpG_ID.eq(cpg)&raw[flag]];a=primary.loc[primary.CpG_ID.eq(cpg)&primary[flag]]
            coverage.append(dict(CpG_ID=cpg,family=family,mapped_target_genes=int(pairs.loc[pairs.CpG_ID.eq(cpg)&pairs[flag]].shape[0]),
                                 tumor_tested=int(u.loc[u.tissue.eq('Primary tumor'),'p_value'].notna().sum()),
                                 normal_tested=int(u.loc[u.tissue.eq('Solid tissue normal'),'p_value'].notna().sum()),
                                 adjusted_tested=int(a.p_value.notna().sum()),adjusted_significant=int(a.FDR.lt(.05).sum())))
    coverage=pd.DataFrame(coverage);coverage.to_csv(out/'data/all_33_CpG_coverage.tsv',sep='\t',index=False)
    figures(out,raw,adjusted)
    report(base,out,pairs,raw,adjusted,coverage,legacy)
    sources=[base/'data/methylation_matched.tsv.gz',base/'direction_concordance/data/expression_genome.tsv.gz',
             out/'data/sample_metadata_with_purity.tsv',out/'data/gene_copy_number.tsv.gz',out/'data/all_distinct_tested_targets.tsv',
             out/'source/refGene.hg19.txt.gz',out/'source/HGNC_complete_set.tsv',out/'source/Tumor.purity.rda']
    manifest=dict(no_pooled_correlations=True,minimum_complete_n=20,random_seed=SEED,window_bp=250000,
                  raw_method='Within-tissue Spearman; annotated CIs: 2000 paired-patient bootstrap resamples, cis-only CIs: 1000; 95% percentile intervals; raw p-values asymptotic',
                  adjustment='Partial rank correlation: CPE+gene CN primary; ABSOLUTE+CN and CPE+CN+age+PAM50 sensitivity. Continuous covariates ranked; PAM50 dummy encoded. Complete cases, no imputation.',
                  adjusted_CI='Approximate Fisher-z intervals with effective design rank; not bootstrap intervals',
                  FDR='BH across the distinct union of all annotated and cis-search pairs, separately by tissue/raw model and each adjusted model; additional joint-tissue/model FDR columns supplied',
                  purity='Published Aran et al purity estimates distributed in TCGAbiolinks; exact vial preferred, otherwise unique same-sample-type match; normal purity left missing',
                  coordinates='hg19; GPL13534 CpG coordinates converted from 1-based; refGene transcript TSS +/-250 kb; HGNC approved/previous-symbol mappings, no heuristic nearest-gene substitution',
                  limitations=['No new independent GEO validation cohort','No direct enhancer contacts or tissue-specific enhancer annotation','No full immune/stromal cell-type deconvolution','Technical batch not adjusted','CPE derives partly from expression and methylation; ABSOLUTE sensitivity supplied','Copy number and purity adjustment do not establish causation'],
                  input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary=primary.loc[primary.status.eq('tested')]
    print(json.dumps({'distinct_targets':len(pairs),'primary_tested':len(summary),'primary_significant':int(summary.FDR.lt(.05).sum()),'annotated_primary_significant':int(summary.loc[summary.annotated,'FDR'].lt(.05).sum()),'normal_tested':int(raw.loc[raw.tissue.eq('Solid tissue normal'),'p_value'].notna().sum())},indent=2))
    print(summary.sort_values('p_value')[['CpG_ID','expression_gene','annotated','cis_250kb','n','rho','CI_low','CI_high','FDR']].head(15).to_string(index=False))


def figures(out,raw,adjusted):
    style={'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'}
    core=raw.loc[raw.annotated & raw.status.eq('tested')].copy()
    keys=core.loc[core.tissue.eq('Primary tumor')].sort_values('rho')[['CpG_ID','canonical_gene','expression_gene']]
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(13,11),sharey=True)
        for ax,tissue,color in zip(axes,['Primary tumor','Solid tissue normal'],['#0072B2','#D55E00']):
            indexed=core.loc[core.tissue.eq(tissue)].set_index(['CpG_ID','canonical_gene'])
            for i,key in enumerate(keys.itertuples()):
                if (key.CpG_ID,key.canonical_gene) not in indexed.index:continue
                r=indexed.loc[(key.CpG_ID,key.canonical_gene)]
                ax.plot([r.CI_low,r.CI_high],[i,i],color=color,lw=1.3)
                ax.plot(r.rho,i,'o',color=color,ms=4,markerfacecolor=color if r.FDR<.05 else 'white')
            ax.axvline(0,color='#999999',lw=1);ax.set_xlim(-1,1);ax.set_xlabel('Spearman ρ with 95% bootstrap CI')
            ax.set_title(tissue,fontweight='bold');ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        axes[0].set_yticks(range(len(keys)),[f'{r.CpG_ID} · {r.expression_gene}' for r in keys.itertuples()]);axes[0].invert_yaxis()
        fig.suptitle('BC | Within-tissue methylation–expression associations',fontsize=15,fontweight='bold',y=.98)
        fig.text(.03,.02,'Annotated targets; 2,000 paired-patient bootstrap resamples. Filled markers: BH FDR < 0.05 across all annotated + cis targets.\nNormal-tissue intervals reflect smaller sample sizes. No pooled tumor/normal correlation is used.',fontsize=9)
        fig.subplots_adjust(left=.23,right=.98,top=.91,bottom=.11,wspace=.16)
    save_publication(fig,out,'within_tissue_correlations_with_CI')
    primary=adjusted.loc[adjusted.model.eq('CPE_CN')&adjusted.annotated&adjusted.status.eq('tested')].sort_values('rho')
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(10,11))
        for i,r in enumerate(primary.itertuples()):
            ax.plot([r.same_sample_rho,r.rho],[i,i],color='#bbbbbb',lw=1)
            ax.plot(r.same_sample_rho,i,'o',color='#999999',ms=4)
            ax.plot([r.CI_low,r.CI_high],[i,i],color='#009E73',lw=1.1)
            ax.plot(r.rho,i,'D',color='#009E73',ms=5,markerfacecolor='#009E73' if r.FDR<.05 else 'white')
            ax.text(1.02,i,f'{r.n} | {r.FDR:.2g}',va='center',fontsize=8)
        ax.set_yticks(range(len(primary)),[f'{r.CpG_ID} · {r.expression_gene}' for r in primary.itertuples()]);ax.invert_yaxis()
        ax.set_xlim(-1,1.35);ax.set_xticks([-1,-.5,0,.5,1]);ax.axvline(0,color='#999999',lw=1)
        ax.set_xlabel('Rank correlation within primary tumors');ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.text(1.02,1.015,'n | adjusted FDR',transform=ax.get_xaxis_transform(),fontsize=9)
        fig.suptitle('BC | Purity and gene copy-number adjustment',fontsize=15,fontweight='bold',y=.98)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#999999',label='Unadjusted, same patients'),Line2D([],[],marker='D',ls='',color='#009E73',label='CPE + gene CN adjusted')],loc='upper center',bbox_to_anchor=(.57,.955),ncol=2,frameon=False)
        fig.text(.03,.02,'Green intervals: approximate Fisher-z 95% CIs for partial rank correlations. Filled: adjusted BH FDR < 0.05.\nFDR includes every testable annotated + cis target. Purity/CN adjustment does not prove cis regulation.',fontsize=9)
        fig.subplots_adjust(left=.32,right=.96,top=.89,bottom=.11)
    save_publication(fig,out,'purity_copy_number_adjustment')
    top=adjusted.loc[adjusted.model.eq('CPE_CN')&adjusted.cis_250kb&adjusted.status.eq('tested')].sort_values('p_value').head(15).iloc[::-1]
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(10,7))
        for i,r in enumerate(top.itertuples()):
            ax.plot([r.CI_low,r.CI_high],[i,i],color='#0072B2',lw=1.2)
            ax.plot(r.rho,i,'o',color='#0072B2',markerfacecolor='#0072B2' if r.FDR<.05 else 'white')
        ax.set_yticks(range(len(top)),[f'{r.CpG_ID} · {r.expression_gene}' for r in top.itertuples()])
        ax.axvline(0,color='#999999',lw=1);ax.set_xlim(-1,1);ax.set_xlabel('CPE / target-gene CN adjusted rank correlation')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.set_title('BC | Exploratory cis targets within ±250 kb of a TSS',fontweight='bold')
        fig.text(.03,.02,'Top 15 tested associations by adjusted-model raw p-value. All targets enter BH correction; estimates use tumors only.\nIntervals: approximate Fisher-z 95% CIs. Proximity and adjusted association do not establish a regulatory target.',fontsize=9)
        fig.tight_layout(rect=[0,.1,1,1])
    save_publication(fig,out,'cis_250kb_adjusted_candidates')


def report(base,out,pairs,raw,adjusted,coverage,legacy):
    primary=adjusted.loc[adjusted.model.eq('CPE_CN')]
    lines=['# BC: robust methylation–expression follow-up','',
           'This analysis extends the fixed 33-CpG follow-up. It does not use pooled tumor/normal correlations. Group-wise expression differences remain a separate limma result.','',
           '## Methods and safeguards','',
           '1. Within-tissue correlations: Spearman on methylation beta values and the existing log2(RSEM normalized count+1) expression. Primary tumors and solid-tissue normals are analyzed separately; sample IDs are matched exactly and each tissue has at most one sample per patient. At least 20 complete pairs and nonconstant measurements are required. No missing values are imputed.','',
           '2. Confidence intervals: raw annotated-pair correlations receive 2,000 paired-patient bootstrap resamples; cis-only pairs receive 1,000. Each replicate resamples methylation/expression together and reranks both, preserving ties. Reported 95% percentile intervals are pointwise, not simultaneous; constant bootstrap samples are dropped and intervals require at least 90% valid replicates. Small-normal-group uncertainty is visible in the intervals. Raw p-values use the Spearman large-sample approximation and should be interpreted cautiously for smaller subsets.','',
           '3. Purity and copy number: the primary within-tumor partial-rank model controls published consensus purity estimates (CPE) and continuous target-gene GISTIC2 copy number. Continuous variables are ranked; ranks of methylation and expression are residualized against an intercept and ranked covariates. Two-sided t-based p-values use the effective design rank. Adjusted 95% intervals use an approximate Fisher-z formula, not bootstrap. The raw correlation on exactly the same complete-case patients is supplied to distinguish missingness from adjustment.','',
           '4. Sensitivity: ABSOLUTE purity plus gene CN is tested separately, because CPE is partly derived from expression and methylation. A third model also controls age and categorical PAM50 subtype. Purity values match the methylation sample/vial exactly when possible, otherwise a unique same-sample-type identifier; no patient-only cross-tissue matching is used. Normals are not assigned a fabricated tumor purity. Published purity is a cell-composition proxy, not full immune/stromal deconvolution. No technical-batch adjustment or enhancer-contact analysis is claimed.','',
           '5. Mapping: legacy annotation is audited against UCSC hg19 refGene transcript coordinates. MAPINFO is treated as 1-based and refGene as 0-based half-open. Signed nearest-TSS distance is strand-aware. Every gene with at least one TSS within ±250 kb of a candidate CpG is enumerated. HGNC approved and unambiguous previous symbols map to expression rows; ambiguous/missing expression mappings remain explicit. Distance-based candidates are not automatically regulatory targets, and absence from the window does not invalidate a distal or gene-body relationship. No 1-Mb search or enhancer-specific target assignment was performed.','',
           '6. Multiple testing: duplicate annotated/cis CpG–gene pairs are tested only once. BH includes every finite-p pair in the combined annotated+cis target universe, separately for each tissue and each adjustment model. Additional joint-tissue and joint-adjustment FDR columns are supplied. Thus an annotated hit does not escape correction for the expanded target search. Significant means model-wide FDR < 0.05.','',
           '7. Independence: TCGA is separate from the GSE66695 discovery and the external GEO methylation cohorts used for this panel. This remains exploratory candidate evaluation, since the TCGA data were already inspected during prior follow-up. No new untouched matched GEO validation cohort was analyzed, and independence from every historical seed source is not established.','',
           '## Coverage and counts','',f'Distinct target hypotheses: {len(pairs)}. Mapped ±250-kb gene hypotheses: {int(pairs.cis_250kb.sum())}; legacy annotated hypotheses: {int(pairs.annotated.sum())}. All 33 CpGs remain in the coverage audit, including sites with no usable methylation.','',
           '| Analysis | Testable pairs | FDR < 0.05 |','|---|---:|---:|']
    for tissue in raw.tissue.unique():
        r=raw.loc[raw.tissue.eq(tissue)];lines.append(f'| Raw Spearman: {tissue} | {r.p_value.notna().sum()} | {r.FDR.lt(.05).sum()} |')
    for model in adjusted.model.unique():
        r=adjusted.loc[adjusted.model.eq(model)];lines.append(f'| Tumor partial rank: {model} | {r.p_value.notna().sum()} | {r.FDR.lt(.05).sum()} |')
    lines+=['','All counts above include both annotated and cis-search targets. They are not numbers of independent CpGs or independent regulatory mechanisms. Significance in tumors and nonsignificance in normals is not a test that their correlations differ.','',
            '## Annotated targets after CPE / gene-CN adjustment','',
            '| CpG | Gene | n | Same-sample raw ρ | Adjusted ρ | Approximate 95% CI | FDR across all targets |','|---|---|---:|---:|---:|---|---:|']
    core=primary.loc[primary.annotated&primary.status.eq('tested')].sort_values('p_value')
    for r in core.itertuples():lines.append(f'| {r.CpG_ID} | {r.expression_gene} | {r.n} | {r.same_sample_rho:.3f} | {r.rho:.3f} | [{r.CI_low:.3f}, {r.CI_high:.3f}] | {r.FDR:.3g} |')
    lines+=['','## Leading exploratory cis-search associations','',
            '| CpG | Gene | Signed nearest-TSS distance (bp) | n | Adjusted ρ | FDR |','|---|---|---:|---:|---:|---:|']
    top=primary.loc[primary.cis_250kb&primary.status.eq('tested')].sort_values('p_value').head(15)
    for r in top.itertuples():lines.append(f'| {r.CpG_ID} | {r.expression_gene} | {int(r.signed_distance_to_TSS)} | {r.n} | {r.rho:.3f} | {r.FDR:.3g} |')
    lines+=['','## What changed in candidate prioritization','',
            'GDF5 (cg14947478) remains the strongest negative annotated association after CPE/CN adjustment: ρ = −0.617, approximate 95% CI [−0.659, −0.571], n = 759. It remains negative after ABSOLUTE/CN adjustment and after adding age/PAM50. The current hg19 nearest transcript TSS is 46 bases upstream of this CpG; legacy annotation lists first exon / 5′UTR.','',
            'TTYH1 (cg02623991) is a stronger candidate for the promoter inverse-direction hypothesis: CPE/CN-adjusted ρ = −0.386, approximate CI [−0.445, −0.324], n = 759, FDR = 7.62 × 10⁻²⁷. The CpG is 207 bases upstream of the nearest hg19 TSS. After additionally controlling age/PAM50, the effect is smaller (ρ = −0.145) but remains significant (FDR = 0.0067).','',
            'The prior four highlighted tumor–normal sign-concordant sites are not equivalent to four cis-regulatory links. RUNX1 has essentially no CPE/CN-adjusted within-tumor association (ρ = 0.002, FDR = 0.955). CA3 lacks usable methylation measurements, so its within-tumor relationship cannot be tested. TCHH has a positive adjusted within-tumor association (ρ = 0.242), despite opposite tumor–normal changes; current refGene places it 1,307 bases downstream of its nearest TSS and within the gene body, unlike its legacy TSS200 label. It should not be described as direct evidence for promoter methylation silencing.','',
            'DOCK2 illustrates sensitivity to confounding control: its same-sample raw correlation is 0.247, falling to 0.044 after CPE/CN adjustment (FDR = 0.327). This attenuation suggests the original association is not robust to these covariates, without establishing which factor causes the change. Positive adjusted associations such as PCBP3 and CADM3 remain associations and should not be forced into an inverse-methylation model.','',
            'The wider search nominates alternative nearby targets, such as SERPINB5 for cg00005215, NRTN for cg21570818 and PAAF1 for cg25961292. These require independent validation; gene proximity plus adjusted association is not proof of a target interaction.','']
    lines+=['','## Group-wise expression changes are separate','',
            'The previous paired limma model compares 77 tumor/normal patient pairs, using all expressed genes for robust mean–variance moderation and genome-wide BH. The larger 775/84 model uses patient blocking. These tumor–normal logFCs and p-values are copied into separate_tumor_vs_normal_expression_paired.tsv and separate_tumor_vs_normal_expression_blocked.tsv for every target. They are not pooled sample-level methylation–expression correlations. Low-expression-filtered genes retain missing DE results rather than fabricated p-values.','',
            'Opposite Δβ/logFC signs support a promoter-regulation hypothesis but are distinct from within-tissue correlations. A gene-body association can have either sign. Neither association nor adjustment establishes methylation-driven expression: copy number, transcription factors, cell populations, batch and other factors can contribute.','',
            '## Artifacts','',
            '• data/within_tissue_correlations_with_CI.tsv: all raw within-tissue estimates, bootstrap intervals, sample sizes, raw p-values and expanded-universe FDR.','• data/tumor_purity_copy_number_adjusted.tsv: all three adjusted models, same-sample raw effects and approximate adjusted CIs.','• data/cis_250kb_gene_mapping.tsv and legacy_gene_mapping_audit.tsv: coordinates, target mappings, TSS distance and gene-body overlap.','• data/all_33_CpG_coverage.tsv: coverage of every candidate.','• data/separate_tumor_vs_normal_expression_*.tsv: separate group-wise limma results.','• data/sample_metadata_with_purity.tsv, gene_copy_number.tsv.gz and copy_number_gene_mapping.tsv: adjustment inputs and coverage.','• plots/: PDF, SVG and 300-dpi PNG within-tissue CI plots, purity/CN sensitivity and exploratory cis targets.','• manifest.json and source/: source metadata, query caches, input hashes and stated limitations.','',
            'Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC4671203/ ; https://github.com/BioinformaticsFMRP/TCGAbiolinks/blob/master/data/Tumor.purity.rda ; https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/ ; https://www.genenames.org/download/archive/ ; https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_data_by_genes.json']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True,type=Path);main(parser.parse_args().base)
