"""Exploratory BC priorities combining within-tumor robustness and coordinate audit."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .publication import save_publication


def render(base):
    base=Path(base);out=base/'robust_followup';d=out/'data'
    f=pd.read_csv(base/'direction_concordance/data/all_33_CpG_concordance.tsv',sep='\t')
    a=pd.read_csv(d/'tumor_purity_copy_number_adjusted.tsv',sep='\t')
    audit=pd.read_csv(d/'legacy_gene_mapping_audit.tsv',sep='\t')
    de=pd.read_csv(base/'direction_concordance/data/panel_gene_blocked_DE.tsv',sep='\t').set_index('gene')
    candidates=a.loc[a.model.eq('CPE_CN')&a.annotated].merge(
        audit[['CpG_ID','canonical_gene','nearest_TSS_distance']],on=['CpG_ID','canonical_gene'],validate='one_to_one')
    candidates=candidates.merge(f[['CpG_ID','original_delta_beta']],on='CpG_ID',validate='many_to_one')
    candidates['expression_log2FC']=candidates.expression_gene.map(de.logFC)
    candidates['expression_FDR']=candidates.expression_gene.map(de['adj.P.Val'])
    for model in ['ABSOLUTE_CN','CPE_CN_age_PAM50']:
        sensitivity=a.loc[a.model.eq(model),['CpG_ID','canonical_gene','rho','FDR']].rename(columns={'rho':model+'_rho','FDR':model+'_FDR'})
        candidates=candidates.merge(sensitivity,on=['CpG_ID','canonical_gene'],validate='one_to_one')
    keep=(candidates.nearest_TSS_distance.between(-1500,200)&candidates.rho.lt(0)&candidates.FDR.lt(.05)
          & candidates.ABSOLUTE_CN_rho.lt(0)&candidates.ABSOLUTE_CN_FDR.lt(.05)
          & candidates.CPE_CN_age_PAM50_rho.lt(0)&candidates.CPE_CN_age_PAM50_FDR.lt(.05)
          & (candidates.original_delta_beta*candidates.expression_log2FC).lt(0)&candidates.expression_FDR.lt(.05))
    selected=candidates.loc[keep].sort_values('p_value')
    assert not selected.CpG_ID.duplicated().any(),'Define a target rule before plotting multiple gene priorities per CpG'
    selected.to_csv(d/'updated_functional_followup_priorities.tsv',sep='\t',index=False)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(9,7));other=f.loc[~f.CpG_ID.isin(selected.CpG_ID)]
        ax.scatter(other.original_delta_beta,other.blocked_logFC,s=45,color='#bdbdbd',edgecolor='white',linewidth=.5,alpha=.65,label='Other candidates')
        ax.scatter(selected.original_delta_beta,selected.expression_log2FC,s=90,color='#0072B2',edgecolor='white',linewidth=.6,label='Prioritized for functional follow-up')
        offsets={'GDF5':(-105,8),'HOXA4':(-100,6),'TTYH1':(8,-4)}
        for row in selected.itertuples():
            ax.annotate(f'{row.expression_gene}\n{row.CpG_ID}',(row.original_delta_beta,row.expression_log2FC),
                        xytext=offsets.get(row.expression_gene,(8,8)),textcoords='offset points',fontsize=10)
        ax.axvline(0,color='#aaaaaa',lw=1,zorder=0);ax.axhline(0,color='#aaaaaa',lw=1,zorder=0)
        ax.set(xlabel='Methylation Δβ (tumor − normal)',ylabel='Expression log₂FC (tumor / normal)',title='Breast cancer | Updated follow-up priorities')
        ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper right',frameon=False,fontsize=9)
        fig.text(.035,.015,'Highlight: inverse group changes plus negative within-tumor association robust to purity, copy number and age/PAM50;\nTSS proximity checked in hg19 (−1,500 to +200 bp, any transcript). Exploratory priorities, not validated regulatory effects.',fontsize=8.5)
        fig.tight_layout(rect=[0,.08,1,1])
    save_publication(fig,out,'updated_priority_delta_beta_log2FC')
    print(selected[['CpG_ID','expression_gene','nearest_TSS_distance','rho','CI_low','CI_high','FDR']].to_string(index=False))
    note='''\n## Updated exploratory follow-up priorities\n\nThe updated Δβ/log₂FC figure highlights GDF5 (cg14947478), TTYH1 (cg02623991) and HOXA4 (cg07317062). The operational display rule requires: nearest hg19 transcript TSS distance −1,500 to +200 bp; negative tumor association at FDR < 0.05 under CPE/CN, ABSOLUTE/CN and CPE/CN/age/PAM50 models; opposite original Δβ and expression log₂FC signs; and differential expression at genome-wide FDR < 0.05. This is exploratory prioritization, not a prespecified validation test. The hg19 TSS distances are +46, −207 and −12 bp, respectively. HOXA4 and GDF5 have legacy body/first-exon annotations but overlap TSS-proximal contexts in the current transcript audit. Gene-level expression does not identify which transcript is regulated. Exact estimates and rule inputs are in data/updated_functional_followup_priorities.tsv.\n\nThe four sites from the earlier sign-only figure should not be described as four confirmed functional CpGs. The new priority plot combines sign concordance with within-tumor evidence and coordinate checks, retains all 33 candidates in the background and contains no arrows.\n'''
    report=out/'REPORT.md'
    text=report.read_text()
    if '\n## Updated exploratory follow-up priorities' in text:text=text.split('\n## Updated exploratory follow-up priorities')[0]
    report.write_text(text+note)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True,type=Path);render(parser.parse_args().base)
