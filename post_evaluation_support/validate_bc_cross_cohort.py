"""Validate original BC delta beta and patient-blocked genome-wide expression results."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


def main(out):
    out=Path(out);base=out/'methylation_expression';data=base/'direction_concordance/data'
    meta=pd.read_csv(base/'data/sample_metadata.tsv',sep='\t',index_col=0)
    y=pd.read_csv(data/'expression_genome.tsv.gz',sep='\t',index_col=0)
    assert list(y.columns)==list(meta.index)
    assert not meta.duplicated(['patient_id','sample_type']).any()
    normal=meta.sample_type.eq('Solid tissue normal').to_numpy();n=len(meta)
    blocked=pd.read_csv(data/'all_genes_blocked_limma.tsv',sep='\t').set_index('gene')
    methods=dict(line.split(' ',1) for line in (data/'limma_methods.txt').read_text().splitlines())
    correlation=float(methods['full_consensus_patient_correlation'])
    # Independent GLS: invert patient covariance blocks rather than call limma.
    design=np.column_stack([np.ones(n),(~normal).astype(float)])
    precision=np.eye(n)
    for _,ids in meta.reset_index().groupby('patient_id').groups.items():
        ix=np.array(list(ids));k=len(ix)
        covariance=np.full((k,k),correlation);np.fill_diagonal(covariance,1)
        precision[np.ix_(ix,ix)]=np.linalg.inv(covariance)
    expected=np.linalg.solve(design.T@precision@design,design.T@precision@y.loc[blocked.index].to_numpy().T)[1]
    np.testing.assert_allclose(blocked.logFC,expected,atol=1e-8,rtol=1e-8)
    for model in ['paired','blocked']:
        de=pd.read_csv(data/f'all_genes_{model}_limma.tsv',sep='\t')
        np.testing.assert_allclose(de['adj.P.Val'],false_discovery_control(de['P.Value'].to_numpy()),atol=1e-12)
        assert (de['CI.L']<=de.logFC).all() and (de.logFC<=de['CI.R']).all()
    plotted=pd.read_csv(base/'cross_cohort_comparison/data/all_33_cross_cohort_CpGs.tsv',sep='\t')
    frozen=pd.read_csv(Path(__file__).parent/'resources/fixed_BC_33_CpGs.tsv',sep='\t')
    assert len(plotted)==33 and set(plotted.CpG_ID)==set(frozen.CpG_ID)
    betas=pd.read_csv(out/'data/original_candidate_beta.tsv.gz',sep='\t',index_col=0)
    ometa=pd.read_csv(out/'data/original_patient_annotations.tsv',sep='\t').set_index('sample_id').loc[betas.index]
    delta=betas.loc[ometa.label.eq(1)].mean()-betas.loc[ometa.label.eq(0)].mean()
    np.testing.assert_allclose(plotted.GSE66695_delta_beta,plotted.CpG_ID.map(delta),atol=1e-12)
    # Independently reconstruct filtered CpG aggregates from all saved gene pairs.
    pairs=pd.read_csv(base/'cross_cohort_comparison/data'/'gene_level_cross_cohort_results.tsv',sep='\t')
    selected=pairs.loc[(pairs.expression_p < 0.01)&(pairs.expression_FDR_genome < 0.01)&pairs.logFC.notna()].drop_duplicates(['CpG_ID','expression_gene'])
    expected=selected.groupby('CpG_ID').logFC.mean()
    np.testing.assert_allclose(plotted.TCGA_BRCA_log2FC,plotted.CpG_ID.map(expected),equal_nan=True,atol=1e-12)
    assert set(plotted.loc[plotted.TCGA_BRCA_log2FC.notna(),'CpG_ID'])==set(expected.index)
    labels=pd.read_csv(base/'cross_cohort_comparison/data/displayed_CpG_gene_labels.tsv',sep='\t')
    assert labels.CpG_ID.is_unique
    assert set(labels.CpG_ID)==set(plotted.loc[plotted.TCGA_BRCA_log2FC.notna(),'CpG_ID'])
    result=dict(status='passed',fixed_CpGs=33,labels=len(labels),expression_samples=n,blocked_genes=len(blocked),
                checks=['Genome-wide patient-blocked GLS matches independent numpy calculation','Genome-wide BH FDR matches scipy',
                        'Coefficient intervals contain point estimates','Frozen CpGs and original sample-level methylation contrasts match','All plotted candidates labeled'])
    (base/'cross_cohort_comparison/validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);main(parser.parse_args().base)
