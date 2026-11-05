"""Independent numerical checks of saved AML analysis artifacts."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


def main(base):
    base=Path(base)
    out=base/'methylation_expression/cross_cohort_comparison';data=out/'data'
    meta=pd.read_csv(data/'GSE15061_sample_metadata.tsv',sep='\t')
    assert len(meta)==271 and meta.sample_id.is_unique and meta.biological_sample_id.is_unique
    assert meta.normalization_variant.eq('DS').all()
    x=pd.read_csv(data/'GSE15061_normalized_gene_expression.tsv.gz',sep='\t',index_col=0)
    assert list(x.columns)==list(meta.sample_id)
    for model in ['center_adjusted','unadjusted']:
        d=pd.read_csv(data/f'design_{model}.tsv',sep='\t',index_col=0)
        de=pd.read_csv(data/f'all_genes_{model}_limma.tsv',sep='\t').set_index('gene')
        # OLS coefficients computed independently of R/limma, across the whole genome.
        coefficients=np.linalg.lstsq(d.to_numpy(),x.to_numpy().T,rcond=None)[0]
        expected=pd.Series(coefficients[list(d.columns).index('groupAML')],index=x.index)
        np.testing.assert_allclose(de.logFC,expected.reindex(de.index),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(de['adj.P.Val'],false_discovery_control(de['P.Value'].to_numpy(),method='bh'),atol=1e-12)
        assert (de['CI.L']<=de.logFC).all() and (de.logFC<=de['CI.R']).all()
    f=pd.read_csv(data/'all_17_cross_cohort_CpGs.tsv',sep='\t')
    fixed=set(pd.read_csv(Path(__file__).parent/'resources/fixed_AML_17_CpGs.tsv',sep='\t').CpG_ID)
    assert len(f)==17 and f.CpG_ID.is_unique and set(f.CpG_ID)==fixed
    betas=pd.read_csv(base/'data/original_candidate_beta.tsv.gz',sep='\t',index_col=0)
    m=pd.read_csv(base/'data/original_patient_annotations.tsv',sep='\t').set_index('sample_id').loc[betas.index]
    assert m.label.value_counts().to_dict()=={0:464,1:62}
    delta=betas.loc[m.label.eq(1)].mean()-betas.loc[m.label.eq(0)].mean()
    np.testing.assert_allclose(f.methylation_delta_beta,f.CpG_ID.map(delta),atol=1e-12,rtol=1e-12)
    # Independently reconstruct filtered CpG aggregates from all saved gene pairs.
    pairs=pd.read_csv(data/'all_CpG_gene_expression_results.tsv',sep='\t')
    pairs=pairs.loc[pairs.model.eq('center_adjusted')]
    selected=pairs.loc[(pairs.expression_p < 0.01)&(pairs.expression_FDR_genome < 0.01)&pairs.expression_log2FC.notna()].drop_duplicates(['CpG_ID','expression_gene'])
    expected=selected.groupby('CpG_ID').expression_log2FC.mean()
    np.testing.assert_allclose(f.expression_log2FC,f.CpG_ID.map(expected),equal_nan=True,atol=1e-12)
    assert set(f.loc[f.expression_log2FC.notna(),'CpG_ID'])==set(expected.index)
    labels=pd.read_csv(data/'displayed_CpG_gene_labels.tsv',sep='\t')
    assert len(labels)>0 and labels.CpG_ID.is_unique
    assert set(labels.CpG_ID)==set(f.loc[f.expression_log2FC.notna(),'CpG_ID'])
    assert f.expression_log2FC.notna().sum()==len(expected)
    assert pd.isna(f.set_index('CpG_ID').loc['cg07911080','expression_log2FC'])
    assert not set(meta.sample_id)&set(m.index)
    checks=dict(status='passed',genes_tested=len(x),unique_expression_samples=len(meta),frozen_CpGs=len(f),evaluable_CpGs=len(expected),labels=len(labels),
                checks=['Genome-wide limma coefficients match independent numpy OLS','Genome-wide BH FDR matches scipy','Gene coefficient confidence intervals contain point estimate',
                        'Methylation contrasts reproduce 62 vs 464 raw cached sample means','Frozen CpG identities unchanged','All plotted CpGs labeled',
                        'DS-only unique biological samples','No methylation/expression GSM overlap'])
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args().base)
