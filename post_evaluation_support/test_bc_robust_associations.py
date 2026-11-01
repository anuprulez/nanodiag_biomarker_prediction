"""Checks for resampling alignment, covariate effects and unique target families."""
import unittest
import numpy as np
import pandas as pd
from .bc_robust_associations import bootstrap_spearman,partial_rank_model,build_union

class RobustAssociationTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_perfect_relationship_and_missingness(self):
        x=np.arange(40,dtype=float);y=-x.copy();y[:5]=np.nan
        lo,hi,n=bootstrap_spearman(x,y,repetitions=200,seed=7)
        self.assertEqual(n,200);self.assertAlmostEqual(lo,-1);self.assertAlmostEqual(hi,-1)
        self.assertEqual(bootstrap_spearman(np.ones(30),np.arange(30))[2],0)

    def test_partial_rank_controls_shared_covariates_and_uses_complete_cases(self):
        rng=np.random.default_rng(23);z=np.arange(1000,dtype=float)
        x=z+rng.normal(0,80,1000);y=z+rng.normal(0,80,1000)
        cov=pd.DataFrame({'purity':z,'copy_number':rng.normal(size=1000),'subtype':np.where(np.arange(1000)%2,'A','B')})
        r=partial_rank_model(x,y,cov)
        self.assertLess(abs(r['rho']),.1);self.assertGreater(r['same_sample_rho'],.8)
        self.assertLess(r['CI_low'],r['rho']);self.assertGreater(r['CI_high'],r['rho'])
        cov.loc[:19,'purity']=np.nan
        self.assertEqual(partial_rank_model(x,y,cov)['n'],980)

    def test_annotation_and_cis_overlap_is_tested_once(self):
        leg=pd.DataFrame([{'CpG_ID':'cg1','canonical_gene':'GENE','expression_gene':'GENE','annotated_gene':'OLD','legacy_gene_regions':'Body','legacy_promoter':False}])
        cis=pd.DataFrame([{'CpG_ID':'cg1','canonical_gene':'GENE','expression_gene':'GENE','signed_distance_to_TSS':20,'strand':'+'}])
        result=build_union(leg,cis)
        self.assertEqual(len(result),1);self.assertTrue(result.iloc[0].annotated);self.assertTrue(result.iloc[0].cis_250kb)

if __name__=='__main__':unittest.main()
