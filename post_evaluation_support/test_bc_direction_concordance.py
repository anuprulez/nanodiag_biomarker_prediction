"""Checks of feature-level direction tests and promoter-specific gene handling."""
import unittest
import numpy as np
import pandas as pd
from .bc_direction_concordance import summarize

class ConcordanceTests(unittest.TestCase):
    def test_binomial_one_cpg_one_point(self):
        f=pd.DataFrame({'promoter':[True]*8,'delta':np.arange(1,9,dtype=float),
                        'paired_logFC':[-1]*7+[1],'paired_promoter_logFC':[-1]*7+[1]})
        result=summarize(f,'delta','paired_logFC','All')
        self.assertEqual(result['n_CpGs'],8)
        self.assertEqual(result['inverse_direction'],7)
        self.assertAlmostEqual(result['binomial_p_greater'],9/256)

    def test_tss_uses_promoter_gene_not_other_mapped_gene(self):
        f=pd.DataFrame({'promoter':[True,True,False],'delta':[.2,.3,.4],
                        'paired_logFC':[1,1,1],'paired_promoter_logFC':[-1,-2,np.nan]})
        self.assertEqual(summarize(f,'delta','paired_logFC','TSS')['inverse_direction'],2)
        self.assertEqual(summarize(f,'delta','paired_logFC','All')['inverse_direction'],0)

    def test_missing_and_zero_effects_excluded_from_denominator(self):
        f=pd.DataFrame({'promoter':[True]*4,'delta':[.2,0,np.nan,-.2],
                        'paired_logFC':[-1,-1,1,np.nan],'paired_promoter_logFC':[-1,-1,1,np.nan]})
        result=summarize(f,'delta','paired_logFC','All')
        self.assertEqual(result['n_CpGs'],1);self.assertEqual(result['fraction'],1)

if __name__=='__main__':unittest.main()
