"""Statistical checks for missingness, multiple testing and covariate control."""
import unittest
import numpy as np
from .methylation_expression import association, partial_rank, bh

class ExpressionTests(unittest.TestCase):
    def test_bh_preserves_missing_and_corrects_all_finite_tests(self):
        np.testing.assert_allclose(bh([.01,np.nan,.04,.03]),[.03,np.nan,.04,.04],equal_nan=True)

    def test_pairwise_complete_and_constant_measurements(self):
        x=np.arange(30,dtype=float);y=-x.copy();y[:5]=np.nan
        result=association(x,y)
        self.assertEqual(result['n'],25);self.assertAlmostEqual(result['rho'],-1)
        self.assertEqual(association(np.ones(30),x)['status'],'constant measurement')
        self.assertTrue(np.isnan(association(x[:10],y[:10])['p_value']))

    def test_partial_rank_removes_age_association(self):
        rng=np.random.default_rng(17);age=np.arange(1000,dtype=float)
        x=age+rng.normal(0,100,1000);y=age+rng.normal(0,100,1000)
        group=np.where(np.arange(1000)%2,'A','B')
        raw=association(x,y);adjusted=partial_rank(x,y,age,group)
        self.assertGreater(raw['rho'],.8)
        self.assertLess(abs(adjusted['rho']),.1)
        age[:10]=np.nan
        self.assertEqual(partial_rank(x,y,age,group)['n'],990)

if __name__=='__main__':unittest.main()
