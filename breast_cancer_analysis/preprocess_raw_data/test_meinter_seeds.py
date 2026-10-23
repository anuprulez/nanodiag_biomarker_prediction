import unittest
import numpy as np
import pandas as pd
from select_meinter_gse66695_seeds import bh_adjust,select_dms

class MeinterSeedTests(unittest.TestCase):
    def test_bh_uses_all_tests_and_preserves_input_order(self):
        np.testing.assert_allclose(bh_adjust([.04,.001,.03,.2]),[.05333333333333334,.004,.05333333333333334,.2])
    def test_strict_effect_and_significance_thresholds(self):
        frame=pd.DataFrame({'deltaBeta':[.3,-.3,.31,-.31,.4,.5],
                            'P.Value':[.001,.001,.001,.001,.01,.001],
                            'FDR':[.001,.001,.009,.009,.001,.01]})
        self.assertEqual(select_dms(frame).index.tolist(),[2,3])

if __name__=='__main__':unittest.main()
