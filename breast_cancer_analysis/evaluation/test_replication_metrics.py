"""Independent numerical checks for cross-study statistics and candidate rules."""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from replication_metrics import (bh_adjust, external_inference, annotate_candidates,
                                 sensitivity_table, INTERNAL, EXTERNAL)


class ReplicationTests(unittest.TestCase):
    def fixture(self):
        healthy=np.array([.1,.12,.15,.17,.18])
        cancer=np.array([.2,.23,.26,.29,.3,.33])
        frame=pd.DataFrame({'CpG_ID':['cg1','cg2','cg3'],
            'Healthy_GSE101961_mean_beta':[healthy.mean(),.2,.2],
            'Healthy_GSE101961_SD':[healthy.std(ddof=1),.1,np.nan],
            'Healthy_GSE101961_n':[len(healthy),10,1],
            'BC_GSE226569_mean_beta':[cancer.mean(),.21,.5],
            'BC_GSE226569_SD':[cancer.std(ddof=1),.1,.1],
            'BC_GSE226569_n':[len(cancer),10,10],
            EXTERNAL:[cancer.mean()-healthy.mean(),.01,.3]})
        return frame, healthy, cancer

    def test_welch_matches_sample_level_test_and_CI(self):
        frame,healthy,cancer=self.fixture()
        result=external_inference(frame)
        reference=ttest_ind(cancer,healthy,equal_var=False)
        self.assertAlmostEqual(result.External_p_value.iloc[0],reference.pvalue)
        ci=reference.confidence_interval(.95)
        self.assertAlmostEqual(result.External_CI95_low.iloc[0],ci.low)
        self.assertAlmostEqual(result.External_CI95_high.iloc[0],ci.high)
        self.assertFalse(result.External_testable.iloc[2])
        self.assertTrue(np.isnan(result.External_FDR.iloc[2]))

    def test_plots_show_only_likely_positive(self):
        import matplotlib.pyplot as plt
        from replication_metrics import validation_plots, sensitivity_plot
        frame,_,_=self.fixture()
        frame=external_inference(frame.iloc[:2].copy())
        frame[INTERNAL]=[.1,.02]
        frame=annotate_candidates(frame)
        groups=[('likely_positive','Predicted likely positive','unused','#dd8452')]
        summary=[dict(pearson_r=.8,spearman_r=.8,same_direction_fraction=1.,
                      mean_absolute_error=.02,exploratory_criterion_fraction=.5,
                      exploratory_criterion_count=1,plotted_unique_cpgs=2)]
        data={'likely_positive':frame}
        for fig in [validation_plots(data,summary,groups,'Original'),
                    sensitivity_plot(sensitivity_table(data),groups)]:
            text=' '.join(ax.get_title()+' '+ ' '.join(t.get_text() for t in ax.get_xticklabels())
                          for ax in fig.axes)
            self.assertNotIn('negative',text.lower())
            self.assertIn('likely positive',text.lower())
            plt.close(fig)

    def test_bh_full_family_and_missing(self):
        self.assertTrue(np.allclose(bh_adjust([.01,.04,.03,.2]),[.04,.0533333333,.0533333333,.2]))
        result=bh_adjust([.01,np.nan])
        self.assertAlmostEqual(result[0],.02)
        self.assertTrue(np.isnan(result[1]))
        with self.assertRaises(ValueError):bh_adjust([1.1])

    def test_duplicate_universe_and_mismatched_delta_rejected(self):
        frame,_,_=self.fixture()
        with self.assertRaises(ValueError):external_inference(pd.concat([frame,frame]))
        frame.loc[0,EXTERNAL]=.8
        with self.assertRaises(ValueError):external_inference(frame)

    def test_candidate_direction_threshold_and_missing_FDR(self):
        frame=pd.DataFrame({INTERNAL:[.2,-.2,.2,0,.2], EXTERNAL:[.1,.2,.09,0,.2],
                            'External_FDR':[.01,.01,.01,.01,np.nan]})
        result=annotate_candidates(frame)
        self.assertEqual(result.Meets_exploratory_criterion.tolist(),[True,False,False,False,False])
        self.assertEqual(result.Direction_concordant.tolist(),[True,False,True,True,True])
        sensitivity=sensitivity_table({'likely_positive':result})
        row=sensitivity.query('effect_threshold == 0.10 and FDR_threshold == 0.05').iloc[0]
        self.assertEqual(row['count'],1)
        self.assertEqual(row.denominator,5)


if __name__=='__main__':unittest.main()
