"""XAI node overrides, group routing, and highest-confidence selection."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from xai_selection import prediction_path, select_xai_targets, top_prediction_nodes


class SelectionTests(unittest.TestCase):
    def test_group_files_and_stable_probability_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            lp=Path(temp)/'lp.csv'; rn=Path(temp)/'rn.csv'
            lp.write_text('test_gene_ids\tpred_probs\n17\t0.3\n19\t0.9\n20\t0.9\n')
            rn.write_text('test_gene_ids,pred_probs\n99,0.1\n88,0.8\n')
            config=SimpleNamespace(p_pred_likely_pos=str(lp),p_pred_negatives=str(rn))
            self.assertEqual(select_xai_targets(config),[(19,None)])
            self.assertEqual(select_xai_targets(config,group='RN'),[(88,None)])
            self.assertEqual(select_xai_targets(config,top_k=2),[(19,None),(20,None)])
            self.assertEqual(prediction_path(config,'RN'),rn)

    def test_explicit_target_bypasses_exports(self):
        config=SimpleNamespace()
        self.assertEqual(select_xai_targets(config,node_id=7980,group='RN'),[(7980,None)])
        self.assertEqual(select_xai_targets(config,node_name='cg1_A'),[(None,'cg1_A')])

    def test_missing_export_is_reconstructed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rn.csv'
            config=SimpleNamespace(p_pred_negatives=str(path))
            def refresh(current):
                self.assertIs(current,config)
                path.write_text('test_gene_ids\tpred_probs\n31\t0.95\n')
            self.assertEqual(select_xai_targets(config,group='RN',refresh_predictions=refresh),[(31,None)])

    def test_invalid_or_empty_predictions_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'lp.csv'
            for contents, error in [('test_gene_ids\tpred_probs\n','no predicted'),
                                    ('test_gene_ids\tpred_probs\n1\tnan\n','invalid prediction'),
                                    ('test_gene_ids\tpred_probs\n1\t0.9\n1\t0.8\n','duplicate'),
                                    ('test_gene_ids\n1\n','require')]:
                path.write_text(contents)
                with self.assertRaisesRegex(ValueError,error):
                    top_prediction_nodes(path)
            path.write_text('test_gene_ids\tpred_probs\n1\t0.9\n')
            with self.assertRaisesRegex(ValueError,'only 1 available'):
                top_prediction_nodes(path,2)
            with self.assertRaisesRegex(ValueError,'at least 1'):
                top_prediction_nodes(path,0)


if __name__ == '__main__':
    unittest.main()
