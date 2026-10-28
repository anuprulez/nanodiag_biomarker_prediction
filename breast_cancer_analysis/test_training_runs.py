"""Successful-run preservation and cumulative numeric/plot integration."""
import json
from pathlib import Path
import tempfile
import unittest
from omegaconf import OmegaConf
from training_runs import archive_training_run, completed_runs


class TrainingRunTests(unittest.TestCase):
    def test_comparison_routes_saved_and_current_artifacts(self):
        from unittest.mock import patch
        from evaluation import run_comparisons as runner
        import os
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            generated = folder/'generated'
            generated.mkdir()
            log = folder/'train.log'
            log.write_text('train')
            cfg = OmegaConf.create(dict(config_scope='dataset', p_base=str(folder)+'/',
                p_generated_files=str(generated)+'/', p_arrays=str(folder/'matrix.tsv'),
                model_type=['PNA'], model_overrides={}))
            OmegaConf.save(cfg, folder/'config.yaml')
            archive_training_run(folder, cfg, log)
            cwd = Path.cwd()
            try:
                for flag in [[], ['--run', '1'], ['--run', 'run1']]:
                    with patch.object(runner.sys, 'argv', ['run_comparisons.py', '--dry-run', *flag]), \
                         patch.object(runner, 'downstream_configs', return_value=[cfg]), \
                         patch('builtins.print') as printed:
                        runner.main()
                    command = printed.call_args_list[0].args[0]
                    expected = folder/'runs/run1/generated' if flag else generated
                    self.assertIn('--generated-dir '+str(expected), command)
                    expected_output = folder/'evaluation/run1' if flag else folder/'evaluation'
                    self.assertIn('--output '+str(expected_output)+' ', command)
            finally:
                os.chdir(cwd)
            from training_runs import comparison_run, run_name
            with self.assertRaises(FileNotFoundError):
                comparison_run(folder, 'run2')
            with self.assertRaises(ValueError):
                run_name('../run1')

    def test_archives_and_variable_epoch_aggregation(self):
        from train_gnn.plot_aggregations import plot_cumulative_runs
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder/'generated/data'
            data.mkdir(parents=True)
            log = folder/'train.log'
            log.write_text('first training log')
            cfg = OmegaConf.create(dict(p_base=str(folder)+'/', model_type=['PNA'],
                                        p_generated_files=str(folder/'generated')+'/', n_edges=10, n_epo=3, dpi=72))
            for score, history in [(.5, [1., 2., 3.]), (.9, [3.])]:
                (data/'all_metrics_PNA.json').write_text(json.dumps(dict(
                    best_val_acc=score, te_acc=score, tr_loss=history)))
                names = ['cg00000001_GENE_A', 'cg00000001_GENE_B',
                         'cg00000002_X' if score == .5 else 'cg00000003_Y']
                (data/'pred_likely_pos_bc_PNA.csv').write_text(
                    'test_gene_names\tpred_labels\n' + ''.join(f'{name}\t2\n' for name in names))
                archive_training_run(folder, cfg, log)
                log.write_text('second training log')
            self.assertEqual([p.name for p in completed_runs(folder)], ['run1', 'run2'])
            self.assertEqual((folder/'runs/run1/logs/train.log').read_text(), 'first training log')
            output = plot_cumulative_runs(cfg)
            self.assertEqual(output, folder/'accumulated_results')
            overlap = json.loads((output/'data/PNA_likely_positive_cpg_overlap.json').read_text())
            self.assertEqual(overlap['union_cpgs'], 3)
            self.assertEqual(overlap['common_to_all_runs'], 1)
            counts = pd.read_csv(output/'data/PNA_likely_positive_cpg_overlap_counts.csv')
            self.assertEqual(counts.unique_cpgs.tolist(), [2, 1])
            self.assertTrue((output/'plots/PNA_likely_positive_cpg_overlap_pie.png').is_file())
            self.assertTrue((output/'plots/PNA_likely_positive_cpg_overlap_pie.pdf').is_file())
            summary = pd.read_csv(output/'data/metrics_mean_std.csv')
            self.assertAlmostEqual(summary.iloc[0]['mean'], .7)
            curve = pd.read_csv(output/'data/PNA_tr_loss_mean_std.csv')
            self.assertEqual(curve.n_runs.tolist(), [2, 1, 1])
            self.assertEqual(curve['mean'].tolist(), [2., 2., 3.])
            self.assertTrue((output/'plots/radar_mean_metrics.pdf').is_file())
            self.assertTrue(list((output/'plots').glob('Heatmap*.pdf')))


if __name__ == '__main__':
    unittest.main()
