"""Accumulation rebuilds missing archived evaluations and reuses complete ones."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import accumulate_runs


class ArchivedEvaluationTests(unittest.TestCase):
    def test_missing_tables_are_built_once_from_archived_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/'data_test'
            run = base/'runs/run1'
            run.mkdir(parents=True)
            config = SimpleNamespace(p_base=str(base), model_type=['PNA'])
            current = SimpleNamespace(p_arrays='/tmp/matrix.tsv', p_sample_metadata='/tmp/metadata.tsv')
            def compare(command, check):
                self.assertTrue(check)
                self.assertIn('--tables-only', command)
                self.assertEqual(command[command.index('--generated-dir')+1], str(run/'generated'))
                output = Path(command[command.index('--output')+1])
                output.mkdir(parents=True, exist_ok=True)
                for stem in ('summary', 'criterion_sensitivity', 'likely_positive_common_unique_cpgs', 'negative_common_unique_cpgs'):
                    (output/f'{stem}_PNA_test.tsv').write_text('fixture')
            with patch('training_runs.completed_runs', return_value=[run]), \
                 patch('training_runs.comparison_run', return_value=(run, current, ['PNA'])), \
                 patch('training_runs.archived_generated_root', return_value=run/'generated'), \
                 patch('train_gnn.model_settings.apply_model_settings', return_value=current), \
                 patch.object(accumulate_runs.subprocess, 'run', side_effect=compare) as execute:
                self.assertEqual(accumulate_runs.ensure_archived_evaluations(config), 1)
                self.assertEqual(accumulate_runs.ensure_archived_evaluations(config), 0)
                self.assertEqual(execute.call_count, 1)

    def test_no_training_archives_is_an_explicit_error(self):
        with patch('training_runs.completed_runs', return_value=[]):
            with self.assertRaisesRegex(FileNotFoundError, 'no completed training archives'):
                accumulate_runs.ensure_archived_evaluations(SimpleNamespace(p_base='/tmp/missing'))


if __name__ == '__main__':
    unittest.main()
