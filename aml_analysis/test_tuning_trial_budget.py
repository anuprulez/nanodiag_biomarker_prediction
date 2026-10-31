"""CLI trial budgets must override search-space defaults."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent/'train_gnn'))
import tune_models as tuning


class TrialBudgetTests(unittest.TestCase):
    def test_explicit_trials_override_ten_trial_search_default(self):
        for requested, expected in ((['--trials', '25'], 25), ([], 10)):
            with self.subTest(requested=requested):
                config = OmegaConf.create({'config_scope': 'dataset_preprocessing'})
                search = OmegaConf.create({'metric': 'accuracy', 'n_trials': 10, 'models': ['PNA']})
                with patch.object(sys, 'argv', ['tune_models.py', *requested]), \
                     patch.object(tuning.OmegaConf, 'load', return_value=search), \
                     patch.object(tuning, 'tuning_source', return_value=config), \
                     patch.object(tuning, 'preprocessing_configs', return_value=[config]), \
                     patch.object(tuning, 'validate_seed_manifest'), \
                     patch.object(tuning.os, 'chdir'), \
                     patch.object(tuning, 'write_tuned_dataset'), \
                     patch.object(tuning, 'optimize_model', return_value=Path('/tmp/study')) as optimize, \
                     contextlib.redirect_stdout(io.StringIO()):
                    tuning.main()
                self.assertEqual(optimize.call_args.args[3], expected)
                self.assertIsNone(optimize.call_args.args[4])


if __name__ == '__main__':
    unittest.main()
