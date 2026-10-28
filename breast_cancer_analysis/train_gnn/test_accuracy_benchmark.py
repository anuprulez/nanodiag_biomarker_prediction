"""Check exact trial budgets, resumability, and freezing all winners before test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optuna
from omegaconf import OmegaConf

from run_accuracy_benchmark import benchmark
from tune_models import write_json


class AccuracyBenchmarkTests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = OmegaConf.create(dict(p_generated_files=str(Path(self.temp.name)/'generated')+'/',
                                           model_type=['GCN','GraphSAGE']))
        self.search = OmegaConf.create(dict(models=['GCN','GraphSAGE'],metric='accuracy'))
        self.models = ['GCN','GraphSAGE']
        self.root = Path(self.config.p_generated_files)/'optuna/test'
        self.fail_first = False
        self.test_calls = []

    def fake_optimize(self, config, search, name, n_trials, study_tag):
        directory = self.root/name
        directory.mkdir(parents=True,exist_ok=True)
        study = optuna.create_study(study_name=name, direction='maximize',load_if_exists=True,
                                   storage='sqlite:///'+str(directory/'study.sqlite3'))
        def objective(trial):
            if self.fail_first and name=='GCN' and trial.number==0:
                raise RuntimeError('intentional trial failure')
            return .5 + .01*trial.number + (.05 if name=='GraphSAGE' else 0)
        study.optimize(objective,n_trials=n_trials)
        write_json(directory/'best_params.json',dict(model=name,trial=study.best_trial.number,
                   validation_metric='accuracy',validation_score=study.best_value,
                   input_fingerprint='same-inputs',parameters={}))
        return directory

    def fake_evaluate(self,candidate,output):
        frozen=json.loads((self.root/'validation_selection_frozen.json').read_text())
        self.assertEqual(len(frozen['candidates']),2)
        self.assertTrue(all(row['trials_finished']==3 for row in frozen['candidates']))
        row=json.loads(candidate.read_text());self.test_calls.append(row['model'])
        result=dict(row,validation=dict(accuracy=row['validation_score'],macro_f1=.5),
                    test=dict(accuracy=.45,macro_f1=.4))
        output.mkdir(parents=True)
        write_json(output/'result.json',result)
        return result

    def run_benchmark(self):
        return benchmark(self.config,self.search,self.models,3,'test')

    def test_fixed_budget_and_resume_do_not_repeat_training_or_test(self):
        with patch('run_accuracy_benchmark.validate_seed_manifest'), \
             patch('run_accuracy_benchmark.input_fingerprint',return_value='same-inputs'), \
             patch('run_accuracy_benchmark.optimize_model',side_effect=self.fake_optimize) as optimize, \
             patch('run_accuracy_benchmark.evaluate_candidate',side_effect=self.fake_evaluate) as evaluate, \
             patch('builtins.print'):
            self.assertTrue(self.run_benchmark())
            self.assertEqual(optimize.call_count,2)
            self.assertEqual(evaluate.call_count,2)
            self.assertTrue(self.run_benchmark())
            self.assertEqual(optimize.call_count,2)
            self.assertEqual(evaluate.call_count,2)
        status=json.loads((self.root/'status.json').read_text())
        self.assertEqual(status['state'],'complete')
        self.assertTrue((self.root/'accuracy_comparison.csv').is_file())

    def test_failed_trial_consumes_one_slot_and_remaining_budget_continues(self):
        self.fail_first=True
        with patch('run_accuracy_benchmark.validate_seed_manifest'), \
             patch('run_accuracy_benchmark.input_fingerprint',return_value='same-inputs'), \
             patch('run_accuracy_benchmark.optimize_model',side_effect=self.fake_optimize), \
             patch('run_accuracy_benchmark.evaluate_candidate',side_effect=self.fake_evaluate), \
             patch('builtins.print'):
            self.assertTrue(self.run_benchmark())
        rows=json.loads((self.root/'status.json').read_text())['results']
        row=next(row for row in rows if row['model']=='GCN')
        self.assertEqual((row['trials_finished'],row['trials_completed'],row['trials_failed']),(3,2,1))


if __name__=='__main__':
    unittest.main()
