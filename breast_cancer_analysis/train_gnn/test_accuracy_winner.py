"""Protect validation-only selection and the one-time test boundary."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from evaluate_accuracy_winner import evaluate_candidate


class AccuracyWinnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / 'GCN'
        self.study.mkdir()
        (self.study / 'best_config.yaml').write_text('model_type: GCN\n')
        self.checkpoint = self.study / 'best_model.pt'
        self.checkpoint.write_bytes(b'frozen checkpoint')
        self.winner = dict(model='GCN', trial=0, validation_metric='accuracy',
                           validation_score=1., input_fingerprint='fixed-inputs')
        self.candidate = self.root / 'candidate.json'
        self.output = self.root / 'evaluation'
        self.write_candidate()
        self.called = []

    def write_candidate(self):
        (self.study / 'best_params.json').write_text(json.dumps(self.winner))
        self.candidate.write_text(json.dumps(dict(self.winner, study=str(self.study))))

    def predict(self, split):
        self.called.append(split)
        if split == 'test':
            self.assertTrue((self.output / 'selection_frozen_before_test.json').is_file())
        truth = np.array([0, 1])
        return np.array([7, 8]), truth, truth.copy(), np.eye(5)[truth]

    def test_freezes_before_test_and_refuses_repeat(self):
        with patch('evaluate_accuracy_winner.load_predictor', return_value=(self.predict, self.checkpoint)):
            result = evaluate_candidate(self.candidate, self.output)
            self.assertEqual(self.called, ['val', 'test'])
            self.assertEqual(result['test']['accuracy'], 1.)
            with self.assertRaisesRegex(ValueError, 'repeat test'):
                evaluate_candidate(self.candidate, self.output)
        self.assertEqual(self.called, ['val', 'test'])

    def test_validation_mismatch_never_predicts_test(self):
        self.winner['validation_score'] = .5
        self.write_candidate()
        with patch('evaluate_accuracy_winner.load_predictor', return_value=(self.predict, self.checkpoint)):
            with self.assertRaisesRegex(ValueError, 'validation differs'):
                evaluate_candidate(self.candidate, self.output)
        self.assertEqual(self.called, ['val'])
        self.assertFalse((self.output / 'selection_frozen_before_test.json').exists())

    def test_declared_nondeterministic_drift_preserves_original_selection(self):
        self.winner['validation_score'] = .999
        self.write_candidate()
        def nondeterministic(split):
            return self.predict(split)
        nondeterministic.non_deterministic = True
        with patch('evaluate_accuracy_winner.load_predictor', return_value=(nondeterministic, self.checkpoint)):
            result = evaluate_candidate(self.candidate, self.output)
        self.assertEqual(result['validation_score'], .999)
        self.assertEqual(result['validation']['accuracy'], 1.)
        self.assertFalse(result['validation_reproduction']['exact'])
        self.assertTrue(result['validation_reproduction']['estimator_declares_nondeterministic'])
        self.assertEqual(self.called, ['val', 'test'])

    def test_wrong_source_checkpoint_blocks_prediction(self):
        source = self.study / 'trial_00000/model.pt'
        source.parent.mkdir()
        source.write_bytes(b'wrong checkpoint')
        with patch('evaluate_accuracy_winner.load_predictor', return_value=(self.predict, self.checkpoint)):
            with self.assertRaisesRegex(ValueError, 'selected trial artifact'):
                evaluate_candidate(self.candidate, self.output)
        self.assertEqual(self.called, [])

    def test_cli_resolves_output_before_changing_directory(self):
        from evaluate_accuracy_winner import main
        original = Path.cwd()
        try:
            os.chdir(self.root)
            with patch('sys.argv', ['evaluate_accuracy_winner.py', '--candidate',
                                    'candidate.json', '--output', 'relative-output']), \
                 patch('evaluate_accuracy_winner.evaluate_candidate') as evaluate:
                main()
                evaluate.assert_called_once_with(self.candidate.resolve(),
                                                  self.root / 'relative-output')
        finally:
            os.chdir(original)

    def test_stale_candidate_rejected_before_loading(self):
        stale = dict(self.winner, input_fingerprint='different-inputs')
        (self.study / 'best_params.json').write_text(json.dumps(stale))
        with patch('evaluate_accuracy_winner.load_predictor') as loader:
            with self.assertRaisesRegex(ValueError, 'stale'):
                evaluate_candidate(self.candidate, self.output)
            loader.assert_not_called()


class GNNEvaluationIntegrationTests(unittest.TestCase):
    def test_saved_gnn_predictions_match_partition_ids(self):
        import torch
        from torch_geometric.data import Data
        from omegaconf import OmegaConf
        from gnn_network import GCN
        from evaluate_accuracy_winner import load_predictor
        torch.set_num_threads(1)
        config = OmegaConf.create(dict(model_type='GCN', SEED=32, gene_dim=6,
            hidden_dim=16, gnn_layers=1, dropout=.1, feature_dropout=0.,
            embedding_dropout=0., num_classes=5, batch_size=4, num_workers=0,
            hpo_device='cpu', neighbors_spread=[2], p_torch_data='/unused/data_fixture.pt'))
        source = torch.arange(12)
        edges = torch.stack((torch.cat((source, (source+1)%12)),
                             torch.cat(((source+1)%12, source))))
        data = Data(x=torch.randn(12,6), y=source%5, edge_index=edges,
            train_mask=source<6, val_mask=(source>=6)&(source<9), test_mask=source>=9)
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            graph = study/'prepared/models/data_fixture.pt'
            graph.parent.mkdir(parents=True)
            torch.save(data,graph)
            model = GCN(config,False)
            torch.save(model.state_dict(),study/'best_model.pt')
            predictor, checkpoint = load_predictor(study,config,study/'output')
            ids,truth,prediction,probabilities = predictor('val')
            self.assertEqual(set(ids),{6,7,8})
            self.assertTrue(np.array_equal(truth,data.y[ids].numpy()))
            self.assertEqual(probabilities.shape,(3,5))
            self.assertTrue(np.allclose(probabilities.sum(axis=1),1.))
            ids2,truth2,pred2,prob2 = predictor('val')
            self.assertTrue(np.array_equal(prediction,pred2))
            self.assertTrue(np.array_equal(ids,ids2))
            self.assertTrue(np.array_equal(probabilities,prob2))


if __name__ == '__main__':
    unittest.main()
