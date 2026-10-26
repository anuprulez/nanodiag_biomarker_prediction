"""Stage reset boundaries: keep upstream inputs and unrelated model artifacts."""
from pathlib import Path
import tempfile
import unittest
from omegaconf import OmegaConf
from stage_reset import reset_stage_outputs


class StageResetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'dataset'
        self.folder.mkdir()
        self.config = OmegaConf.create({
            'p_base': str(self.folder) + '/', 'p_base_files': '${p_base}preprocessed/',
            'p_generated_files': '${p_base}generated/', 'artifact_layout': 'split_v1',
            'model_type': ['PNA', 'GCN'], 'model_overrides': {},
            'p_torch_model': '${p_generated_files}model_${model_type}.pt',
            'p_pred_likely_pos': '${p_generated_files}likely_${model_type}.csv',
        })

    def make(self, name):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('keep or reset')
        return path

    def test_training_discards_selected_models_and_errors_but_keeps_studies(self):
        removed = [self.make(name) for name in (
            'generated/models/model_PNA.pt', 'generated/data/likely_PNA.csv',
            'generated/data/training_error_PNA.log', 'generated/data/all_metrics_PNA.json',
            'generated/plots/Model_loss_1000_links_100_epochs_PNA.pdf',
            'generated/plots/Heatmap_avg_edges_1000_epochs_100.pdf')]
        kept = [self.make(name) for name in (
            'generated/models/model_GCN.pt', 'generated/data/all_metrics_GCN.json',
            'generated/optuna/PNA/study.sqlite3', 'generated/optuna/PNA/trial_00000/model.pt',
            'preprocessed/features.csv', 'config.yaml')]
        reset_stage_outputs(self.folder, 'train', self.config, ['PNA'])
        self.assertTrue(all(not path.exists() for path in removed))
        self.assertTrue(all(path.exists() for path in kept))

    def test_xai_removes_old_explanations_and_keeps_training_inputs(self):
        removed = [self.make(name) for name in (
            'generated/data/df_xai_node_links_first_123.csv',
            'generated/plots/Explanation_local_subgraph_node_123_PNA.pdf',
            'generated/plots/Explaination_global_subgraph_node_123_PNA.pdf',
            'generated/plots/Feature_importance_violin_node_123_PNA.pdf')]
        kept = [self.make(name) for name in (
            'generated/models/model_PNA.pt', 'generated/models/data_PNA.pt',
            'generated/data/likely_PNA.csv', 'generated/plots/Model_loss_PNA.pdf')]
        reset_stage_outputs(self.folder, 'xai', self.config)
        self.assertTrue(all(not path.exists() for path in removed))
        self.assertTrue(all(path.exists() for path in kept))

    def test_reset_rejects_outside_output_and_keeps_raw_inputs(self):
        outside = self.root / 'raw'
        outside.mkdir(); marker = outside / 'input.csv'; marker.write_text('raw')
        self.config.p_base_files = str(outside)
        with self.assertRaisesRegex(ValueError, 'outside dataset'):
            reset_stage_outputs(self.folder, 'preprocess', self.config)
        self.assertEqual(marker.read_text(), 'raw')

    def test_training_rejects_untuned_model_before_removing_files(self):
        marker = self.make('generated/models/model_PNA.pt')
        with self.assertRaisesRegex(ValueError, 'no tuned parameters'):
            reset_stage_outputs(self.folder, 'train', self.config, ['Missing'])
        self.assertTrue(marker.exists())


if __name__ == '__main__':
    unittest.main()
