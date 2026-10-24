"""Check BC input compatibility and provenance across the parameter workflow."""
import tempfile
from pathlib import Path
import unittest
from omegaconf import OmegaConf
from preprocessing_grid import preprocessing_configs
from seed_inputs import seed_signature, write_seed_manifest, validate_seed_manifest

class BreastCancerWorkflowTests(unittest.TestCase):
    def test_grid_isolates_outputs_and_shares_bc_inputs(self):
        root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(root / 'config/config.yaml')
        config.corr_threshold = [0.2, 0.33]
        config.knn_neighbours = [500, 3000]
        config.size_negative = [5000, 10000]
        runs = preprocessing_configs(config)
        self.assertEqual(len(runs), 8)
        self.assertEqual(len({c.p_base_files for c in runs}), 8)
        self.assertEqual(len({str(Path(c.p_base)/'generated') for c in runs}), 8)
        self.assertEqual(len({c.p_seeds for c in runs}), 1)
        self.assertTrue(all(c.gene_dim == config.gene_dim for c in runs))

    def test_bc_seed_schema_and_stale_manifest_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / 'seeds.tsv'
            seed.write_text('probeID\tdeltaBeta\tP.Value\tCHR\ncg00000001\t-0.2\t0.01\t1\n')
            config = OmegaConf.create({'p_seeds':str(seed), 'p_preprocessing_manifest':str(Path(directory)/'manifest.json')})
            with self.assertRaisesRegex(ValueError, 'not completed'):
                validate_seed_manifest(config)
            signature = seed_signature(seed)
            write_seed_manifest(config, signature)
            validate_seed_manifest(config)
            seed.write_text(seed.read_text().replace('-0.2', '0.3'))
            with self.assertRaisesRegex(ValueError, 'different seed'):
                validate_seed_manifest(config)
            with self.assertRaisesRegex(ValueError, 'changed during'):
                write_seed_manifest(config, signature)
