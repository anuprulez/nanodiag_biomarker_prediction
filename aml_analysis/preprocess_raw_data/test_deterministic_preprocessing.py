"""Regression checks for seeded negative selection and CpG partitions."""
import random
import unittest

import numpy as np
import pandas as pd

from process_raw_illumina_arrays_aml import select_hv_features, partition_probe_features


class DeterministicPreprocessingTests(unittest.TestCase):
    def setUp(self):
        values = np.random.default_rng(17).uniform(0.05, 0.95, (40, 120))
        self.signals = pd.DataFrame(
            values, columns=[f'cg{i:08d}_GENE' for i in range(120)]
        )
        self.ratios = [0.7, 0.15, 0.15]

    def test_selection_and_partitions_survive_reordering_and_rng_changes(self):
        for method in ('variance', 'mad', 'seurat'):
            with self.subTest(method=method):
                original = select_hv_features(self.signals, 60, method, seed=32)
                random.seed(991)
                np.random.seed(991)
                reordered = select_hv_features(
                    self.signals.sample(frac=1, random_state=8).sample(
                        frac=1, axis=1, random_state=9), 60, method, seed=32
                )
                self.assertEqual(original.columns.tolist(), reordered.columns.tolist())
                self.assertEqual(original.shape[1], 60)
                first = partition_probe_features(
                    original.columns.tolist(), self.ratios, random.Random('32:negative-split'))
                second = partition_probe_features(
                    list(reversed(reordered.columns)), self.ratios, random.Random('32:negative-split'))
                self.assertEqual(first, second)
                smaller = select_hv_features(self.signals, 30, method, seed=32)
                self.assertEqual(smaller.columns.tolist(), original.columns[:30].tolist())

    def test_ties_have_exact_seeded_selection(self):
        tied = pd.DataFrame(np.tile(np.linspace(0.05, 0.95, 40)[:, None], (1, 120)),
                            columns=self.signals.columns)
        for method in ('variance', 'mad', 'seurat'):
            with self.subTest(method=method):
                first = select_hv_features(tied, 60, method, seed=32)
                repeated = select_hv_features(tied, 60, method, seed=32)
                changed = select_hv_features(tied, 60, method, seed=33)
                self.assertEqual(first.shape[1], 60)
                self.assertEqual(first.columns.tolist(), repeated.columns.tolist())
                self.assertNotEqual(first.columns.tolist(), changed.columns.tolist())

    def test_cpg_mappings_stay_together_and_seed_changes_partition(self):
        names = [f'cg{i:08d}_{gene}' for i in range(100) for gene in ('A', 'B')]
        first = partition_probe_features(names, self.ratios, random.Random('32:negative-split'))
        reordered = partition_probe_features(list(reversed(names)), self.ratios,
                                             random.Random('32:negative-split'))
        changed = partition_probe_features(names, self.ratios, random.Random('33:negative-split'))
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)
        owner = {}
        for split, features in first.items():
            for feature in features:
                probe = feature.split('_', 1)[0]
                self.assertEqual(owner.setdefault(probe, split), split)
        self.assertEqual(sum(map(len, first.values())), len(names))


if __name__ == '__main__':
    unittest.main()
