"""Heatmap colors must represent annotated scores on the colorbar scale."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plot_gnn import plot_model_comparison


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class HeatmapTests(unittest.TestCase):
    def test_raw_scores_and_missing_values_share_colorbar_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            config=Config(p_plot=temp, p_data=temp, n_edges=10, n_epo=5, dpi=50,
                          NORMALIZE_FOR_HEATMAP=True)
            results={'A': {'te_f1_macro': .377, 'te_acc': .9},
                     'B': {'te_f1_macro': .1, 'te_acc': .9}}
            def inspect(fig, path, dpi):
                ax=fig.axes[0]
                image=ax.images[0]
                self.assertEqual(image.norm.vmin, 0.)
                self.assertEqual(image.norm.vmax, 1.)
                self.assertAlmostEqual(image.get_array()[0, 2], .377)
                self.assertAlmostEqual(image.get_array()[1, 2], .1)
                self.assertAlmostEqual(image.get_array()[0, 1], .9)
                self.assertAlmostEqual(image.get_array()[1, 1], .9)
                self.assertTrue(np.ma.is_masked(image.get_array()[0, 0]))
                self.assertIn('0.377', [t.get_text() for t in ax.texts])
                self.assertIn('N/A', [t.get_text() for t in ax.texts])
                self.assertEqual(fig.axes[1].get_ylabel(), 'Metric score')
                np.testing.assert_allclose(image.to_rgba(.377), image.cmap(.377))
            with patch('plot_gnn._save', side_effect=inspect) as save:
                self.assertIsNotNone(plot_model_comparison(results, ['A','B'], config))
                save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
