"""Exercise configurable PNA convolutions and compatibility with existing checkpoints."""
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from gnn_network import GPNA
from pna_options import pna_conv_options


class Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class PNAOptionsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Config(hidden_dim=32, gnn_layers=2, dropout=0.1,
                             feature_dropout=0., embedding_dropout=0.,
                             SEED=32, gene_dim=6, num_classes=5)
        src = torch.arange(12).repeat_interleave(2)
        dst = torch.stack(((torch.arange(12) + 1) % 12,
                           (torch.arange(12) - 1) % 12), dim=1).flatten()
        self.data = Data(x=torch.randn(12, 6), edge_index=torch.stack((src, dst)),
                         train_mask=torch.ones(12, dtype=torch.bool))

    def test_default_checkpoint_compatibility(self):
        default = GPNA(self.config, self.data)
        self.config.pna_aggregators = 'mean,min,max,std'
        self.config.pna_scalers = 'identity,amplification,attenuation'
        self.config.pna_towers = self.config.pna_pre_layers = self.config.pna_post_layers = 1
        explicit = GPNA(self.config, self.data)
        explicit.load_state_dict(default.state_dict(), strict=True)
        self.assertTrue(torch.equal(default(self.data.x, self.data.edge_index)[0],
                                    explicit(self.data.x, self.data.edge_index)[0]))

    def test_variants_forward_and_backward(self):
        for changes in [dict(pna_scalers='identity'), dict(pna_scalers='identity,attenuation'),
                        dict(pna_aggregators='mean,max,std'), dict(pna_aggregators='mean,std'),
                        dict(pna_towers=4), dict(pna_pre_layers=2), dict(pna_post_layers=2)]:
            with self.subTest(changes=changes):
                config = Config(**vars(self.config), **changes)
                model = GPNA(config, self.data, training=True)
                logits = model(self.data.x, self.data.edge_index)[0]
                self.assertEqual(tuple(logits.shape), (12, 5))
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(12) % 5)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_invalid_options(self):
        for changes in [dict(pna_towers=3), dict(pna_scalers='bad'),
                        dict(pna_aggregators='mean,mean'), dict(pna_post_layers=0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pna_conv_options(Config(**changes), 32)


if __name__ == '__main__':
    unittest.main()
