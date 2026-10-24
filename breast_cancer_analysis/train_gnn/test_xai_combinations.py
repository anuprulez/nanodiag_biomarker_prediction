"""Regression checks for per-configuration XAI routing and isolated state."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from omegaconf import OmegaConf

import run_xai_explainer as xai


class XaiCombinationTests(unittest.TestCase):
    def config(self):
        config = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(OmegaConf.load(xai.HERE.parent / 'config/optuna.yaml').runtime_defaults,resolve=False)), OmegaConf.load(xai.HERE.parent / 'config/config.yaml'))
        config.pop('config_scope',None)  # Legacy full-config grid compatibility fixture.
        config.corr_threshold = [0.2, 0.33, 0.5, 0.7]
        config.knn_neighbours = [500, 1500, 3000, 4500]
        config.size_negative = [5000, 10000, 15000]
        config.model_type = ['PNA', 'GCN']
        config.best_trained_model = 'GCN'
        config.model_overrides.GCN = {'hidden_dim': 64}
        return config

    def test_tuned_dataset_config_keeps_its_path_and_model_settings(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'config.yaml'
            OmegaConf.save(OmegaConf.create({'config_scope':'dataset','p_base':directory+'/',
                'best_trained_model':'PNA','model_type':['PNA'],'hidden_dim':32,
                'model_overrides':{'PNA':{'hidden_dim':128}}}),path)
            with patch.object(xai,'explain_configuration') as explain:
                xai.run_explanations(path,node_id=7)
            current=explain.call_args.args[0]
            self.assertEqual(current.p_base,directory+'/')
            self.assertEqual(current.hidden_dim,128)
            self.assertEqual(current.model_type,'PNA')

    def test_all_48_combinations_bind_model_and_paths(self):
        config = self.config()
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'config.yaml'
            OmegaConf.save(config, source)
            with patch.object(xai, 'explain_configuration') as explain, contextlib.redirect_stdout(io.StringIO()):
                xai.run_explanations(source, node_name='cg00000001_GENE')
            self.assertEqual(explain.call_count, 48)
            bases = set()
            for call in explain.call_args_list:
                current = call.args[0]
                bases.add(current.p_base)
                self.assertEqual(current.model_type, 'GCN')
                self.assertEqual(current.hidden_dim, 64)
                self.assertTrue(current.p_torch_data.startswith(current.p_base + 'generated/'))
                self.assertTrue(current.p_torch_data.endswith('_GCN.pt'))
                self.assertTrue(current.p_torch_model.endswith('_GCN.pt'))
                self.assertEqual(call.kwargs['node_name'], 'cg00000001_GENE')
            self.assertEqual(len(bases), 48)

    def test_scalar_overrides_select_one_combination(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'config.yaml'
            OmegaConf.save(self.config(), source)
            args = SimpleNamespace(corr_threshold=0.5, knn_neighbours=1500, size_negative=5000)
            with patch.object(xai, 'explain_configuration') as explain, contextlib.redirect_stdout(io.StringIO()):
                xai.run_explanations(source, args, node_id=7)
            explain.assert_called_once()
            self.assertEqual(Path(explain.call_args.args[0].p_base),xai.HERE.parent/'data_corr_0.5_knn_n_1500_n_neg_5000')
            self.assertEqual(explain.call_args.kwargs['node_id'], 7)

    def test_name_resolution_uses_each_datasets_ids_and_checks_bounds(self):
        with TemporaryDirectory() as directory:
            table = Path(directory) / 'nodes.csv'
            config = OmegaConf.create({'p_out_genes':str(table),'p_base':directory})
            for expected in [1, 2]:
                names = ['other', 'cg00000001_GENE', 'third'] if expected == 1 else ['other', 'third', 'cg00000001_GENE']
                pd.DataFrame({0:range(3),1:names}).to_csv(table,sep=' ',header=False,index=False)
                self.assertEqual(xai.resolve_explanation_node(SimpleNamespace(num_nodes=3),config,
                                 node_name='cg00000001_GENE'),expected)
            with self.assertRaisesRegex(ValueError, 'outside'):
                xai.resolve_explanation_node(SimpleNamespace(num_nodes=3),config,node_id=3)
            with self.assertRaisesRegex(ValueError, 'found 0'):
                xai.resolve_explanation_node(SimpleNamespace(num_nodes=3),config,node_name='missing')

    def test_model_loader_uses_explicit_config(self):
        config = OmegaConf.create({'gene_dim':4,'hidden_dim':32,'gnn_layers':1,
                                   'dropout':0.1,'num_classes':5,'SEED':32})
        data = Data(x=torch.randn(3,4),edge_index=torch.tensor([[0,1],[1,0]]))
        model = xai.utils.choose_model(config,data,'GCN',False)
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'model.pt'
            torch.save(model.state_dict(),checkpoint)
            restored=xai.load_model(checkpoint,data,'GCN',config)
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,restored.state_dict()[key]))

    def test_model_loader_reports_stale_architecture_override(self):
        config = OmegaConf.create({'gene_dim':4,'hidden_dim':32,'gnn_layers':1,
                                   'dropout':0.1,'num_classes':5,'SEED':32})
        data = Data(x=torch.randn(3,4),edge_index=torch.tensor([[0,1],[1,0]]))
        model = xai.utils.choose_model(config,data,'PNA',False)
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'model.pt'
            torch.save(model.state_dict(),checkpoint)
            config.hidden_dim=16;config.gnn_layers=3
            with self.assertRaisesRegex(RuntimeError, 'Checkpoint hidden_dim=32, gnn_layers=1'):
                xai.load_model(checkpoint,data,'PNA',config)

    def test_feature_plot_receives_current_dataset(self):
        data=Data(x=torch.rand(3,4),y=torch.tensor([0,1,2]),
                  edge_index=torch.tensor([[0,1],[1,0]]),n_id=torch.arange(3))
        graph=nx.Graph([(0,1),(1,2)])
        config=OmegaConf.create({'masks_for_seed':1,'explainer_epochs':1,'neighbour_predictions':[0,1,2,3,4],
               'explanation_nodes_ratio':1.0,'explain_neighbors_spread':[2], 'graph_subtype':'bidirectional','num_workers':0})
        class Model(torch.nn.Module):
            def forward(self,x,edge_index):
                return torch.ones(len(x),5), x
        explanation=SimpleNamespace(edge_mask=torch.ones(2),node_mask=torch.ones(3,4))
        with patch.object(xai.torch.cuda,'is_available',return_value=False), \
             patch.object(xai,'NeighborLoader',return_value=[data]), \
             patch.object(xai,'Explainer',return_value=lambda *a,**kw: explanation), \
             patch.object(xai,'compute_rankings',return_value=[0,1]), \
             patch.object(xai.plot_gnn,'draw_xai_local_graph',return_value=[0,1]), \
             patch.object(xai.plot_gnn,'draw_xai_global_graph'), \
             patch.object(xai.plot_gnn,'plot_feature_importance') as plot, \
             contextlib.redirect_stdout(io.StringIO()):
            xai.explain_candiate_gene(Model(),data,0,graph,'GCN',config)
        self.assertIs(plot.call_args.args[0],data)


if __name__ == '__main__':
    unittest.main()
