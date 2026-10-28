"""Small real-model Optuna integration tests, independent of breast cancer source data."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tune_models import HERE, GNN_MODELS, optimize_model, trial_config, suggest_parameters


class OptunaTuningTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(OmegaConf.load(HERE.parent / 'config/optuna.yaml').runtime_defaults,resolve=False)), OmegaConf.load(HERE.parent / 'config/config.yaml'))
        # Production PNA winners must not override the tiny test epoch budget.
        self.config.model_overrides.pop('PNA', None)
        self.config.p_base = str(self.root) + '/'
        self.config.p_base_files = str(self.root / 'preprocessed') + '/'
        self.config.p_generated_files = str(self.root / 'generated') + '/'
        Path(self.config.p_base_files).mkdir()
        Path(self.config.p_generated_files).mkdir()
        self.config.gene_dim = 6
        self.config.n_epo = 2
        self.config.n_edges = 1000
        self.config.hpo_device = 'cpu'
        self.config.tabular_device = 'cpu'
        self.config.tabular_threads = 1
        self.config.neighbors_spread = [2, 2, 2]
        self.config.model_type = 'PNA'
        self.config.corr_threshold = 0.33
        self.config.knn_neighbours = 500
        self.config.size_negative = 10000
        self.config.p_seeds = str(self.root / 'seeds.tsv')
        Path(self.config.p_seeds).write_text('fixture seed input')
        self.search = OmegaConf.load(HERE.parent / 'config/optuna.yaml')
        # Keep test models tiny, while exercising the real model/training code.
        self.search.spaces.GNN = {
            'batch_size': {'type':'categorical','choices':[32]},
            'hidden_dim': {'type':'categorical','choices':[32]},
            'gnn_layers': {'type':'int','low':1,'high':2},
            'dropout': {'type':'float','low':0.1,'high':0.2},
            'grad_clip_norm': {'type':'float','low':1.0,'high':2.0},
            'learning_rate': {'type':'float','low':0.001,'high':0.002},
            'weight_decay': {'type':'float','low':0.001,'high':0.002},
            'neighbors_spread': {'type':'categorical','choices':[[2,2,2]]}}
        for model, field in [('XGBoost','n_estimators'),('CatBoost','iterations')]:
            if model not in self.search.spaces:
                self.search.spaces[model] = {}
            self.search.spaces[model][field] = {'type':'int','low':5,'high':10,'step':5}
        for model, field in [('XGBoost','max_depth'),('CatBoost','depth')]:
            self.search.spaces[model][field] = {'type':'int','low':2,'high':3}
        names = [f'cg{i:08d}_GENE' for i in range(100)]
        splits = ['train']*50 + ['val']*25 + ['test']*25
        rng = np.random.default_rng(32)
        pd.DataFrame({0:range(100),1:names}).to_csv(self.config.p_out_genes,sep=' ',index=False,header=False)
        pd.DataFrame({'feature_name':names,'split':splits}).to_csv(self.config.p_split_manifest,index=False)
        features = pd.DataFrame(rng.random((100,4)),columns=['ring','NetRank','HeatDiff','InfoDiff'])
        features.insert(0,'name',names)
        features.insert(1,'class',[i%2 for i in range(100)])
        features.to_csv(self.config.p_nedbit_features,index=False)
        pd.DataFrame(rng.random((2,100)),columns=names).to_csv(self.config.p_combined_pos_neg_signals,sep='\t',index=False)
        pd.DataFrame({0:names,1:rng.random(100),2:[i%5+1 for i in range(100)]}).to_csv(self.config.p_out_gene_rankings,sep=' ',header=False,index=False)
        edges=[]
        for first,last in [(0,50),(50,75),(75,100)]:
            edges.extend((i,first+(i-first+1)%(last-first)) for i in range(first,last))
        pd.DataFrame(edges).to_csv(self.config.p_out_links,sep=' ',index=False,header=False)

    def optimize(self, name, trials=1):
        # Silence large existing feature-loader dumps without hiding failures.
        with contextlib.redirect_stdout(io.StringIO()):
            return optimize_model(self.config,self.search,name,trials)

    def test_all_gnn_families_train_and_export(self):
        import optuna
        for name in GNN_MODELS:
            with self.subTest(model=name):
                directory=self.optimize(name)
                result=json.loads((directory/'best_params.json').read_text())
                self.assertFalse(result['test_evaluated'])
                self.assertTrue((directory/'best_model.pt').is_file())
                best=OmegaConf.load(directory/'best_config.yaml')
                self.assertEqual(len(best.neighbors_spread),best.gnn_layers)
                self.assertIsInstance(result['parameters']['neighbors_spread'],list)
                self.assertEqual(result['parameters']['neighbors_spread'],list(best.neighbors_spread))
                self.assertEqual(best.model_type,name)
                self.assertNotIn('model_overrides',best)
                if name in ('GATv2','GraphTransformer'):
                    self.assertEqual(best.hidden_dim%best.heads,0)

    def test_trees_resume_and_reject_changed_inputs(self):
        import optuna
        from benchmark_tabular import predict as original_predict
        for name in ('XGBoost','CatBoost'):
            with self.subTest(model=name):
                # Objective must predict validation rows only (25 nodes).
                def checked_predict(model,x):
                    self.assertEqual(len(x),25)
                    return original_predict(model,x)
                with patch('benchmark_tabular.predict',side_effect=checked_predict):
                    directory=self.optimize(name)
                    self.optimize(name)
                study=optuna.load_study(study_name=name,storage='sqlite:///'+str(directory/'study.sqlite3'))
                self.assertEqual(len(study.trials),2)
                self.assertNotEqual(study.trials[0].params, study.trials[1].params)
                self.assertTrue((directory/'best_model.joblib').is_file())
                result=json.loads((directory/'best_params.json').read_text())
                best=OmegaConf.load(directory/'best_config.yaml')
                for key,value in result['parameters'].items():
                    self.assertEqual(best.estimator_params[key],value)
                # Ordinary training must accept the exported config and params.
                from train_tabular import train_tabular_model
                Path(best.p_generated_files).mkdir(parents=True, exist_ok=True)
                with contextlib.redirect_stdout(io.StringIO()), patch('plot_gnn.plot_features'), patch('plot_gnn.plot_confusion_matrix'):
                    metrics = train_tabular_model(best)
                self.assertTrue(Path(metrics['model_path']).exists())
                # Changing a training setting must not silently mix objectives.
                self.config.n_epo=3
                with self.assertRaisesRegex(ValueError,'inputs/settings changed'):
                    self.optimize(name)
                self.config.n_epo=2

    def test_fresh_tuning_discards_trials_sampler_and_old_outputs(self):
        import optuna
        directory=self.optimize('XGBoost')
        stale=directory/'stale.log';stale.write_text('old run')
        unrelated=directory.parent/'GCN/keep.txt';unrelated.parent.mkdir();unrelated.write_text('keep')
        # Fresh studies must also accept updated input fingerprints.
        self.config.n_epo=3
        with contextlib.redirect_stdout(io.StringIO()):
            optimize_model(self.config,self.search,'XGBoost',1,fresh=True)
        study=optuna.load_study(study_name='XGBoost',storage='sqlite:///'+str(directory/'study.sqlite3'))
        self.assertEqual(len(study.trials),1)
        self.assertEqual(study.trials[0].number,0)
        self.assertFalse(stale.exists());self.assertTrue(unrelated.exists())
        self.assertTrue((directory/'best_model.joblib').is_file())

    def test_unpromising_gnn_trial_is_pruned(self):
        import optuna
        with patch('optuna.pruners.MedianPruner', return_value=optuna.pruners.ThresholdPruner(lower=1.1)):
            with self.assertRaisesRegex(RuntimeError, 'No completed trials'):
                self.optimize('GCN')
        directory = Path(self.config.p_generated_files) / 'optuna/GCN'
        study = optuna.load_study(study_name='GCN', storage='sqlite:///' + str(directory/'study.sqlite3'))
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.PRUNED)
        self.assertFalse((directory/'best_params.json').exists())

    def test_model_specific_space_and_neighbor_fanouts(self):
        import optuna
        self.search.spaces.PNA = {
            'hidden_dim': {'type':'categorical','choices':[64]},
            'neighbor_fanout': {'type':'categorical','choices':[20]},
            'class_weight_power': {'type':'categorical','choices':[0.5]}}
        trial = optuna.create_study().ask()
        parameters = suggest_parameters(trial,self.search,'PNA')
        current = trial_config(self.config,parameters,'PNA')
        self.assertEqual(current.hidden_dim,64)
        self.assertEqual(current.class_weight_power,0.5)
        self.assertEqual(list(current.neighbors_spread),[20,10,5][:current.gnn_layers])
        self.assertNotIn('neighbor_fanout',current)
        # PNA-only parameters must never leak into GCN suggestions.
        other = suggest_parameters(optuna.create_study().ask(),self.search,'GCN')
        self.assertNotIn('class_weight_power',other)
        self.assertNotIn('neighbor_fanout',other)

    def test_shared_dropout_and_neighborhood_spaces_for_every_gnn(self):
        import optuna
        search = OmegaConf.load(HERE.parent / 'config/optuna.yaml')
        for name in GNN_MODELS:
            parameters = suggest_parameters(optuna.create_study().ask(), search, name)
            self.assertIn('feature_dropout', parameters)
            self.assertIn('embedding_dropout', parameters)
            self.assertIn('neighbors_spread', parameters)
            current = trial_config(self.config, parameters, name)
            hops = json.loads(parameters['neighbors_spread'])
            expected = (hops + [hops[-1]] * current.gnn_layers)[:current.gnn_layers]
            self.assertEqual(list(current.neighbors_spread), expected)
            self.assertEqual(current.feature_dropout, parameters['feature_dropout'])
            self.assertEqual(current.embedding_dropout, parameters['embedding_dropout'])
            if name in ('GATv2','GraphTransformer'):
                self.assertIn(parameters['heads'], [1,2,4,8])
                self.assertEqual(current.hidden_dim % current.heads, 0)
            else:
                self.assertNotIn('heads', parameters)

    def test_foundation_device_defaults_and_explicit_overrides(self):
        from benchmark_tabular import default_tabular_device
        for name in ('TabPFN', 'TabICL'):
            self.assertEqual(default_tabular_device(name), 'auto')
            self.assertEqual(default_tabular_device(name, 'cpu'), 'cpu')
            self.assertEqual(default_tabular_device(name, 'cuda'), 'cuda')
        for name in ('XGBoost', 'CatBoost'):
            self.assertEqual(default_tabular_device(name), 'cpu')
            self.assertEqual(default_tabular_device(name, 'cuda'), 'cuda')

    def test_foundation_spaces_match_installed_constructor_parameters(self):
        import inspect
        import optuna
        from tabpfn import TabPFNClassifier
        from tabicl import TabICLClassifier
        for name,cls in [('TabPFN',TabPFNClassifier),('TabICL',TabICLClassifier)]:
            study=optuna.create_study()
            parameters=suggest_parameters(study.ask(),self.search,name)
            self.assertTrue(set(parameters)<=set(inspect.signature(cls).parameters))


if __name__=='__main__':
    unittest.main()
