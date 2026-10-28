"""Dataset config lifecycle without running expensive model training."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from omegaconf import OmegaConf
from dataset_configs import downstream_configs,tuning_source,write_tuned_dataset
from model_settings import apply_model_settings

class DatasetConfigTests(unittest.TestCase):
    def test_tuning_creates_dataset_configs_without_editing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'preprocess.yaml'
            seed=root/'seeds.tsv';seed.write_text('CpG_ID\tGene\tdeltaBeta_AML_minus_Normal\tCpG_Gene\ncg00000001\tGENE\t.4\tcg00000001_GENE\n')
            cfg=OmegaConf.create(dict(config_scope='preprocessing',corr_threshold=[.2,.33],knn_neighbours=[3000],size_negative=[10000],gene_dim=124,p_raw=str(root)+'/',p_arrays='${p_raw}matrix.gz',p_seeds=str(seed),p_base=str(root/'dataset_a')+'/'))
            OmegaConf.save(cfg,source);original=source.read_bytes()
            search=OmegaConf.create({'runtime_defaults':{'hidden_dim':32,'best_trained_model':'PNA','model_type':['PNA','GCN'],'model_overrides':{},'p_generated_files':'${p_base}generated/','p_torch_model':'${p_generated_files}model_${model_type}.pt'}})
            current=tuning_source(source,search);current.corr_threshold=.33;current.knn_neighbours=3000;current.size_negative=10000
            self.assertNotIn('hidden_dim',OmegaConf.load(source))
            self.assertEqual(current.p_arrays,str(root/'matrix.gz'))
            self.assertEqual(current.p_generated_files,str(root/'dataset_a/generated')+'/')
            exports=root/'exports';exports.mkdir()
            OmegaConf.save(OmegaConf.create(dict(hidden_dim=128,gnn_layers=1,neighbors_spread=[20],checkpoint_selection_metric='macro_f1',p_generated_files='/trial/final/')),exports/'best_config.yaml')
            (exports/'best_params.json').write_text(json.dumps(dict(model='PNA',parameters={'hidden_dim':128,'gnn_layers':1,'neighbor_fanout':20})))
            target=write_tuned_dataset(current,exports)
            saved=downstream_configs(target)[0];self.assertEqual(source.read_bytes(),original)
            self.assertEqual(list(saved.model_type),['PNA'])
            self.assertEqual(saved.p_base,str(root/'dataset_a')+'/')
            self.assertEqual(saved.p_arrays,str(root/'matrix.gz'))
            active=apply_model_settings(saved,'PNA');self.assertEqual(active.hidden_dim,128)
            self.assertEqual(list(active.neighbors_spread),[20]);self.assertEqual(active.p_torch_model,str(root/'dataset_a/generated/models/model_PNA.pt'))
            with self.assertRaisesRegex(ValueError,'cannot override'):
                downstream_configs(target,SimpleNamespace(corr_threshold=.2,knn_neighbours=None,size_negative=None))
            OmegaConf.save(OmegaConf.create({'estimator_params':{'max_depth':3}}),exports/'best_config.yaml')
            (exports/'best_params.json').write_text(json.dumps(dict(model='XGBoost',parameters={'max_depth':3})))
            write_tuned_dataset(current,exports)
            saved=downstream_configs(target)[0]
            self.assertEqual(list(saved.model_type),['PNA','XGBoost'])
            self.assertEqual(apply_model_settings(saved,'XGBoost').estimator_params.max_depth,3)
            self.assertEqual(apply_model_settings(saved,'PNA').hidden_dim,128)
            # Every dataset owns its independent config, even under one source grid.
            current.p_base=str(root/'dataset_b')+'/'
            other=write_tuned_dataset(current,exports)
            self.assertEqual(list(downstream_configs(other)[0].model_type),['XGBoost'])
            seed.write_text(seed.read_text().replace('.4','.5'))
            current.p_base=str(root/'dataset_a')+'/'
            with self.assertRaisesRegex(ValueError,'different seeds'):
                write_tuned_dataset(current,exports)

    def test_training_pipeline_passes_dataset_config_to_training(self):
        import run_pipeline
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'train_gnn').mkdir()
            folder=root/'data_corr_0.33_knn_n_3000_n_neg_10000';folder.mkdir()
            source=root/'preprocess.yaml'
            OmegaConf.save(OmegaConf.create({'config_scope':'preprocessing','corr_threshold':[.33],'knn_neighbours':[3000],'size_negative':[10000],'p_dataset_root':str(root)}),source)
            snapshot=folder/'config.yaml'
            OmegaConf.save(OmegaConf.create({'config_scope':'dataset','corr_threshold':.33,'knn_neighbours':3000,'size_negative':10000,'p_base':str(folder)+'/','model_type':['PNA']}),snapshot)
            with patch.object(run_pipeline,'HERE',root), patch.object(run_pipeline.os,'chdir'), patch.object(run_pipeline.sys,'argv',['run_pipeline.py','--stage','train','--config',str(source)]), patch.object(run_pipeline.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as execute:
                run_pipeline.main()
            command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--config')+1],str(snapshot))
            self.assertNotIn('--corr-threshold',command)

    def test_training_folder_argument_selects_one_snapshot_without_grid_expansion(self):
        import main as training
        import dataset_configs
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'data_corr_0.33_knn_n_500_n_neg_5000';folder.mkdir()
            OmegaConf.save(OmegaConf.create({'config_scope':'dataset','p_base':'../wrong_folder/',
                'corr_threshold':.33,'knn_neighbours':500,'size_negative':5000,'model_type':['PNA']}),folder/'config.yaml')
            args=SimpleNamespace(folder=folder.name+'/',models=None,corr_threshold=None,knn_neighbours=None,size_negative=None)
            with patch.object(dataset_configs,'ROOT',root),patch.object(training,'run_training_config',side_effect=lambda c:c) as train:
                result=training.run_training(root/'unused.yaml',args)
            train.assert_called_once()
            self.assertEqual(result.p_base,str(folder)+'/')
            self.assertEqual(result.p_generated_files,str(folder/'generated')+'/')
            self.assertFalse((root/'wrong_folder').exists())

    def test_model_outputs_outside_dataset_are_rejected_before_writing(self):
        import main as training
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cfg=OmegaConf.create({'config_scope':'dataset','model_type':['PNA'],
                'p_base':str(root/'selected')+'/','p_torch_model':str(root/'outside/model.pt'),
                'model_overrides':{}})
            with self.assertRaisesRegex(ValueError,'outside the selected dataset folder'):
                training.model_configs(cfg)
            self.assertFalse((root/'outside').exists())

    def test_preprocessing_config_routes_to_saved_snapshot_and_requires_tuning(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'preprocess.yaml'
            OmegaConf.save(OmegaConf.create({'config_scope':'preprocessing','corr_threshold':[.33],'knn_neighbours':[3000],'size_negative':[10000],'p_dataset_root':str(root)}),source)
            with self.assertRaisesRegex(FileNotFoundError,'tuning stage'):
                downstream_configs(source)
            folder=root/'data_corr_0.33_knn_n_3000_n_neg_10000';folder.mkdir()
            saved=OmegaConf.create({'config_scope':'dataset','corr_threshold':.33,'knn_neighbours':3000,'size_negative':10000,'hidden_dim':128,'model_type':['PNA'],'p_base':str(folder)+'/'})
            OmegaConf.save(saved,folder/'config.yaml')
            self.assertEqual(downstream_configs(source)[0].hidden_dim,128)

if __name__=='__main__':unittest.main()
