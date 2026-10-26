"""Folder routing, XAI ID selection, and generated artifact layout integration."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from omegaconf import OmegaConf
import run_pipeline as pipeline
from artifact_layout import apply_artifact_layout,migrate_generated,artifact_file

class PipelineFolderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'preprocess.yaml'
        self.config=OmegaConf.create({'config_scope':'preprocessing','corr_threshold':[.2,.33],
            'knn_neighbours':[3000],'size_negative':[10000],'p_dataset_root':str(self.root),
            'p_base_files':'${p_base}preprocessed/','p_preprocessing_manifest':'${p_base_files}seed_preprocessing_manifest.json'})
        OmegaConf.save(self.config,self.source)
        self.folders=[]
        for corr in [.2,.33]:
            folder=self.root/f'data_corr_{corr}_knn_n_3000_n_neg_10000';folder.mkdir()
            cfg=OmegaConf.create({'config_scope':'dataset','corr_threshold':corr,'knn_neighbours':3000,
                'size_negative':10000,'p_base':str(folder)+'/', 'model_type':['PNA'],
                'p_generated_files':'${p_base}generated/', 'p_data':'${p_generated_files}',
                'p_plot':'${p_generated_files}', 'p_model':'${p_generated_files}models/',
                'p_pred_likely_pos':'${p_generated_files}pred_likely_pos_aml_${model_type}.csv',
                'p_torch_data':'${p_generated_files}data_aml_${model_type}.pt',
                'model_overrides':{},'best_trained_model':'PNA'})
            OmegaConf.save(cfg,folder/'config.yaml');self.folders.append(folder)
            gen=folder/'generated';gen.mkdir()
            (gen/'pred_likely_pos_aml_PNA.csv').write_text('test_gene_ids\ttest_gene_names\n17\tcg00000001_GENE\n19\tcg00000002_GENE\n')

    def invoke(self,stage,folder=None):
        argv=['run_pipeline.py','--stage',stage,'--config',str(self.source)]
        if folder:argv+=['--folder',str(folder)]
        with patch.object(pipeline.sys,'argv',argv),patch.object(pipeline.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as execute:
            pipeline.main()
        return [call.args[0] for call in execute.call_args_list]

    def test_no_folder_selects_all_existing_for_all_downstream_stages(self):
        for stage in ['tune','train','xai','compare-predictions']:
            # Training is mocked, so supply its expected prediction table for XAI.
            if stage=='xai':
                for folder in self.folders:
                    target=folder/'generated/data'/('pred_likely_pos_' + ('aml' if 'aml' in pipeline.SCRIPTS['preprocess'] else 'bc') + '_PNA.csv')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_text('test_gene_ids\ttest_gene_names\n17\tcg00000001_GENE\n')
            commands=self.invoke(stage)
            self.assertEqual(len(commands),2,stage)
            for command,folder in zip(commands,self.folders):
                self.assertEqual(command[command.index('--config')+1],str(folder/('preprocessing_config.yaml' if stage=='tune' else 'config.yaml')))
                if stage=='xai':self.assertEqual(command[command.index('--node-id')+1],'17')
                if stage=='compare-predictions':self.assertTrue((folder/'evaluation').is_dir())

    def test_specific_folder_and_preprocess_parameter_binding(self):
        folder=self.folders[1]
        commands=self.invoke('preprocess',folder)
        self.assertEqual(len(commands),2) # preprocessing plus validation
        saved=OmegaConf.load(folder/'preprocessing_config.yaml')
        self.assertEqual(saved.corr_threshold,.33);self.assertEqual(saved.p_base,str(folder)+'/')
        self.assertEqual(saved.config_scope,'dataset_preprocessing')
        for stage in ['tune','train','xai','compare-predictions']:
            self.assertEqual(len(self.invoke(stage,folder)),1)

    def test_preprocess_skips_only_completed_not_just_existing_folders(self):
        folder=self.folders[0];(folder/'preprocessed').mkdir()
        (folder/'preprocessed/seed_preprocessing_manifest.json').write_text('{}')
        self.assertEqual(self.invoke('preprocess',folder),[])
        self.assertEqual(len(self.invoke('preprocess',self.folders[1])),2)

    def test_batch_preprocessing_restarts_completed_folders_and_clears_logs(self):
        for folder in self.folders:
            preprocessed=folder/'preprocessed';preprocessed.mkdir()
            (preprocessed/'seed_preprocessing_manifest.json').write_text('{}')
            (preprocessed/'stale.csv').write_text('old')
            logs=folder/'logs';logs.mkdir()
            (logs/'preprocess.log').write_text('old preprocessing log')
            (logs/'train.log').write_text('keep other stage log')
        commands=self.invoke('preprocess')
        self.assertEqual(len(commands),4)
        for folder in self.folders:
            self.assertFalse((folder/'preprocessed/stale.csv').exists())
            self.assertFalse((folder/'preprocessed/seed_preprocessing_manifest.json').exists())
            self.assertEqual((folder/'logs/preprocess.log').read_text(),'')
            self.assertEqual((folder/'logs/train.log').read_text(),'keep other stage log')

    def test_batch_tuning_requests_new_studies(self):
        self.assertTrue(all('--fresh' in command for command in self.invoke('tune')))
        self.assertNotIn('--fresh', self.invoke('tune',self.folders[0])[0])

    def test_log_is_cleared_before_stage_preparation_failure(self):
        folder=self.folders[0];logs=folder/'logs';logs.mkdir()
        (logs/'train.log').write_text('stale success')
        (folder/'config.yaml').unlink()
        with self.assertRaises(FileNotFoundError):self.invoke('train',folder)
        self.assertEqual((logs/'train.log').read_text(),'')

    def test_batch_comparison_removes_previous_results(self):
        for folder in self.folders:
            output=folder/'evaluation';output.mkdir()
            (output/'old_report.json').write_text('old')
        self.invoke('compare-predictions')
        for folder in self.folders:
            self.assertTrue((folder/'evaluation').is_dir())
            self.assertFalse((folder/'evaluation/old_report.json').exists())

    def test_batch_dry_run_preserves_logs_and_completed_outputs(self):
        folder=self.folders[0];output=folder/'preprocessed';output.mkdir()
        marker=output/'seed_preprocessing_manifest.json';marker.write_text('{}')
        logs=folder/'logs';logs.mkdir();log=logs/'preprocess.log';log.write_text('keep')
        with patch.object(pipeline.sys,'argv',['run_pipeline.py','--stage','preprocess','--config',str(self.source),'--dry-run']),patch.object(pipeline.subprocess,'run') as execute:
            pipeline.main()
        execute.assert_not_called()
        self.assertEqual(log.read_text(),'keep');self.assertEqual(marker.read_text(),'{}')

    def test_all_stage_order(self):
        commands=self.invoke('all',self.folders[0])
        scripts=[Path(command[2] if command[1]=='-u' else command[1]).name for command in commands]
        self.assertEqual(scripts,['process_raw_illumina_arrays_aml.py','validate_aml_outputs.py','tune_models.py','main.py','run_xai_explainer.py','run_comparisons.py'])

    def test_layout_moves_files_and_rejects_conflicts_without_overwrite(self):
        gen=self.folders[0]/'generated'
        for name in ['summary.csv','accuracy.pdf','data_aml_PNA.pt']:(gen/name).write_text(name)
        moved=migrate_generated(gen)
        self.assertEqual(len(moved),4)
        self.assertTrue((gen/'data/summary.csv').is_file());self.assertTrue((gen/'plots/accuracy.pdf').is_file())
        self.assertTrue((gen/'models/data_aml_PNA.pt').is_file())
        (gen/'summary.csv').write_text('new')
        with self.assertRaises(FileExistsError):migrate_generated(gen)
        self.assertEqual((gen/'summary.csv').read_text(),'new')
        self.assertEqual(artifact_file(gen,'summary.csv'),gen/'data/summary.csv')

    def test_first_node_uses_id_column_and_empty_file_is_clear(self):
        config=pipeline.prepare_dataset(self.folders[0])
        self.assertEqual(pipeline.first_likely_positive_node(self.folders[0],config),17)
        path=self.folders[0]/'generated/data/pred_likely_pos_aml_PNA.csv'
        path.write_text('test_gene_ids\ttest_gene_names\n')
        with self.assertRaisesRegex(ValueError,'no likely-positive'):
            pipeline.first_likely_positive_node(self.folders[0],config)

    def test_missing_likely_positive_table_is_reconstructed_before_selecting(self):
        folder=self.folders[0];config=pipeline.prepare_dataset(folder)
        path=folder/'generated/data/pred_likely_pos_aml_PNA.csv';path.unlink()
        def reconstruct(*args,**kwargs):
            path.write_text('test_gene_ids\ttest_gene_names\n23\tcg00000001_GENE\n')
        with patch('evaluation.extract_predictions.extract',side_effect=reconstruct) as extract:
            self.assertEqual(pipeline.first_likely_positive_node(folder,config),23)
        extract.assert_called_once()

    def test_relative_folder_ignores_an_outside_folder_with_same_name(self):
        import os
        import dataset_configs
        folder=self.folders[0]
        with tempfile.TemporaryDirectory() as external:
            outside=Path(external)/folder.name;outside.mkdir()
            cwd=Path.cwd()
            try:
                os.chdir(external)
                with patch.object(dataset_configs,'ROOT',self.root):
                    selected=pipeline.selected_folders(self.config,Path(folder.name))
                self.assertEqual(selected,[folder])
                self.assertEqual(list(outside.iterdir()),[])
            finally:os.chdir(cwd)

    def test_stale_snapshot_outputs_are_bound_to_selected_folder(self):
        folder=self.folders[0];path=folder/'config.yaml';cfg=OmegaConf.load(path)
        cfg.p_base=str(self.root/'outside')+'/'
        cfg.p_generated_files=str(self.root/'outside/generated')+'/'
        cfg.p_out_genes=str(self.root/'outside/preprocessed/out_genes_aml.csv')
        cfg.model_overrides.PNA={'p_generated_files':str(self.root/'outside/generated/pna_custom')+'/'}
        OmegaConf.save(cfg,path)
        selected=pipeline.prepare_dataset(folder,dry_run=True)
        self.assertEqual(selected.p_base,str(folder)+'/')
        self.assertEqual(selected.p_out_genes,str(folder/'preprocessed/out_genes_aml.csv'))
        self.assertEqual(selected.p_generated_files,str(folder/'generated')+'/')
        self.assertEqual(selected.model_overrides.PNA.p_generated_files,str(folder/'generated/pna_custom')+'/')
        self.assertFalse((self.root/'outside').exists())

    def test_generated_grid_roots_do_not_depend_on_working_directory(self):
        import os
        from preprocessing_grid import preprocessing_configs,ANALYSIS_ROOT
        source=OmegaConf.create({'corr_threshold':[.33],'knn_neighbours':[500],'size_negative':[5000]})
        cwd=Path.cwd()
        try:
            os.chdir(self.root)
            chosen=preprocessing_configs(source)[0]
        finally:os.chdir(cwd)
        self.assertEqual(Path(chosen.p_base),ANALYSIS_ROOT/'data_corr_0.33_knn_n_500_n_neg_5000')
        self.assertTrue(Path(chosen.p_base).is_absolute())

    def test_exact_train_folder_command_never_expands_parameter_grid(self):
        folder=self.folders[0]
        before=set(self.root.iterdir())
        with patch.object(pipeline,'preprocessing_configs',side_effect=AssertionError('Training must not expand the grid')):
            commands=self.invoke('train',str(folder)+'/')
        self.assertEqual(len(commands),1)
        self.assertEqual(commands[0][commands[0].index('--config')+1],str(folder/'config.yaml'))
        self.assertEqual(set(self.root.iterdir()),before)

    def test_dry_run_does_not_write_or_move(self):
        folder=self.folders[0];before=(folder/'config.yaml').read_bytes()
        with patch.object(pipeline.sys,'argv',['run_pipeline.py','--stage','xai','--config',str(self.source),'--folder',str(folder),'--dry-run']),patch.object(pipeline.subprocess,'run') as execute:
            pipeline.main()
        execute.assert_not_called();self.assertEqual((folder/'config.yaml').read_bytes(),before)
        self.assertTrue((folder/'generated/pred_likely_pos_aml_PNA.csv').is_file())
        self.assertFalse((folder/'generated/data').exists())

if __name__=='__main__':unittest.main()
