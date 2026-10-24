"""Fixtures for sample alignment, beta parsing, gene expansion and later comparisons."""
import io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import os
import numpy as np
import pandas as pd
import torch
from geo_common import statistics,tumor_columns,beta_rows
from evaluate_datasets import write_results,DELTA
from extract_predictions import extract,comparison_groups,prediction_frame
from compare_predictions import internal_statistics,configured_matrix,ROOT

class EvaluationTests(unittest.TestCase):
    def test_comparison_runner_uses_dataset_model_artifact_directory(self):
        import run_comparisons
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'config.yaml'
            OmegaConf.save(OmegaConf.create({'config_scope':'dataset','p_base':str(root)+'/',
                'p_arrays':str(root/'matrix.gz'),'model_type':['PNA'],'best_trained_model':'PNA',
                'p_generated_files':str(root/'generated')+'/',
                'model_overrides':{'PNA':{'p_generated_files':str(root/'pna_artifacts')+'/'}}}),config)
            cwd=Path.cwd()
            try:
                with patch.object(sys,'argv',['run_comparisons.py','--config',str(config)]), patch.object(run_comparisons.subprocess,'run') as execute:
                    run_comparisons.main()
            finally:os.chdir(cwd)
            command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--config')+1],str(config))
            self.assertEqual(command[command.index('--generated-dir')+1],str(root/'pna_artifacts'))
            self.assertEqual(command[command.index('--matrix')+1],str(root/'matrix.gz'))

    def test_matrix_uses_config_and_pipeline_relative_base(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'config.yaml'
            config.write_text('p_raw: ../raw/\np_arrays: ${p_raw}GSE66695_matrix_processed.txt.gz\n')
            self.assertEqual(configured_matrix(config),(ROOT/'raw/GSE66695_matrix_processed.txt.gz').resolve())
            config.write_text('p_arrays: /tmp/custom_matrix.txt.gz\n')
            self.assertEqual(configured_matrix(config),Path('/tmp/custom_matrix.txt.gz'))

    def test_detection_columns_and_sample_order(self):
        self.assertEqual(tumor_columns(['','B','B.DetectionPvalue','A','A.DetectionPvalue'],['A','B']),[3,1])
        with self.assertRaises(ValueError):tumor_columns(['','A','B'],['A'])

    def test_missing_invalid_betas(self):
        mean,sd,n=statistics(['0','1','NA'])
        self.assertEqual((mean,n),(.5,2));self.assertAlmostEqual(sd,2**-.5)
        self.assertEqual(statistics(['NA']),(None,None,0))
        for value in ['1.2','inf','-0.1','garbage']:
            with self.assertRaises(ValueError):statistics([value])

    def test_truncated_and_duplicate_rows(self):
        for text in ['cg1\t0.1\n','cg1\t0.1\ncg1\t0.2\n!series_matrix_table_end\n']:
            with self.assertRaises(ValueError):list(beta_rows(io.StringIO(text),1))

    def test_common_sites_expanded_genes_and_delta(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            report=write_results({'cg1':(.2,.1,3),'cg2':(.5,.1,3),'cg3':(.6,.1,3)},
                                 {'cg1':(.8,.1,4),'cg2':(.4,.1,4),'cg4':(.3,.1,4)},
                                 {'cg1':{'genes':['A','B']}},root)
            frame=pd.read_csv(root/'cpg_gene_beta_comparison.tsv',sep='\t')
            self.assertEqual(frame['CpG_Gene'].tolist(),['cg1_A','cg1_B'])
            self.assertTrue(np.allclose(frame[DELTA],.6))
            self.assertEqual((report['shared_cpgs'],report['shared_unannotated_cpgs']),(2,1))
            self.assertEqual(len(pd.read_csv(root/'GSE101961_cpg_gene_statistics.tsv',sep='\t')),4)

    def fixture(self,root):
        data=root/'data_corr_0.33_knn_n_3000_n_neg_10000'
        gen=data/'generated';prep=data/'preprocessed';gen.mkdir(parents=True);prep.mkdir()
        names=[f'cg{i:08d}_GENE' for i in range(4)]
        pd.DataFrame({0:range(4),1:names}).to_csv(prep/'out_genes_bc.csv',sep=' ',header=False,index=False)
        for prefix,values in [('test_ids',[2,0,3,1]),('true_labels',[2,1,4,3]),('pred_labels',[2,1,4,1]),('best_class_pred_probs',[.8]*4)]:
            torch.save(values,gen/f'{prefix}_bc_PNA.pt')
        pd.DataFrame({'test_gene_ids':range(4),'test_gene_names':names}).to_csv(gen/'test_probe_genes_bc_PNA.csv',index=False)
        signals=np.array([[.8,.6,.4,.2],[.7,.5,.3,.1],[.2,.2,.2,.2],[.1,.1,.1,.1]])
        pd.DataFrame(signals,columns=names).to_csv(prep/'combined_pos_neg_signals_bc.csv',sep='\t',index=False)
        matrix=root/'matrix.tsv'
        raw=pd.DataFrame(signals.T,columns=['BC_1','BC_2','NC_1','Nc_2']);raw.insert(0,'ID_REF',[n.split('_')[0] for n in names]);raw.to_csv(matrix,sep='\t',index=False)
        out=root/'external';out.mkdir()
        h={n.split('_')[0]:(.2,.1,3) for n in names};c={n.split('_')[0]:(.3+i*.1,.1,4) for i,n in enumerate(names)}
        write_results(h,c,{n.split('_')[0]:{'genes':['GENE','OTHER']} for n in names},out)
        return data,matrix,out

    def test_inference_alignment_and_group_rules(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root)
            groups=extract(data,'PNA',root/'extracted')
            self.assertEqual(groups['negative']['test_gene_ids'].tolist(),[2,3])
            self.assertEqual(groups['likely_positive']['test_gene_ids'].tolist(),[0,1])
            self.assertEqual(len(internal_statistics(list(groups.values()),data/'preprocessed/combined_pos_neg_signals_bc.csv',matrix)),4)

    def test_split_layout_preserves_inference_alignment(self):
        from artifact_layout import migrate_generated
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root)
            migrate_generated(data/'generated')
            groups=extract(data,'PNA',root/'extracted')
            self.assertEqual(groups['negative']['test_gene_ids'].tolist(),[2,3])
            self.assertEqual(groups['likely_positive']['test_gene_ids'].tolist(),[0,1])
            self.assertTrue((data/'generated/models/test_ids_bc_PNA.pt').is_file())
            self.assertTrue((data/'generated/data/test_probe_genes_bc_PNA.csv').is_file())

    def test_comparison_preserves_custom_xai_exported_groups(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root);generated=data/'generated'
            output=generated/'data'
            extract(data,'PNA',output)
            frame=prediction_frame(data,'PNA')
            # The user's edited XAI rule: labels 2-5 and predicted labels 4-5.
            custom=frame.loc[frame.labels.isin([2,3,4,5])&frame.pred_labels.isin([4,5])]
            source=output/'pred_negatives_PNA.csv'
            custom=custom.copy();custom['pred_probs']=custom.pred_probs.round(8)
            custom.to_csv(source,sep='\t',index=False)
            original=source.read_bytes()
            groups,paths=comparison_groups(data,'PNA',root/'comparison')
            self.assertEqual(groups['negative'].test_gene_ids.tolist(),custom.test_gene_ids.tolist())
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual((root/'comparison'/source.name).read_bytes(),original)
            self.assertIn(source,paths)

    def test_comparison_rejects_misaligned_exported_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root);output=data/'generated/data'
            extract(data,'PNA',output)
            path=output/'pred_negatives_PNA.csv';frame=pd.read_csv(path,sep='\t')
            frame.loc[0,'pred_labels']=1;frame.to_csv(path,sep='\t',index=False)
            with self.assertRaisesRegex(ValueError,'differs from current predictions'):
                comparison_groups(data,'PNA',root/'comparison')

    def test_comparison_reconstructs_only_when_exports_are_absent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root)
            groups,paths=comparison_groups(data,'PNA',root/'comparison')
            self.assertEqual(paths,[]);self.assertEqual(groups['negative'].test_gene_ids.tolist(),[2,3])

    def test_three_column_nedbit_graph_with_repeated_seed_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root)
            path=data/'preprocessed/out_genes_bc.csv'
            nodes=pd.read_csv(path,sep=r'\s+',header=None)
            nodes[2]=[0.,0.,.1,0.]
            nodes.to_csv(path,sep=' ',header=False,index=False)
            groups=extract(data,'PNA',root/'extracted')
            self.assertEqual(groups['negative']['test_gene_ids'].tolist(),[2,3])
            self.assertEqual(groups['likely_positive']['test_gene_names'].tolist(),['cg00000000_GENE','cg00000001_GENE'])
            nodes.loc[1,0]=0
            nodes.to_csv(path,sep=' ',header=False,index=False)
            with self.assertRaisesRegex(ValueError,'unique contiguous'):
                extract(data,'PNA',root/'bad')

    def test_permuted_signal_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root);groups=extract(data,'PNA',root/'extracted')
            p=data/'preprocessed/combined_pos_neg_signals_bc.csv'
            pd.read_csv(p,sep='\t').iloc[::-1].to_csv(p,sep='\t',index=False)
            with self.assertRaisesRegex(ValueError,'sample order'):
                internal_statistics(list(groups.values()),p,matrix)

    def test_full_comparison_without_training_or_xai(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,matrix,external=self.fixture(root);out=root/'comparison'
            import gzip
            compressed=root/'GSE66695_matrix_processed.txt.gz'
            with gzip.open(compressed,'wt') as handle:handle.write(matrix.read_text())
            matrix=compressed
            # Completed improved artifacts must take precedence over old PNA files.
            import shutil
            generated=data/'generated';improved=generated/'pna_accuracy_improved';improved.mkdir()
            for artifact in list(generated.iterdir()):
                if artifact.is_file():shutil.move(str(artifact),improved/artifact.name)
            (improved/'improvement_report.json').write_text('{}')
            result=subprocess.run([sys.executable,str(Path(__file__).parent/'compare_predictions.py'),'--data',str(data),
                                   '--matrix',str(matrix),'--external',str(external/'cpg_gene_beta_comparison.tsv'),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads(next(out.glob('analysis_report*.json')).read_text())
            self.assertEqual(report['original_samples'],{'healthy':2,'BC':2})
            self.assertEqual(report['model_artifact_directory'],str(improved))
            self.assertEqual(report['training_series'],'GSE66695')
            self.assertEqual(report['cohort_platforms']['GSE66695'],'GPL13534 (450K)')
            self.assertTrue(all(row['common_unique_cpgs']==2 for row in report['groups']))
            self.assertEqual(len(list(out.glob('*.png'))),2)

if __name__=='__main__':unittest.main()
