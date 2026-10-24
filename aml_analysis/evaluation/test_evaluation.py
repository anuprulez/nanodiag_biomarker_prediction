"""AML prediction alignment, split artifacts and metadata-aware comparisons."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from extract_predictions import extract,comparison_groups,prediction_frame
from compare_predictions import internal_statistics,EXTERNAL_STATS

class AmlEvaluationTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data_corr_0.33_knn_n_3000_n_neg_10000';prep=data/'preprocessed';gen=data/'generated'
        prep.mkdir(parents=True);gen.mkdir()
        names=['cg00000001_A','cg00000002_B','cg00000003_C','cg00000004_D']
        pd.DataFrame({0:range(4),1:names,2:[.1,.1,0.,0.]}).to_csv(prep/'out_genes_aml.csv',sep=' ',header=False,index=False)
        signals=pd.DataFrame({name:[.1,.8,.3,.6] for name in names})
        signals.to_csv(prep/'combined_pos_neg_signals_aml.csv',sep='\t',index=False)
        metadata=root/'metadata.tsv'
        pd.DataFrame({'sample_id':['H1','A1','H2','A2'],'group':['healthy_control','AML','healthy_control','AML']}).to_csv(metadata,sep='\t',index=False)
        matrix=root/'combined_beta_cpg_gene.tsv.gz'
        with gzip.open(matrix,'wt') as target:target.write('CpG_Gene\tH1\tA1\tH2\tA2\n')
        ids=[1,0,3,2]
        for prefix,values in [('test_ids',ids),('true_labels',[2,1,4,3]),('pred_labels',[1,1,4,4]),('best_class_pred_probs',[.9,.8,.7,.6])]:
            torch.save(values,gen/f'{prefix}_aml_PNA.pt')
        pd.DataFrame({'test_gene_ids':range(4),'test_gene_names':names}).to_csv(gen/'test_probe_genes_aml_PNA.csv',index=False)
        rows=[]
        for name in names:
            probe,gene=name.split('_',1)
            rows.append(dict(CpG_ID=probe,Gene=gene,Healthy_GSE40279_mean_beta=.2,Healthy_GSE40279_SD=.1,
                Healthy_GSE40279_n=10,AML_GSE62298_mean_beta=.7,AML_GSE62298_SD=.1,AML_GSE62298_n=8,Delta_beta_AML_minus_healthy=.5))
        external=root/'external.tsv';pd.DataFrame(rows).to_csv(external,sep='\t',index=False)
        pd.DataFrame(rows).drop(columns='Gene').to_csv(root/'shared_cpg_statistics.tsv',sep='\t',index=False)
        return data,metadata,matrix,external

    def test_inference_ids_align_names_with_repeated_seed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,metadata,matrix,external=self.fixture(root)
            groups=extract(data,'PNA',root/'extracted')
            self.assertEqual(groups['likely_positive'].test_gene_ids.tolist(),[1,0])
            self.assertEqual(groups['likely_positive'].test_gene_names.tolist(),['cg00000002_B','cg00000001_A'])
            self.assertEqual(groups['negative'].test_gene_ids.tolist(),[3,2])

    def test_comparison_preserves_custom_xai_exported_groups(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,metadata,matrix,external=self.fixture(root);generated=data/'generated'
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
            root=Path(d);data,metadata,matrix,external=self.fixture(root);output=data/'generated/data'
            extract(data,'PNA',output)
            path=output/'pred_negatives_PNA.csv';frame=pd.read_csv(path,sep='\t')
            frame.loc[0,'pred_labels']=1;frame.to_csv(path,sep='\t',index=False)
            with self.assertRaisesRegex(ValueError,'differs from current predictions'):
                comparison_groups(data,'PNA',root/'comparison')

    def test_comparison_reconstructs_only_when_exports_are_absent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,metadata,matrix,external=self.fixture(root)
            groups,paths=comparison_groups(data,'PNA',root/'comparison')
            self.assertEqual(paths,[]);self.assertEqual(groups['negative'].test_gene_ids.tolist(),[3,2])

    def test_split_layout_is_readable(self):
        from artifact_layout import migrate_generated
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,metadata,matrix,external=self.fixture(root)
            migrate_generated(data/'generated')
            groups=extract(data,'PNA',root/'extracted')
            self.assertEqual(groups['likely_positive'].test_gene_ids.tolist(),[1,0])

    def test_internal_effect_uses_metadata_groups_in_original_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,metadata,matrix,external=self.fixture(root)
            groups=extract(data,'PNA',root/'extracted')
            stats=internal_statistics(list(groups.values()),data/'preprocessed/combined_pos_neg_signals_aml.csv',metadata,matrix)
            np.testing.assert_allclose(stats.Internal_delta_beta_AML_minus_healthy,.5)
            self.assertTrue((stats.Internal_healthy_n==2).all());self.assertTrue((stats.Internal_AML_n==2).all())
            pd.read_csv(metadata,sep='\t').iloc[::-1].to_csv(metadata,sep='\t',index=False)
            with self.assertRaisesRegex(ValueError,'sample order'):
                internal_statistics(list(groups.values()),data/'preprocessed/combined_pos_neg_signals_aml.csv',metadata,matrix)

    def test_complete_comparison_writes_inside_dataset_evaluation(self):
        from artifact_layout import migrate_generated
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,metadata,matrix,external=self.fixture(root);migrate_generated(data/'generated')
            command=[sys.executable,str(Path(__file__).parent/'compare_predictions.py'),'--data',str(data),
                '--metadata',str(metadata),'--matrix',str(matrix),'--external',str(external),'--model','PNA']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            reports=list((data/'evaluation').glob('analysis_report*.json'));self.assertEqual(len(reports),1)
            report=json.loads(reports[0].read_text());self.assertEqual(report['model'],'PNA')
            self.assertEqual(report['original_samples'],{'healthy':2,'AML':2})
            self.assertTrue(list((data/'evaluation').glob('*.pdf')))
            self.assertEqual(report['external_cohorts'],{'healthy':'GSE40279','AML':'GSE62298'})
            self.assertEqual(len(list((data/'evaluation').glob('*.png'))),3)
            self.assertEqual([g['prediction_group'] for g in report['groups']],['likely_positive','negative'])
            self.assertEqual(report['external_inference']['FDR_family_size'],4)

    def test_runner_passes_dataset_metadata_matrix_and_artifact_paths(self):
        import run_comparisons
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'config.yaml'
            OmegaConf.save(OmegaConf.create({'config_scope':'dataset','p_base':str(root)+'/',
                'p_arrays':str(root/'matrix.gz'),'p_sample_metadata':str(root/'metadata.tsv'),
                'p_generated_files':str(root/'generated')+'/','best_trained_model':'PNA','model_type':['PNA'],'model_overrides':{}}),path)
            cwd=Path.cwd()
            try:
                with patch.object(sys,'argv',['run_comparisons.py','--config',str(path)]),patch.object(run_comparisons.subprocess,'run') as execute:
                    run_comparisons.main()
            finally:os.chdir(cwd)
            command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--metadata')+1],str(root/'metadata.tsv'))
            self.assertEqual(command[command.index('--output')+1],str(root/'evaluation'))

if __name__=='__main__':unittest.main()
