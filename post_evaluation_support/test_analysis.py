"""Fixed selection, patient isolation, and annotation-row identity checks."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from .analysis import load_panel, read_original, oof_predictions, bh, estimator, annotation_and_pathways, load_common_prediction_panels

class PostEvaluationTests(unittest.TestCase):
    def test_fixed_panel_selection_and_expected_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'data_example';out=base/'accumulated_results/evaluation';out.mkdir(parents=True)
            frame=pd.DataFrame({'CpG_ID':['cg1','cg2','cg3','cg4','cg5'],'n_runs':[5,5,5,4,5],
                                'Internal_delta_beta_AML_minus_healthy':[.2]*5,
                                'Delta_beta_AML_minus_healthy':[.30,.30,.30,.30,-.30],
                                'External_p_value':[.001,.01,.001,.001,.001],
                                'External_FDR':[.001,.001,.01,.001,.001]})
            frame.to_csv(out/'likely_positive_common_unique_cpgs_PNA_example.tsv',sep='\t',index=False)
            panel,*_=load_panel(base,'AML',1)
            self.assertEqual(panel.CpG_ID.tolist(),['cg1'])
            with self.assertRaisesRegex(ValueError,'expected fixed 17'):load_panel(base,'AML',17)

    def test_full_common_group_panels_do_not_apply_effect_or_significance_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'data_example';out=base/'accumulated_results/evaluation';out.mkdir(parents=True)
            for group,size in [('likely_positive',189),('negative',70)]:
                frame=pd.DataFrame({'CpG_ID':[f'cg{group}{i}' for i in range(size)],
                                    'n_runs':5,'Internal_delta_beta_BC_minus_healthy':.001,
                                    'Delta_beta_BC_minus_healthy':-.001,
                                    'External_p_value':.9,'External_FDR':.9})
                frame.to_csv(out/f'{group}_common_unique_cpgs_PNA_example.tsv',sep='\t',index=False)
            panels,sources=load_common_prediction_panels(base,'BC')
            self.assertEqual(len(panels['all_common_likely_positive']),189)
            self.assertEqual(len(panels['all_common_predicted_negative']),70)
            self.assertEqual(sources['all_common_likely_positive']['n_features'],189)
            path=out/'negative_common_unique_cpgs_PNA_example.tsv'
            table=pd.read_csv(path,sep='\t');table.loc[0,'n_runs']=4;table.to_csv(path,sep='\t',index=False)
            with self.assertRaisesRegex(ValueError,'five runs'):
                load_common_prediction_panels(base,'BC')

    def test_duplicate_gene_annotations_do_not_duplicate_cpgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'matrix.tsv'
            path.write_text('ID\tpatient1\tpatient2\ncg1_A\t0.1\t0.2\ncg1_B\t0.1\t0.2\ncg2_X\t0.3\t0.4\n')
            matrix=read_original(path,{'cg1'})
            self.assertEqual(matrix.shape,(2,1));self.assertEqual(matrix.index.tolist(),['patient1','patient2'])
            path.write_text(path.read_text().replace('cg1_B\t0.1','cg1_B\t0.9'))
            with self.assertRaisesRegex(ValueError,'Conflicting original'):read_original(path,{'cg1'})

    def test_ranking_and_imputation_use_training_patients_only(self):
        frame=pd.DataFrame({'training_signal':[0.,.1,.9,1.,np.nan,.2,.8,1.],
                            'heldout_signal':[.5,.5,.5,.5,0,0,1,1]})
        y=np.array([0,0,1,1,0,0,1,1]);train=np.array([0,1,2,3]);test=np.array([4,5,6,7]);splits=[(train,test),(test,train)]
        probabilities,selections=oof_predictions(frame,y,list(frame),splits,top_k=1)
        self.assertEqual(selections[0]['CpG_ID'],'training_signal')
        fitted=estimator().fit(frame.iloc[train][['training_signal']],y[train])
        np.testing.assert_allclose(probabilities[test],fitted.predict_proba(frame.iloc[test][['training_signal']])[:,1])
        modified=frame.copy();modified.loc[test,'heldout_signal']=[-100,-100,100,100]
        _,changed=oof_predictions(modified,y,list(frame),splits,top_k=1)
        self.assertEqual(changed[0]['CpG_ID'],'training_signal')

    def test_pathway_background_and_zero_overlap_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'aml_analysis';(root/'raw').mkdir(parents=True)
            out=Path(tmp)/'output'
            for name in ('data','plots'):(out/name).mkdir(parents=True)
            ids=[f'cg{i}' for i in range(30)]
            pd.DataFrame({'ID':ids,'UCSC_RefGene_Name':[f'G{i}' for i in range(30)],
                          'CHR':'1','MAPINFO':range(30)}).to_csv(root/'raw/GPL13534_probe_annotation.tsv',sep='\t',index=False)
            gmt=Path(tmp)/'test.gmt'
            gmt.write_text('hit\ttest\t'+'\t'.join(f'G{i}' for i in range(10))+'\n'+
                           'no_hit\ttest\t'+'\t'.join(f'G{i}' for i in range(10,20))+'\n')
            status=annotation_and_pathways(root,pd.DataFrame({'CpG_ID':ids[:3]}),ids,out,gmt,offline=True)
            result=pd.read_csv(out/'data/reactome_CpG_set_enrichment.tsv',sep='\t').set_index('pathway')
            self.assertEqual(status['pathways']['tested'],2)
            self.assertEqual(result.loc['no_hit','overlap_CpGs'],0)
            self.assertEqual(result.loc['no_hit','p_value'],1.)
            self.assertAlmostEqual(result.loc['hit','FDR'],2*result.loc['hit','p_value'])

    def test_bh_includes_zero_hit_pathways(self):
        np.testing.assert_allclose(bh([.01,.04,1.]),[.03,.06,1.])

if __name__=='__main__':unittest.main()
