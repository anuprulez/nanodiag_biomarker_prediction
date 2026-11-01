"""Guard against label leakage and node misalignment in matched UMAP panels."""
import unittest
import numpy as np
import pandas as pd
from .niapu_umap import aligned_inputs, quality

class UmapAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.features=pd.DataFrame(dict(name=['cg1_A','cg2_B','cg3_C','cg4_D'],
            ring=[1,2,3,5],NetRank=[.1,.8,.3,.6],GSM1=[.9,.4,.2,.6],GSM2=[.8,.3,.1,.5],labels=[1,2,3,4]))
        self.pred=pd.DataFrame(dict(test_gene_ids=[3,1,0],test_gene_names=['cg4_D','cg2_B','cg1_A'],labels=[4,2,1],pred_labels=[3,2,1]))
        self.rankings=pd.DataFrame(dict(node_name=self.features.name,niapu_class=self.features.labels))
    def test_name_alignment_and_balanced_blocks_exclude_labels(self):
        names,y,p,blocks,info=aligned_inputs(self.features,self.pred,self.rankings,['ring','NetRank'])
        self.assertEqual(list(names),['cg4_D','cg2_B','cg1_A'])
        self.assertEqual(list(y),[4,2,1]);self.assertEqual(list(p),[3,2,1])
        self.assertEqual(info[1]['columns'],['GSM1','GSM2'])
        for block in blocks:self.assertAlmostEqual(np.var(block,axis=0).sum(),1)
    def test_stale_graph_ids_are_rejected_even_if_names_exist(self):
        self.pred.loc[0,'test_gene_ids']=2
        with self.assertRaisesRegex(ValueError,'graph IDs'):aligned_inputs(self.features,self.pred,self.rankings,['ring','NetRank'])
    def test_changed_niapu_labels_are_rejected(self):
        self.rankings.loc[1,'niapu_class']=5
        with self.assertRaisesRegex(ValueError,'NIAPU labels differ'):aligned_inputs(self.features,self.pred,self.rankings,['ring','NetRank'])
    def test_labels_cannot_be_a_feature_block_column(self):
        with self.assertRaisesRegex(ValueError,'labels cannot be features'):
            aligned_inputs(self.features,self.pred,self.rankings,['ring','labels'])
    def test_single_predicted_class_reports_undefined_separation(self):
        scores=quality(np.ones((3,2)),np.ones((3,2)),np.ones(3))
        self.assertTrue(all(v is None for v in scores.values()))
