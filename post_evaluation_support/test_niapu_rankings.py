"""NIAPU source isolation, deduplication and exact criterion boundaries."""
from pathlib import Path
import tempfile,unittest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from .niapu_rankings import load_rankings,select_candidates,resolve_config

class NIAPUSelectionTests(unittest.TestCase):
    def test_only_class_two_nodes_and_unique_cpgs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rankings.csv'
            p.write_text('cg00000001_SEED 0.99 1\ncg00000002_A 0.2 2\ncg00000002_B 0.3 2\ncg00000003_OTHER 0.1 3\ncg00000004_NEG 0.01 5\n')
            all_nodes,lp,cpgs=load_rankings(p)
        self.assertEqual(len(all_nodes),5);self.assertEqual(len(lp),2)
        self.assertEqual(cpgs.CpG_ID.tolist(),['cg00000002'])
        self.assertEqual(cpgs.iloc[0].niapu_score,.3)
        self.assertEqual(cpgs.iloc[0].annotated_genes,'A;B')

    def test_criterion_inclusive_effect_strict_p_fdr_and_direction(self):
        f=pd.DataFrame({'internal':[.2]*7+[-.1,0,np.nan],
                        'external':[.30,.299,.4,.4,-.4,.4,.4,-.4,.4,.4],
                        'External_p_value':[.009,.009,.01,.009,.009,.009,np.nan,.009,.009,.009],
                        'External_FDR':[.009,.009,.009,.01,.009,np.nan,.009,.009,.009,.009]})
        result=select_candidates(f,'internal','external')
        self.assertEqual(result.Meets_exploratory_criterion.tolist(),[True,False,False,False,False,False,False,True,False,False])

    def test_preprocessing_only_paths_resolve_without_tuned_config(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'aml_analysis';root.mkdir()
            config=OmegaConf.create({'p_base':'../data_custom/','p_arrays':'../raw/beta.tsv.gz',
                                    'p_sample_metadata':'../raw/metadata.tsv',
                                    'p_out_gene_rankings':'${p_base}preprocessed/rankings.csv'})
            result=resolve_config(root,config)
            self.assertEqual(result.p_base,str(root/'data_custom')+'/')
            self.assertEqual(result.p_arrays,str(root/'raw/beta.tsv.gz'))
            self.assertEqual(result.p_out_gene_rankings,str(root/'data_custom/preprocessed/rankings.csv'))

    def test_legacy_preprocessing_config_inherits_shared_raw_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'analysis';(root/'config').mkdir(parents=True)
            OmegaConf.save(OmegaConf.create({'p_raw':'../raw/',
                'p_arrays':'${p_raw}beta.tsv.gz','p_sample_metadata':'${p_raw}meta.tsv'}),root/'config/config.yaml')
            c=OmegaConf.create({'p_base':'../data_custom/',
                               'p_out_gene_rankings':'${p_base}preprocessed/rankings.csv'})
            result=resolve_config(root,c)
            self.assertEqual(result.p_arrays,str(root/'raw/beta.tsv.gz'))
            self.assertEqual(result.p_sample_metadata,str(root/'raw/meta.tsv'))

    def test_invalid_or_duplicate_nodes_fail(self):
        for text in ['cg00000001_A nan 2\n','cg00000001_A 0.2 6\n','gene 0.2 2\n','cg00000001_A 0.2 2\ncg00000001_A 0.3 2\n']:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'r';p.write_text(text)
                with self.assertRaises(ValueError):load_rankings(p)

if __name__=='__main__':unittest.main()
