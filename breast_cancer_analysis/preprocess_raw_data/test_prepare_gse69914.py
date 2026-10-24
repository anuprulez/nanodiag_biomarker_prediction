"""Fixtures for strict cohort selection and preserving preprocessing input values."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from prepare_gse69914 import prepare,select_samples,differential
from preprocessing_grid import preprocessing_configs

class Gse69914PreparationTests(unittest.TestCase):
    def test_selection_uses_status_and_excludes_adjacent_and_brca(self):
        samples=[f'GSM{i}' for i in range(7)]
        meta={'!Sample_platform_id':[['GPL16304']*7],
              '!Sample_characteristics_ch1':[['status(0=normal 1=normal-adjacent 2=breast cancer 3=normal-brca1 4=cancer-brca1): '+str(i) for i in [2,1,0,3,2,0,4]]]}
        indices,aliases,codes=select_samples(samples,meta,expected=(2,2))
        self.assertEqual(indices,[0,4,2,5])
        self.assertEqual(aliases,['BC_GSM0','BC_GSM4','NC_GSM2','NC_GSM5'])
        with self.assertRaises(ValueError):select_samples(samples,meta)

    def test_output_roundtrip_and_missing_selected_sample_policy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.gz';out=root/'ready'
            samples=[f'GSM{i}' for i in range(7)]
            with gzip.open(source,'wt') as handle:
                writer=csv.writer(handle,delimiter='\t',lineterminator='\n')
                for name,values in [('!Sample_geo_accession',samples),('!Sample_platform_id',['GPL16304']*7),
                        ('!Sample_title',samples),('!Sample_source_name_ch1',['breast']*7),
                        ('!Sample_characteristics_ch1',['status(0=normal 1=normal-adjacent 2=breast cancer 3=normal-brca1 4=cancer-brca1): '+str(i) for i in [2,1,0,3,2,0,4]])]:
                    writer.writerow([name,*values])
                writer.writerow(['!series_matrix_table_begin']);writer.writerow(['ID_REF',*samples])
                writer.writerow(['cg00000001','.9','.5','.1','.4','.8','.2','.7'])
                writer.writerow(['cg00000002','.8','NA','.2','NA','.7','.1','NA'])
                writer.writerow(['cg00000003','NA','.5','.1','.4','.8','.2','.7'])
                writer.writerow(['rs123','.9','.5','.1','.4','.8','.2','.7'])
                writer.writerow(['!series_matrix_table_end'])
            annotation=root/'annotation.tsv'
            pd.DataFrame({'ID':[f'cg{i:08d}' for i in range(1,4)],'UCSC_RefGene_Name':['A;A;B','C','D'],
                          'CHR':['1']*3,'MAPINFO':['1','2','3']}).to_csv(annotation,sep='\t',index=False)
            seeds=root/'seeds.tsv';pd.DataFrame({'probeID':['cg00000001','cg00000003']}).to_csv(seeds,sep='\t',index=False)
            report=prepare(source,out,annotation,seeds,expected=(2,2))
            matrix=pd.read_csv(out/'GSE69914_matrix_processed.txt.gz',sep='\t')
            self.assertEqual(matrix.columns.tolist(),['ID_REF','BC_GSM0','BC_GSM4','NC_GSM2','NC_GSM5'])
            self.assertEqual(matrix.ID_REF.tolist(),['cg00000001','cg00000002'])
            self.assertTrue(np.allclose(matrix.iloc[0,1:].astype(float),[.9,.8,.1,.2]))
            mapping=pd.read_csv(out/'probe_genes.csv',sep='\t')
            self.assertEqual(mapping.iloc[0].UCSC_RefGene_Name,'A;B')
            self.assertAlmostEqual(mapping.iloc[0].deltaBeta,.7)
            seed=pd.read_csv(out/'seed_probes.csv',sep='\t')
            self.assertEqual(seed.probeID.tolist(),['cg00000001'])
            self.assertAlmostEqual(seed.deltaBeta.iloc[0],.7)
            self.assertEqual(report['counts']['incomplete_cpg_rows_excluded'],1)
            self.assertEqual(report['counts']['non_cpg_rows_excluded'],1)

    def test_dataset_root_keeps_old_and_new_preprocessing_outputs_separate(self):
        old=OmegaConf.create({'corr_threshold':.33,'knn_neighbours':3000,'size_negative':10000})
        new=OmegaConf.merge(old,{'p_dataset_root':'../datasets/GSE69914/'})
        self.assertEqual(Path(preprocessing_configs(old)[0].p_base),Path(__file__).resolve().parents[1]/'data_corr_0.33_knn_n_3000_n_neg_10000')
        self.assertEqual(Path(preprocessing_configs(new)[0].p_base),Path(__file__).resolve().parents[1]/'datasets/GSE69914/data_corr_0.33_knn_n_3000_n_neg_10000')

if __name__=='__main__':unittest.main()
