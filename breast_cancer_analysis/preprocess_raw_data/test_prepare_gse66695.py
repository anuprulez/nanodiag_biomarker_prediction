"""Check strict sample labeling and GSE66695 preparation compatibility."""
import csv,gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from prepare_gse66695 import prepare,select_samples

class Gse66695PreparationTests(unittest.TestCase):
    def test_group_selection_and_metadata_disagreement(self):
        samples=['GSM1','GSM2','GSM3','GSM4']
        meta={'!Sample_platform_id':[['GPL13534']*4],
              '!Sample_source_name_ch1':[['Normal','Tumor','Normal','Tumor']],
              '!Sample_characteristics_ch1':[['disease state: '+s for s in ['Normal','Tumor','Normal','Tumor']]]}
        indices,aliases,codes=select_samples(samples,meta,expected=(2,2))
        self.assertEqual(indices,[1,3,0,2]);self.assertEqual(aliases,['BC_GSM2','BC_GSM4','NC_GSM1','NC_GSM3'])
        meta['!Sample_source_name_ch1'][0][0]='Tumor'
        with self.assertRaisesRegex(ValueError,'do not agree'):select_samples(samples,meta,expected=(2,2))

    def test_beta_roundtrip_and_new_seed_effects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.gz';out=root/'prepared'
            samples=['GSM1','GSM2','GSM3','GSM4'];labels=['Normal','Tumor','Normal','Tumor']
            with gzip.open(source,'wt') as handle:
                writer=csv.writer(handle,delimiter='\t',lineterminator='\n')
                for key,values in [('!Sample_geo_accession',samples),('!Sample_title',samples),
                    ('!Sample_platform_id',['GPL13534']*4),('!Sample_source_name_ch1',labels),
                    ('!Sample_characteristics_ch1',['disease state: '+s for s in labels])]:writer.writerow([key,*values])
                writer.writerow(['!series_matrix_table_begin']);writer.writerow(['ID_REF',*samples])
                writer.writerow(['cg00000001','.1','.9','.2','.8'])
                writer.writerow(['cg00000002','.3','NA','.2','.9'])
                writer.writerow(['rs123','.1','.9','.2','.8'])
                writer.writerow(['!series_matrix_table_end'])
            ann=root/'ann.tsv';pd.DataFrame({'ID':['cg00000001','cg00000002'],'UCSC_RefGene_Name':['A;A;B','C'],
                'CHR':['1','2'],'MAPINFO':['100','200']}).to_csv(ann,sep='\t',index=False)
            seeds=root/'seeds.tsv';pd.DataFrame({'probeID':['cg00000001','cg00000002']}).to_csv(seeds,sep='\t',index=False)
            report=prepare(source,out,ann,seeds,expected=(2,2))
            matrix=pd.read_csv(out/'GSE66695_matrix_processed.txt.gz',sep='\t')
            self.assertEqual(matrix.columns.tolist(),['ID_REF','BC_GSM2','BC_GSM4','NC_GSM1','NC_GSM3'])
            self.assertTrue(np.allclose(matrix.iloc[0,1:].astype(float),[.9,.8,.1,.2]))
            self.assertEqual(len(matrix),1)
            mapper=pd.read_csv(out/'probe_genes.csv',sep='\t');seed=pd.read_csv(out/'seed_probes.csv',sep='\t')
            self.assertEqual(mapper.UCSC_RefGene_Name.iloc[0],'A;B')
            self.assertAlmostEqual(seed.deltaBeta.iloc[0],.7)
            self.assertEqual(report['counts']['incomplete_cpg_rows_excluded'],1)
            self.assertEqual(report['normal_samples'],2)

if __name__=='__main__':unittest.main()
