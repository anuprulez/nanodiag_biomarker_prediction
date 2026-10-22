import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from geo_common import matrix_path
from harmonise_geo_datasets import harmonise


class HarmonisationTests(unittest.TestCase):
    def fixture(self, root, series, samples, cells, rows):
        with gzip.open(matrix_path(root, series), 'wt', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['!Sample_geo_accession'] + samples)
            writer.writerow(['!Sample_platform_id'] + ['GPL13534'] * len(samples))
            writer.writerow(['!Sample_source_name_ch1'] + ['Peripheral blood'] * len(samples))
            if cells:
                writer.writerow(['!Sample_characteristics_ch1'] + ['cell type: ' + c for c in cells])
            handle.write('\n')
            writer.writerow(['!series_matrix_table_begin'])
            writer.writerow(['ID_REF'] + samples)
            writer.writerows(rows)
            writer.writerow(['!series_matrix_table_end'])

    def prepare(self, root, bad='NA'):
        self.fixture(root, 'GSE50660', ['GSM1'], None,
                     [['cg2', '.2'], ['cg1', '.1'], ['cg3', bad], ['cg4', '.4']])
        self.fixture(root, 'GSE58477', ['GSM2', 'GSM3'],
                     ['normal bone marrow progenitor cells', 'leukemic blast'],
                     [['cg1', '.9', '.8'], ['cg3', '.9', '.7'], ['cg2', '.9', '.6']])

    def test_alignment_selection_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            harmonise(root, root / 'out')
            with gzip.open(root / 'out/combined_beta.tsv.gz', 'rt') as handle:
                self.assertEqual(list(csv.reader(handle, delimiter='\t')),
                                 [['ID_REF', 'GSM1', 'GSM3'], ['cg2', '.2', '.6'], ['cg1', '.1', '.8']])
            report = json.loads((root / 'out/harmonisation_report.json').read_text())
            self.assertEqual(report['excluded_gse58477_samples'], 1)
            self.assertEqual(report['dropped_missing'], 1)

    def test_reject_out_of_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, '1.2')
            with self.assertRaisesRegex(ValueError, 'Expected beta'):
                harmonise(root, root / 'out')
            self.assertFalse((root / 'out/combined_beta.tsv.gz').exists())

    def test_reject_duplicate_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            self.fixture(root, 'GSE50660', ['GSM1'], None, [['cg1', '.1'], ['cg1', '.2']])
            with self.assertRaisesRegex(ValueError, 'duplicate probe'):
                harmonise(root, root / 'out')


if __name__ == '__main__':
    unittest.main()
