"""Check exclusive-region counts and stable identity handling."""
import tempfile
import unittest
from pathlib import Path
from evaluation.compare_data_overlap import membership_regions, load_splits


class OverlapTests(unittest.TestCase):
    def test_regions_count_union_once(self):
        regions = membership_regions({'train': {'a', 'b', 'c'},
                                      'test': {'b', 'c', 'd'}, 'val': {'c', 'e'}})
        self.assertEqual(sum(regions.values()), 5)
        self.assertEqual(regions[('train', 'test', 'val')], 1)
        self.assertEqual(regions[('train', 'test')], 1)
        self.assertEqual(membership_regions({'train': set()}), {})

    def test_local_numeric_ids_are_not_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); (folder / 'preprocessed').mkdir()
            for split, feature in [('train', 'cg1_GENE'), ('test', 'cg2_GENE'), ('val', 'cg3_OTHER')]:
                (folder / 'preprocessed' / f'out_genes_aml_{split}.csv').write_text(f'0 {feature} 0.5\n')
            splits = load_splits(folder)
            self.assertFalse(splits['train'] & splits['test'])
            self.assertEqual(splits['train'], {'cg1_GENE'})


if __name__ == '__main__':
    unittest.main()
