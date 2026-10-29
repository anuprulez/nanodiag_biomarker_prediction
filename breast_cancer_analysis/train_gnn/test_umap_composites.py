"""Regression tests for archived five-run UMAP selection and PDF composition."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
from umap_composites import select_umap_pairs, plot_umap_run_composites, publication_umap_page


class CompositeTests(unittest.TestCase):
    def sources(self, base):
        runs=[]
        for index in (5, 2, 4, 1, 3):
            run=base/'runs'/f'run{index}'
            folder=run/'generated'/'plots'
            folder.mkdir(parents=True)
            for name in ('UMAP_nedbit_dnam_features_test_before_GNN_PNA.pdf',
                         'UMAP_node_embeddings_20_0.8_PNAConv4_PNA.pdf'):
                with pymupdf.open() as doc:
                    page=doc.new_page(width=400, height=300)
                    page.insert_text((30, 40), f'Source run {index}')
                    doc.save(folder/name)
            runs.append(run)
        return runs

    def test_numeric_order_and_missing_run(self):
        with tempfile.TemporaryDirectory() as temp:
            runs=self.sources(Path(temp))
            self.assertEqual([r[0] for r in select_umap_pairs(runs)],
                             [f'run{i}' for i in range(1, 6)])
            with self.assertWarns(UserWarning):
                self.assertEqual(select_umap_pairs(runs[:4]), [])
            next((runs[0]/'generated').rglob('*PNAConv4*')).unlink()
            with self.assertWarns(UserWarning):
                self.assertEqual(select_umap_pairs(runs), [])

    def test_tick_scaling_preserves_minus_signs(self):
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter([-1, 0, 1], [-1, 0, 1])
        ax.tick_params(labelsize=16.5)
        ax.set_title('Embeddings UMAP of last GNN layer: PNA', fontsize=18)
        ax.set_xlabel('UMAP1', fontsize=18)
        ax.set_ylabel('UMAP2', fontsize=18)
        buffer=io.BytesIO()
        fig.savefig(buffer, format='pdf', bbox_inches='tight')
        plt.close(fig)
        with pymupdf.open(stream=buffer.getvalue(), filetype='pdf') as document:
            original=document[0].read_contents().count(b'minus Do')
            publication_umap_page(document)
            spans=[s for b in document[0].get_text('dict')['blocks']
                   for line in b.get('lines', []) for s in line['spans']]
            self.assertTrue(any(abs(s['size']-23.1)<.1 for s in spans))
            self.assertNotIn('Embeddings UMAP', document[0].get_text())
            if original:
                self.assertGreater(document[0].read_contents().count(b'minus Do'), 0)
                self.assertLessEqual(document[0].read_contents().count(b'minus Do'), original)
            else:
                self.assertIn('−', document[0].get_text())

    def test_vector_pdf_has_all_five_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            # PDF content is the subject of this check; avoid large raster exports.
            with patch.object(pymupdf.Page, 'get_pixmap') as render:
                paths=plot_umap_run_composites(self.sources(base), base/'accumulated_results')
                self.assertEqual(render.call_count, 2)
                self.assertEqual(render.call_args.kwargs['dpi'], 300)
            self.assertEqual(len(paths), 4)
            for stage in ('before','after'):
                with pymupdf.open(base/'accumulated_results'/'plots'/f'UMAP_{stage}_PNA_5_runs.pdf') as doc:
                    self.assertEqual(len(doc), 1)
                    text=doc[0].get_text()
                    if stage == 'after':
                        self.assertIn('Breast Cancer: Trained last PNAConv node embeddings', text)
                    for index in range(1, 6):
                        self.assertIn(f'Run {index}', text)
                        self.assertIn(f'Source run {index}', text)
                    self.assertEqual(len(doc[0].get_images()), 0)


if __name__ == '__main__':
    unittest.main()
