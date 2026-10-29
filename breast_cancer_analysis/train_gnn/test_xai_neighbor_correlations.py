"""Verify neighbor selection, seed colors, and methylation-only correlations."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from plot_xai_neighbor_correlations import correlate_profiles, plotted_nodes, read_profiles, delta_beta_statistics, delta_beta_scatter, cohort_comparison_table, combined_xai_figure


class NeighborCorrelationTests(unittest.TestCase):
    def test_signed_correlations_and_seed_colors(self):
        names = ['cg1_A', 'cg2_B', 'cg3_C', 'cg4_D']
        profiles = pd.DataFrame([[1, 2, 3, np.nan], [2, 4, 6, 8],
                                 [3, 2, 1, 0], [1, 1, 1, 1]], index=names)
        nodes = dict(enumerate(names))
        table = pd.DataFrame({'name': names, 'seed_score': [0., .5, 0., 0.]})
        result = correlate_profiles(profiles, nodes, table, 0).set_index('neighbor_node_id')
        self.assertAlmostEqual(result.loc[1, 'correlation'], 1.)
        self.assertAlmostEqual(result.loc[2, 'correlation'], -1.)
        self.assertEqual(result.loc[1, 'node_group'], 'green_seed')
        self.assertEqual(result.loc[2, 'node_group'], 'blue_other')
        self.assertEqual(result.loc[1, 'n_paired_samples'], 3)
        self.assertEqual(result.loc[3, 'status'], 'constant_profile')
        self.assertTrue(np.isnan(result.loc[3, 'correlation']))
        table.loc[1, 'name'] = 'wrong_name'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            correlate_profiles(profiles, nodes, table, 0)

    def test_delta_beta_sign_and_sample_order(self):
        with tempfile.TemporaryDirectory() as temp:
            matrix = Path(temp)/'matrix.tsv'
            original = pd.DataFrame({'CpG_ID': ['cg1','cg2'], 'NC_h1': [.1,.8],
                                     'NC_h2': [.3,.6], 'BC_a1': [.6,.3], 'BC_a2': [.8,.1]})
            original.to_csv(matrix, sep='\t', index=False)
            profiles=pd.DataFrame([[.1,.3,.6,.8],[.8,.6,.3,.1]],
                                  index=['cg1_A','cg2_B'], columns=['0','1','2','3'])
            stats=delta_beta_statistics(profiles, None, matrix)
            self.assertAlmostEqual(stats.loc['cg1_A','delta_beta_BC_minus_healthy'], .5)
            self.assertAlmostEqual(stats.loc['cg2_B','delta_beta_BC_minus_healthy'], -.5)
            self.assertEqual(stats.loc['cg1_A','healthy_n'],2)
            profiles.iloc[0] = profiles.iloc[0].to_numpy()[::-1]
            with self.assertRaisesRegex(ValueError,'sample order/beta values'):
                delta_beta_statistics(profiles,None,matrix)

    def test_scatter_uses_current_and_external_delta_beta(self):
        import matplotlib.pyplot as plt
        result=pd.DataFrame({'node_id': [1,2,0], 'node_name': ['cg1_A','cg2_B','cg0_X'],
                             'node_group': ['green_seed','blue_other','xai_node'],
                             'delta_beta_BC_minus_healthy': [.3,-.2,.5],
                             'external_delta_beta_BC_minus_healthy': [.25,-.1,.45]})
        fig=delta_beta_scatter(result, 0, 'cg0_X')
        try:
            np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(), [[.3,.25]])
            np.testing.assert_allclose(fig.axes[0].collections[1].get_offsets(), [[-.2,-.1]])
            np.testing.assert_allclose(fig.axes[0].collections[2].get_offsets(), [[.5,.45]])
            np.testing.assert_allclose(fig.axes[0].get_xlim(), fig.axes[0].get_ylim())
            np.testing.assert_allclose(fig.axes[0].get_xticks(), fig.axes[0].get_yticks())
            self.assertEqual({label.get_text() for label in fig.axes[0].texts},
                             {'cg1_A','cg2_B','cg0_X'})
            self.assertTrue(all(label.arrow_patch is None for label in fig.axes[0].texts))
            self.assertIn('Current cohort', fig.axes[0].get_xlabel())
            self.assertIn('External cohort', fig.axes[0].get_ylabel())
            self.assertTrue(any(np.array_equal(line.get_xdata(), line.get_ydata())
                                for line in fig.axes[0].lines))
        finally:
            plt.close(fig)

    def test_external_matching_is_by_cpg_and_checks_duplicate_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'external.tsv'
            external=pd.DataFrame({'CpG_ID': ['cg1','cg1','cg2'],
                'Healthy_GSE101961_mean_beta': [.1,.1,.4],
                'BC_GSE226569_mean_beta': [.4,.4,.2],
                'Delta_beta_BC_minus_healthy': [.3,.3,-.2]})
            external.to_csv(path, sep='\t', index=False)
            nodes={0:'cg1_A',1:'cg2_B',2:'cg3_C'}
            table=pd.DataFrame({'seed_score':[0.,.5,0.]})
            stats=pd.DataFrame({'delta_beta_BC_minus_healthy':[.2,-.1,.4]}, index=list(nodes.values()))
            result=cohort_comparison_table(stats,nodes,table,0,path)
            self.assertEqual(len(result),3)
            self.assertAlmostEqual(result.loc[0,'external_delta_beta_BC_minus_healthy'],.3)
            self.assertEqual(result.node_group.tolist(),['xai_node','green_seed','blue_other'])
            self.assertFalse(result.loc[2,'matched_external'])
            external.loc[1,'Delta_beta_BC_minus_healthy']=.6
            external.to_csv(path, sep='\t', index=False)
            with self.assertRaisesRegex(ValueError,'inconsistent'):
                cohort_comparison_table(stats,nodes,table,0,path)

    def test_composite_contains_all_three_vector_panels(self):
        import pymupdf
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            sources=[]
            for name in ('graph','bars','delta'):
                path=base/f'{name}.pdf'
                with pymupdf.open() as document:
                    page=document.new_page()
                    page.insert_text((30,40), f'{name} content')
                    document.save(path)
                sources.append(path)
            with patch.object(pymupdf.Page, 'get_pixmap'):
                combined_xai_figure(*sources,base/'combined',7980,'cg16909733_ANKMY1')
            with pymupdf.open(base/'combined.pdf') as document:
                self.assertEqual(len(document),1)
                text=document[0].get_text()
                for name in ('graph','bars','delta'):
                    self.assertIn(f'{name} content',text)
                self.assertIn('A. Explanation subgraph',text)
                self.assertIn('B. DNA methylation correlation',text)
                self.assertIn('C. Δβ : Original vs external cohort',text)
                self.assertEqual(len(document[0].get_images()),0)

    def test_exact_pdf_legend_and_named_profile_rows(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as temp:
            pdf = Path(temp)/'xai.pdf'
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((30, 40), '7980:cg16909733_ANKMY1\n7978:cg20474370_PIK3R1')
                document.save(pdf)
            nodes = plotted_nodes(pdf)
            self.assertEqual(nodes, {7980: 'cg16909733_ANKMY1', 7978: 'cg20474370_PIK3R1'})
            path = Path(temp)/'features.csv'
            pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
                         index=['unused', nodes[7978], nodes[7980]]).to_csv(path)
            profiles = read_profiles(path, nodes.values())
            self.assertEqual(profiles.index.tolist(), list(nodes.values()))
            self.assertEqual(profiles.loc[nodes[7980]].tolist(), [7,8,9])


if __name__ == '__main__':
    unittest.main()
