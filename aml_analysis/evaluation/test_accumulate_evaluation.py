"""Five-run regression coverage for evaluation accumulation."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from accumulate_evaluation import average_tables, common_effects, accumulate_evaluation, export_top_likely_positive_cpgs, top20_correlation_plot, threshold_delta_beta_composite, export_within_criterion_cpgs
from replication_metrics import INTERNAL, EXTERNAL, annotate_candidates, agreement_metrics, sensitivity_table, validation_plots


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class AccumulationTests(unittest.TestCase):
    def frame(self, index):
        return annotate_candidates(pd.DataFrame({
            'CpG_ID': ['cg1', 'cg2', f'cg_run{index}'],
            INTERNAL: [.1 + index*.01, -.2, .3],
            EXTERNAL: [.2, -.3, .4],
            'External_p_value': [.0001]*3,
            'External_FDR': [.001, .02, .03], 'External_testable': [True]*3}))

    def test_publication_four_panel_style(self):
        import matplotlib.pyplot as plt
        from accumulate_evaluation import GROUPS
        data = {group: self.frame(0) for group, _, _, _ in GROUPS}
        rows = [dict(plotted_unique_cpgs=3, pearson_r=.9, spearman_r=.835,
                     same_direction_fraction=1., **agreement_metrics(data[group]))
                for group, _, _, _ in GROUPS]
        fig = validation_plots(data, rows, GROUPS, 'Mean original cohort', publication=True)
        try:
            self.assertEqual(len(fig.axes), 4)
            for ax in fig.axes:
                for axis in (ax.xaxis, ax.yaxis):
                    self.assertEqual(axis.label.get_fontsize(), 24)
                    self.assertTrue(all(t.get_fontsize() == 24 for t in axis.get_ticklabels()))
                for annotation in ax.texts:
                    self.assertFalse(any(word in annotation.get_text()
                                         for word in ('Spearman', 'Direction=', 'MAE=')))
            for ax in fig.axes[:2]:
                self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                                 ['Outside criterion', 'Within criterion'])
                self.assertTrue(any('Pearson' in t.get_text() for t in ax.texts))
                self.assertTrue(all(t.get_fontsize() == 22 for t in ax.get_legend().get_texts()))
            fig.savefig('/tmp/nanodiag_publication_preview.png', dpi=65, bbox_inches='tight')
        finally:
            plt.close(fig)

    def test_common_cpgs_and_external_inference(self):
        tables = [self.frame(i) for i in range(5)]
        result = common_effects(tables)
        self.assertEqual(result.CpG_ID.tolist(), ['cg1', 'cg2'])
        self.assertAlmostEqual(result[INTERNAL].iloc[0], .12)
        self.assertTrue(result.n_runs.eq(5).all())
        tables[-1].loc[0, 'External_FDR'] = .9
        with self.assertRaisesRegex(ValueError, 'External cohort statistics differ'):
            common_effects(tables)

    def test_top_twenty_absolute_rankings_and_five_run_requirement(self):
        frame = pd.DataFrame({'CpG_ID': [f'cg{i:03}' for i in range(25)],
                              INTERNAL: np.arange(25)/100,
                              EXTERNAL: -np.arange(25)/100,
                              'n_runs': 5, 'External_FDR': .001})
        frame.loc[24, EXTERNAL] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            paths = export_top_likely_positive_cpgs(frame, temp, 'PNA_test')
            external, internal = [pd.read_csv(path, sep='\t') for path in paths]
            self.assertEqual(len(external), 20)
            self.assertEqual(len(internal), 20)
            self.assertEqual(external.CpG_ID.iloc[0], 'cg023')
            self.assertEqual(internal.CpG_ID.iloc[0], 'cg023')
            self.assertTrue(external.External_delta_beta.abs().is_monotonic_decreasing)
            self.assertTrue(internal.Internal_delta_beta.abs().is_monotonic_decreasing)
            self.assertEqual(external.Rank.tolist(), list(range(1, 21)))
            self.assertNotIn('cg024', internal.CpG_ID.tolist())
            frame['n_runs'] = 4
            with self.assertRaisesRegex(ValueError, 'exactly five'):
                export_top_likely_positive_cpgs(frame, temp, 'PNA_test')

    def test_export_criterion_boundaries_and_gene_annotations(self):
        frame = pd.DataFrame({'CpG_ID': ['cg1', 'cg2', 'cg3', 'cg4', 'cg5'],
                              INTERNAL: [.1, .2, .3, -.2, 0],
                              EXTERNAL: [.1, .099, .2, .2, .2],
                              'External_FDR': [.049, .001, .05, .01, .01],
                              'Predicted_genes': ['GENE1;GENE2']*5,
                              'External_genes': ['GENE2;GENE3']*5,
                              'n_runs': 5})
        with tempfile.TemporaryDirectory() as temp:
            paths = export_top_likely_positive_cpgs(frame, temp, 'PNA_test')
            result = pd.read_csv(paths[0], sep='\t').set_index('CpG_ID')
            self.assertEqual(result.index[result.Meets_exploratory_criterion].tolist(), ['cg1'])
            self.assertTrue(result.Associated_genes.eq('GENE1;GENE2;GENE3').all())
            self.assertFalse(result.loc['cg2', 'External_abs_delta_beta_ge_0_10'])
            self.assertFalse(result.loc['cg3', 'External_significant'])
            self.assertFalse(result.loc['cg4', 'Same_nonzero_direction'])
            self.assertFalse(result.loc['cg5', 'Same_nonzero_direction'])

    def test_top_twenty_correlation_uses_signed_table_values(self):
        import matplotlib.pyplot as plt
        table = pd.DataFrame({'Internal_delta_beta': [-.3, .2, .4],
                              'External_delta_beta': [-.2, .3, .5],
                              'Meets_exploratory_criterion': [True, False, True]})
        fig = top20_correlation_plot(table, 'PNA')
        try:
            ax = fig.axes[0]
            np.testing.assert_allclose(ax.collections[0].get_offsets(), [[.2, .3]])
            np.testing.assert_allclose(ax.collections[1].get_offsets(), [[-.3, -.2], [.4, .5]])
            self.assertEqual(ax.get_xlabel(), 'Mean original cohort Δβ')
            self.assertEqual(ax.get_ylabel(), 'External cohort Δβ')
            self.assertTrue(any('Pearson r=' in text.get_text() for text in ax.texts))
        finally:
            plt.close(fig)

    def test_internal_ranked_correlation_header(self):
        import matplotlib.pyplot as plt
        table = pd.DataFrame({'Internal_delta_beta': [-.3, .4],
                              'External_delta_beta': [-.2, .5],
                              'Meets_exploratory_criterion': [True, True]})
        fig = top20_correlation_plot(table, 'PNA', ranked_by='internal')
        try:
            self.assertIn('|internal Δβ|', fig.axes[0].get_title())
            self.assertEqual(fig.axes[0].get_xlabel(), 'Mean original cohort Δβ')
            self.assertEqual(fig.axes[0].get_ylabel(), 'External cohort Δβ')
        finally:
            plt.close(fig)

    def test_strict_p_value_and_fdr_boundaries(self):
        frame = pd.DataFrame({INTERNAL: [.4]*7 + [-.4],
                              EXTERNAL: [.4]*6 + [.29, .4],
                              'External_p_value': [.009, .01, .001, np.nan, .001, .001, .001, .001],
                              'External_FDR': [.009, .001, .01, .001, np.nan, .011, .001, .001]})
        result = annotate_candidates(frame, effect_threshold=.30,
                                     fdr_threshold=.01, p_value_threshold=.01)
        self.assertEqual(result.Meets_exploratory_criterion.tolist(),
                         [True, False, False, False, False, False, False, False])

    def test_threshold_composites_recompute_markers_and_counts(self):
        import matplotlib.pyplot as plt
        from accumulate_evaluation import GROUPS
        frame = annotate_candidates(pd.DataFrame({
            'CpG_ID': ['cg1', 'cg2', 'cg3', 'cg4', 'cg5'],
            INTERNAL: [.2, -.4, .4, .4, -.4],
            EXTERNAL: [.10, -.30, .299, .4, .4],
            'External_p_value': [.001]*5,
            'External_FDR': [.001, .001, .001, .01, .001],
            'External_testable': True}))
        for threshold, expected in ((.10, 3), (.30, 1)):
            fig, summary = threshold_delta_beta_composite(
                {group: frame for group, _, _, _ in GROUPS}, 'PNA', 5, threshold)
            try:
                self.assertTrue(summary.exploratory_criterion_count.eq(expected).all())
                for ax in fig.axes[:2]:
                    self.assertEqual(len(ax.collections[1].get_offsets()), expected)
                self.assertIn(f'≥ {threshold:.2f}', fig._suptitle.get_text())
            finally:
                plt.close(fig)
        self.assertEqual(int(frame.Meets_exploratory_criterion.sum()), 4)

    def test_within_criterion_export_has_no_top_twenty_limit(self):
        frame = pd.DataFrame({'CpG_ID': [f'cg{i:03}' for i in range(30)],
                              INTERNAL: -.4, EXTERNAL: -.3, 'n_runs': 5,
                              'External_p_value': .001, 'External_FDR': .001, 'Predicted_genes': 'GENE1'})
        frame.loc[0, INTERNAL] = -.299
        frame.loc[4, EXTERNAL] = -.299
        frame.loc[1, 'External_FDR'] = .01
        frame.loc[2, INTERNAL] = .4
        frame.loc[3, INTERNAL] = 0
        with tempfile.TemporaryDirectory() as temp:
            path = export_within_criterion_cpgs(frame, temp, 'PNA_test', .30)
            result = pd.read_csv(path, sep='\t')
            self.assertEqual(len(result), 26)
            self.assertTrue(result.Meets_exploratory_criterion.all())
            self.assertIn('cg000', result.CpG_ID.tolist())
            self.assertTrue(result.External_absolute_delta_beta.ge(.3).all())
            self.assertTrue(result.External_FDR.lt(.01).all())
            self.assertTrue(result.External_p_value.lt(.01).all())
            self.assertTrue(result.Associated_genes.eq('GENE1').all())
            self.assertFalse(set(result.CpG_ID) & {'cg001', 'cg002', 'cg003', 'cg004'})

    def test_equal_run_weighting(self):
        result = average_tables([pd.DataFrame({'prediction_group': ['positive'],
                      'fraction': [v], 'denominator': [n]})
                      for v, n in ((.2, 10), (.8, 100))], ['prediction_group'])
        self.assertAlmostEqual(result.fraction_mean.iloc[0], .5)
        self.assertEqual(result.fraction_count.iloc[0], 2)

    def test_five_run_outputs_for_each_model(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/'data_test'
            for model in ('PNA', 'GCN'):
                for index in range(5):
                    run = base/'evaluation'/f'run{index+1}'
                    run.mkdir(parents=True, exist_ok=True)
                    suffix = f'{model}_test'
                    data, rows = {}, []
                    for group in ('likely_positive', 'negative'):
                        frame = self.frame(index)
                        frame.to_csv(run/f'{group}_common_unique_cpgs_{suffix}.tsv', sep='\t', index=False)
                        data[group] = frame
                        rows.append(dict(prediction_group=group, **agreement_metrics(frame)))
                    pd.DataFrame(rows).to_csv(run/f'summary_{suffix}.tsv', sep='\t', index=False)
                    sensitivity_table(data).to_csv(run/f'criterion_sensitivity_{suffix}.tsv', sep='\t', index=False)
            accumulate_evaluation(Config(p_base=str(base), model_type=['PNA', 'GCN'], dpi=25))
            output = base/'accumulated_results'/'evaluation'
            for model in ('PNA', 'GCN'):
                suffix = f'{model}_test'
                provenance = json.loads((output/f'included_runs_{suffix}.json').read_text())
                self.assertEqual(provenance['runs'], [f'run{i}' for i in range(1, 6)])
                for name in ('criterion_sensitivity', 'delta_beta_criterion',
                             'delta_beta_original_vs_external', 'delta_beta_shift_by_prediction_group',
                             'delta_beta_original_vs_external_top20_external',
                             'delta_beta_original_vs_external_top20_internal',
                             'delta_beta_original_vs_external_abs_external_ge_0.10',
                             'delta_beta_original_vs_external_abs_external_ge_0.30'):
                    for ext in ('png', 'pdf'):
                        self.assertGreater((output/f'{name}_{suffix}.{ext}').stat().st_size, 0)
                frame = pd.read_csv(output/f'likely_positive_common_unique_cpgs_{suffix}.tsv', sep='\t')
                self.assertEqual(frame.CpG_ID.tolist(), ['cg1', 'cg2'])
                for cohort in ('external', 'internal'):
                    top = pd.read_csv(output/f'top20_likely_positive_common_5_runs_by_{cohort}_delta_beta_{suffix}.tsv', sep='\t')
                    self.assertEqual(set(top.CpG_ID), {'cg1', 'cg2'})
                    self.assertTrue(top.n_runs.eq(5).all())
                summary = pd.read_csv(output/f'summary_{suffix}.tsv', sep='\t')
                self.assertTrue(summary.exploratory_criterion_fraction_count.eq(5).all())


if __name__ == '__main__':
    unittest.main()
