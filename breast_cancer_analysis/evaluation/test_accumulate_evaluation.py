"""Five-run regression coverage for evaluation accumulation."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from accumulate_evaluation import average_tables, common_effects, accumulate_evaluation
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
                             'delta_beta_original_vs_external', 'delta_beta_shift_by_prediction_group'):
                    for ext in ('png', 'pdf'):
                        self.assertGreater((output/f'{name}_{suffix}.{ext}').stat().st_size, 0)
                frame = pd.read_csv(output/f'likely_positive_common_unique_cpgs_{suffix}.tsv', sep='\t')
                self.assertEqual(frame.CpG_ID.tolist(), ['cg1', 'cg2'])
                summary = pd.read_csv(output/f'summary_{suffix}.tsv', sep='\t')
                self.assertTrue(summary.exploratory_criterion_fraction_count.eq(5).all())


if __name__ == '__main__':
    unittest.main()
