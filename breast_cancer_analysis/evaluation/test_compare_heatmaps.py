"""Run1-only preprocessing comparisons, provenance, and missing-score handling."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import pandas as pd
from matplotlib.figure import Figure
from compare_heatmaps import read_folder, compare_heatmaps, order_configurations


class HeatmapComparisonTests(unittest.TestCase):
    def write(self, folder, relative, values):
        path=folder/relative/'Heatmap_avg_edges_100_epochs_20.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values).to_csv(path, index=False)
        return path

    def test_configurations_are_grouped_by_parameter_sweep(self):
        rows = [dict(folder=str(i), corr=corr, num_neighbours=neighbours, num_unknown=unknown)
                for i, (corr, neighbours, unknown) in enumerate(
                    ((.33,500,15000), (.33,500,10000), (.7,3000,10000),
                     (.2,3000,10000), (.33,3000,5000), (.33,3000,10000),
                     (.33,1500,10000), (.5,3000,10000)))]
        ordered=order_configurations(rows)
        self.assertEqual(ordered.folder.tolist(), ['3','5','7','2','1','6','4','0'])
        self.assertEqual(ordered.configuration_group.tolist(), [0,0,0,0,1,1,2,2])

    def test_run1_only_never_uses_current_or_accumulated_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'data_corr_0.33_knn_n_3000_n_neg_10000'
            values={'model':['PNA'], 'best_val_acc':[.7], 'te_acc':[.6]}
            source=self.write(folder,'runs/run1/generated/data',values)
            self.write(folder,'generated/data',dict(values,best_val_acc=[.99]))
            self.write(folder,'accumulated_results/data',dict(values,best_val_acc=[.98]))
            self.write(folder,'runs/run2/generated/data',dict(values,best_val_acc=[.97]))
            result=read_folder(folder)
            self.assertEqual(result.validation_accuracy.iloc[0],.7)
            self.assertEqual(result.source_file.iloc[0],str(source))
            self.assertEqual(result.source_scope.iloc[0],'run1')
            source.unlink()
            with self.assertRaisesRegex(FileNotFoundError,'no run1'):
                read_folder(folder)

    def test_missing_models_and_cells_are_filled_from_run1_json(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'data_corr_0.33_knn_n_500_n_neg_10000'
            self.write(folder,'runs/run1/generated/data',{'model':['PNA'], 'best_val_acc':[.7], 'te_acc':[None]})
            data=folder/'runs/run1/generated/data'
            for model, values in {'PNA': {'best_val_acc': .99, 'te_acc': [.6]},
                                  'TabICL': {'best_val_acc': .8, 'te_acc': [.75]},
                                  'XGBoost': {'val_acc': [.7,.85], 'te_acc': [.78]}}.items():
                (data/f'all_metrics_{model}.json').write_text(json.dumps(values))
            result=read_folder(folder).set_index('model')
            self.assertEqual(result.loc['PNA','validation_accuracy'],.7)
            self.assertEqual(result.loc['PNA','test_accuracy'],.6)
            self.assertEqual(result.loc['TabICL','test_accuracy'],.75)
            self.assertEqual(result.loc['XGBoost','validation_accuracy'],.85)
            self.assertTrue(result.loc['TabICL','source_file'].endswith('all_metrics_TabICL.json'))

    def test_selected_missing_folder_and_model_stay_na(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            first=base/'data_corr_0.2_knn_n_3000_n_neg_10000'
            second=base/'data_corr_0.5_knn_n_3000_n_neg_10000'
            self.write(first,'runs/run1/generated/data',{'model':['PNA','GCN'], 'best_val_acc':[.7,.8], 'te_acc':[.6,None]})
            with patch.object(Figure,'savefig'):
                result=compare_heatmaps([first,second],base/'out')
            self.assertEqual(len(result),4)
            missing=result.loc[result.folder.eq(second.name)]
            self.assertTrue(missing.validation_accuracy.isna().all())
            self.assertTrue(missing.test_accuracy.isna().all())
            self.assertTrue(missing.source_scope.eq('missing_run1').all())
            matrix=pd.read_csv(base/'out/validation_accuracy_heatmap.tsv',sep='\t')
            self.assertEqual(len(matrix),2)


if __name__=='__main__':unittest.main()
