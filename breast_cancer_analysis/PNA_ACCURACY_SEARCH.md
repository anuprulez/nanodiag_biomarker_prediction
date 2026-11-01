# Validation-selected PNA accuracy search

This experiment targets the accuracy comparison in
`data_corr_0.33_knn_n_3000_n_neg_10000/generated/Heatmap_avg_edges_9913451_epochs_100.pdf`.
The original PNA test accuracy is 0.836707; the highest original GNN test accuracy
is GraphTransformer's 0.840081. The previous Optuna search selected macro F1;
this search explicitly maximizes validation accuracy.

The original dataset, feature normalization, CpG partitions, propagated labels,
and GNN architectures are retained. The search changes PNA configuration only.
It does not examine the external breast cancer evaluation cohorts or use test
predictions to select a trial.

## Expanded space

`config/optuna_pna_accuracy.yaml` specifies 60 trials with a 150-epoch maximum,
20-epoch early stopping, scheduler patience 5, and pruning after 20 epochs.

| Setting | Search |
| --- | --- |
| Batch size | 64, 128, 256, 512, 1024 |
| Hidden width | 32, 64, 128, 256 |
| Layers | 1, 2 |
| Learning rate | 0.00005–0.006, logarithmic |
| Weight decay | 0.000001–0.05, logarithmic |
| Gradient clipping | 0.5–5, logarithmic |
| Dropout | 0–0.65 |
| Input feature dropout | 0–0.18 |
| Embedding dropout | 0–0.25 |
| Inverse-class-frequency exponent | 0, 0.25, 0.5, 0.75, 1 |
| Sampling templates | [5,5], [10,5], [20,10], [40,10], [80,20], truncated to layer count |

Twelve candidates are specified before adaptive exploration. Six keep the prior
PNA winner's parameters and vary weighting and neighborhood size. Six explore
lower regularization with one or two layers. Remaining trials use Optuna TPE.
Sampled neighborhoods are JSON strings in queued Optuna parameters, matching
the categorical storage representation used by the tuning code. Exported best
configs contain ordinary per-hop lists.

## Reproduce

From `breast_cancer_analysis` in `nanodiag_biomarker_gnn`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train_gnn/tune_models.py \
  --config config/config_pna_accuracy_search.yaml \
  --search-space config/optuna_pna_accuracy.yaml \
  --models PNA --study-tag pna_accuracy_v2 --trials 60
```

The study is stored under
`data_corr_0.33_knn_n_3000_n_neg_10000/generated/optuna/pna_accuracy_v2/PNA/`.
It contains the resumable SQLite study, sampler, each trial's validation history
and checkpoint, and exported `best_config.yaml`, `best_params.json`, and
`best_model.pt`. Repeating the command adds trials. For a fresh replication use
a new `--study-tag`; do not add trials after inspecting the winner's test result
and treat that as an untouched test evaluation.

## Freeze and evaluate once

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train_gnn/promote_pna_accuracy.py \
  --study data_corr_0.33_knn_n_3000_n_neg_10000/generated/optuna/pna_accuracy_v2/PNA \
  --install
```

The finalizer verifies exact equality with the original graph, inputs, labels
and masks, checks CpG/edge partition separation, and reproduces the selected
validation accuracy. It writes a frozen selection record before accessing test.
The selected checkpoint is evaluated once with repeatable sampling, without
retraining or selecting another model based on its test score.

Outputs are saved in `generated/pna_accuracy_improved/`, including the checkpoint,
prediction tensors, embeddings, split tables, graph snapshot, metrics, comparison
heatmap, a self-contained `selected_config.yaml`, and `REPORT.md`.
The previous PNA and other models remain in the original `generated/` directory.
A nonempty output directory is rejected to prevent accidental repeated test
assessment or overwriting results.

`--install` changes only `model_overrides.PNA` in the shared config and directs
PNA outputs to `${p_base}generated/pna_accuracy_improved/`. Other model families,
data paths and preprocessing settings remain unchanged. The prior shared config
is saved beside the improved checkpoint. The default explainer uses the same
per-model settings and therefore finds the improved PNA artifacts.

The external comparison script detects the completed improved PNA automatically:

```bash
python evaluation/compare_predictions.py --data data_corr_0.33_knn_n_3000_n_neg_10000 --model PNA
```

It uses the selected model's predictions and split tables from
`generated/pna_accuracy_improved/`. `--generated-dir` can select a different
artifact directory explicitly. The comparison report records that directory.

## Verification and interpretation

```bash
python -m unittest discover -s train_gnn -p 'test_optuna_tuning.py'
python -m unittest discover -s train_gnn -p 'test_pna_promotion.py'
```

The promotion fixture checks inference and metrics, artifact preservation,
PNA-only config installation and refusal to repeat evaluation. Results are for
one seed on propagated CpG-node targets. Search over one validation split can
overfit it; results do not establish general superiority across seeds, patients
or external biological cohorts.

## Completed result

All 60 validation trials completed or were pruned: {'COMPLETE': 42, 'PRUNED': 18}.
Trial 35 was selected. Validation accuracy rose from 78.96% to 83.06%.
The frozen checkpoint achieved 86.17% test accuracy (original PNA 83.67%;
best other GNN, GraphTransformer, 84.01%) and test macro F1 0.886893.
The result is recorded in `generated/pna_accuracy_improved/REPORT.md` inside
the dataset directory. Only PNA settings and its output directory were
installed in `config/config.yaml`.
