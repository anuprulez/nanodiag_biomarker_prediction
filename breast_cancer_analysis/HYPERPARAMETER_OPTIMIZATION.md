# Optuna hyperparameter optimization

Run these commands from `breast_cancer_analysis` in the existing project environment:

```bash
python -m pip install -r train_gnn/requirements-optuna.txt
python run_pipeline.py --stage tune --trials 50
```

The default models are PNA, GCN, GraphSAGE, GATv2, GraphTransformer, TabPFN,
TabICL and XGBoost; CatBoost is also supported. Each selected preprocessing combination in `config/config.yaml`
gets a separate study for each model. Preprocessing must already have completed.
Tuning does not change the shared config or recompute negative selection/splits.
`--stage all` performs preprocessing, validation, tuning, training, XAI and prediction comparison in order.
Model/XAI defaults and output templates are under `runtime_defaults` in
`config/optuna.yaml`; `config/config.yaml` is preprocessing-only.

Select a smaller set of models or apply a time limit:

```bash
python run_pipeline.py --stage tune --models PNA XGBoost CatBoost --trials 30 --timeout 7200
```

`--trials` is the number of **additional** trials per model per dataset on each
invocation. `--timeout` is seconds per model per dataset, checked between trials.
The actual total budget multiplies by the number of dataset configurations and
models. Trials run sequentially. Repeating a command resumes its SQLite studies
and seeded sampler state. CPU and GPU numerical results can differ.

Use the tuner directly to select a single dataset or a custom search space:

```bash
python train_gnn/tune_models.py \
  --corr-threshold 0.33 --knn-neighbours 3000 --size-negative 10000 \
  --models PNA XGBoost CatBoost --trials 30 \
  --search-space config/optuna.yaml
```

## Search spaces

Edit `config/optuna.yaml` to adjust these initial ranges. They are starting
search ranges, not claims that the optimal values lie inside them.

| GNN parameter | Search range |
| --- | --- |
| `batch_size` | 64, 128, 256, 512 |
| `grad_clip_norm` | 0.5–5, logarithmic |
| `hidden_dim` | 32, 64, 128, 256 |
| `learning_rate` | 0.00001–0.01, logarithmic |
| `weight_decay` | 0.000001–0.1, logarithmic |
| `dropout` | 0–0.9 |
| `gnn_layers` | 1–3 |
| `heads` (GATv2 / GraphTransformer) | 1, 2, 4, 8 |
| `neighbors_spread` | [5,5,5], [10,5,5], [20,10,5], [40,20,10]; truncated to layer count |
| `embedding_dropout` | 0–0.3 |
| `feature_dropout` | 0–0.2 |

The architecture currently supports at most three layers. Attention heads must
divide the hidden dimension. Neighborhood sampling uses one configured fanout
per layer: `neighbors_spread` is truncated or extended with its last value.
Other GNN settings, including class weighting,
epoch budget (`n_epo`), and early stopping patience, remain fixed from the shared
config. Device can be selected with `hpo_device: cpu` or `hpo_device: cuda`.
The default uses CUDA when available.

| XGBoost parameter | Search range |
| --- | --- |
| `n_estimators` | 200–2000, step 100 |
| `max_depth` | 3–10 |
| `learning_rate` | 0.005–0.3, logarithmic |
| `min_child_weight` | 0.1–30, logarithmic |
| `subsample` | 0.5–1 |
| `colsample_bytree` | 0.5–1 |
| `gamma` | 0–5 |
| `reg_alpha` | 0.00000001–10, logarithmic |
| `reg_lambda` | 0.000001–100, logarithmic |
| `max_bin` | 128, 256, 512 |

| CatBoost parameter | Search range |
| --- | --- |
| `iterations` | 200–2000, step 100 |
| `depth` | 3–10 |
| `learning_rate` | 0.005–0.3, logarithmic |
| `l2_leaf_reg` | 0.001–100, logarithmic |
| `random_strength` | 0.00000001–10, logarithmic |
| `bagging_temperature` | 0–5, Bayesian bootstrap |
| `border_count` | 32, 64, 128, 254 |
| `auto_class_weights` | None, Balanced, SqrtBalanced |

The GNN settings `batch_size`, `grad_clip_norm`, `hidden_dim`, `weight_decay`,
`dropout`, and `gnn_layers` do not apply to these tree estimators. Their own
regularization and sampling parameters above serve different purposes.
Tree settings are also available in ordinary training via
`model_overrides.XGBoost.estimator_params` and
`model_overrides.CatBoost.estimator_params` in each dataset folder's `config.yaml`.
`tabular_patience` controls tree early stopping (default 30). Trees default to CPU
in tuning; set `tabular_device: cuda` for GPU, and `tabular_threads` for CPU
parallelism. XGBoost uses histogram trees and CatBoost uses Bayesian bootstrap
when tuning bagging temperature.

## TabPFN and TabICL

These pretrained models have optional **inference** searches, enabled explicitly:

```bash
python run_pipeline.py --stage tune --models TabPFN TabICL --trials 20
```

| Model | Tunable settings in this implementation |
| --- | --- |
| TabPFN | `n_estimators`: 4/8/16/32; `softmax_temperature`: 0.5–1.5; `balance_probabilities`: false/true; `average_before_softmax`: false/true |
| TabICL | `n_estimators`: 4/8/16/32; `softmax_temperature`: 0.5–1.5; `outlier_threshold`: 2–8; `feat_shuffle_method`: latin/random; `class_shuffle_method`: shift/none; `average_logits`: false/true |

The checkpoint weights are fixed; no neural weight fine-tuning is performed.
Checkpoint paths remain the explicit verified versions from the project config.
Temperature changes confidence; its effect on argmax classification can be
limited, so macro F1 may be insensitive to it. `tabular_preprocessing` is fixed
per study (`original` by default); it can be compared in separate studies.
Additional possible experiments include TabICL `norm_methods` and TabPFN
`inference_config` preprocessing choices, but these structured parameters are
not included in the supplied scalar search spaces. Device, caching, precision,
and inference batch sizes primarily control speed and memory, and are fixed.

## Validation and saved results

The objective maximizes **validation macro F1**, configurable to `accuracy` in
`config/optuna.yaml`. Every trial uses the existing CpG-level partitions.
Scalers are fitted on training nodes. Models fit training labels only; validation
controls selection and early stopping. Test predictions and metrics are never
computed during tuning. GNNs use Optuna median pruning and early stopping;
trees use their native validation-loss early stopping. Tree iteration counts
are upper budgets, and actual selected iterations are recorded in the study.

Results are stored under:

```text
data_corr_<corr>_knn_n_<knn>_n_neg_<negative>/generated/optuna/<model>/
  study.sqlite3
  sampler.joblib
  trials.csv
  best_params.json
  best_config.yaml
  best_model.pt              # GNN weights
  best_model.joblib          # tabular estimator plus scaler, instead of .pt
  prepared/                 # isolated feature/graph exports
  trial_00000/              # checkpoint, and GNN validation history
```

After each model finishes successfully, tuning writes its validation winner to
`<dataset-folder>/config.yaml`. The shared preprocessing config is not changed.
The dataset config contains the preprocessing snapshot with absolute root/input
paths, model-specific overrides, plotting/XAI settings and tuning provenance.
Only successfully tuned model names enter its `model_type` list. Models within
one dataset share `generated/data`, `generated/plots` and `generated/models`
with model-specific filenames;
Optuna `best_config.yaml` files remain standalone per-model trial exports.

Training, XAI and comparisons use the dataset config directly:

```bash
python train_gnn/main.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml
python train_gnn/run_xai_explainer.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml --node-name cg00000001_GENE
python evaluation/run_comparisons.py --config data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml --models PNA
```

Or use `python run_pipeline.py --stage train` for all existing dataset folders,
or add `--folder <dataset-folder>` to select one. The same selection works with
`--stage preprocess`, `tune`, `xai` and `compare-predictions`. Missing dataset configs require tuning first.
To create configs from already completed matching Optuna studies without training:

```bash
python train_gnn/tune_models.py --export-existing
```

Per-model `generated/optuna/<model>/best_config.yaml` remains an optional export
for a separate fresh training run into that study's `final/` folder. The main
pipeline uses the dataset-root config and the split generated subfolders.
Each dataset also has `evaluation/` for comparison outputs and `logs/` for stage logs.

Studies fingerprint preprocessing files, training settings, search settings, and
library versions. A changed objective cannot silently resume the same study.
To start again after changing inputs or search settings, move the existing model
study directory out of the way. Completed preprocessing manifests are skipped; incomplete existing folders can be rebuilt.

## Verification

The integration tests run two-epoch synthetic GNN trials for all five families,
short real XGBoost and CatBoost trials, resume studies without restarting the
sampler, check the exported estimator parameters, and reject changed settings.
They also compare optional foundation search names against installed classifier
constructors; they do not download or run foundation checkpoints.

```bash
python -m unittest discover -s train_gnn -p 'test_optuna_tuning.py'
```

Parameter/API references:
[Optuna studies](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html),
[Optuna trials and pruning](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html),
[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/),
[TabPFN classifier](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py),
[TabICL classifier](https://tabicl.readthedocs.io/en/latest/api.html).

## Focused PNA refinement

`config/optuna_pna_refined.yaml` adds PNA-specific distributions for class-weight
power, neighborhood fanout, feature dropout and embedding dropout. Per-model
GNN spaces override entries in `spaces.GNN`. `neighbor_fanout` expands into
`[fanout, fanout // 2, fanout // 4]` (minimum one) and is truncated to the number
of GNN layers; the exported config stores the actual `neighbors_spread`.
`initial_trials` is empty: populate it with candidates selected on breast cancer validation data.

The companion `config/config_pna_refined.yaml` uses the configured breast cancer dataset
and partitions, with a 150-epoch upper budget, early stopping patience 20, and
scheduler patience 5. Pruning waits until epoch 20. This is a larger search
budget than the original 25-trial comparison.

```bash
python train_gnn/tune_models.py \
  --config config/config_pna_refined.yaml \
  --search-space config/optuna_pna_refined.yaml \
  --models PNA --study-tag pna_refined --trials 50
```

`--study-tag` isolates this experiment under
`generated/optuna/pna_refined/PNA/`, preserving the original model studies.
Repeating the command adds trials to the refinement study. Compare validation
macro F1 only when choosing settings; a validation improvement does not establish
superiority on held-out test data or across different data splits.

TabPFN and TabICL default to `tabular_device: auto` in tuning and ordinary
training, allowing their libraries to select the available GPU. Tree models
retain CPU defaults. An explicit `tabular_device: cpu` or `cuda` is respected.
TabPFN still requires an explicit opt-in for large CPU datasets: set
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` before running if CPU execution is intended.
The GPU fix does not require editing existing configs or deleting studies.

The standard search now tunes neighborhood templates, embedding dropout and
feature dropout for every GNN. Attention heads are listed explicitly under
`spaces.GATv2` and `spaces.GraphTransformer`. Optuna stores neighborhood
choices as JSON strings internally; `best_config.yaml` and `best_params.json`
export the actual per-hop list applied to the selected layer count. Legacy
`spaces.Attention` and `neighbor_fanout` searches remain supported.
Because this changes the search objective/fingerprint, use a new study group
for the expanded search, preserving previous results:

```bash
python train_gnn/tune_models.py --models PNA GCN GraphSAGE GATv2 GraphTransformer   --study-tag expanded_gnn --trials 50
```

The `models` list in `config/optuna.yaml` is unchanged; `--models` overrides it
when running the GNN search above.
