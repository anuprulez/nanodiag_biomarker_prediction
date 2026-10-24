# Breast cancer analysis

The active cohort is GSE66695: 80 breast tumors and 40 normal breast samples
on GPL13534, with 124 inputs (120 beta values plus four NeDBIT features).
The seed list reconstructs the MeinteR paper's 293 DMS+ and 62 DMS− sites.

Configuration follows the stages:

- `config/config.yaml` contains preprocessing inputs, filters, graph parameters,
  split settings and cohort dimensions only.
- `config/optuna.yaml` contains search spaces, trial settings and
  `runtime_defaults` for model training, output filenames, plotting and XAI.
- Tuning creates `<dataset-folder>/config.yaml`, with a fixed preprocessing
  snapshot, seed provenance and each completed model's winning settings.
  Training, XAI and prediction comparisons load this dataset config. Shared
  preprocessing settings are never overwritten by tuning.

## Changes identified and ported

| Area | Corresponding breast cancer update |
| --- | --- |
| Pipeline | `run_pipeline.py` runs preprocessing, output validation, training or tuning with per-dataset logs and bounded compute threads. |
| Parameter sweeps | `preprocessing_grid.py` accepts scalar/list correlation, kNN and negative-count parameters, expands their Cartesian product and supports CLI overrides. |
| Output layout | Each dataset has `preprocessed/`, `generated/{data,plots,models}`, `evaluation/` and `logs/`. |
| Reproducibility | Canonical feature ordering, seeded tie breaking, exact nested negative sets, and independent positive/negative CpG split generators. All mappings of a CpG stay together. |
| Input and seed checks | Shared raw input checks, finite beta/dimension checks, seed retention report, and SHA256 seed manifest written only after preprocessing completes. Training, tuning and explanations reject stale or missing provenance. |
| Validation | `validate_bc_outputs.py` checks graph IDs, labels, dimensions, finite values, class coverage, CpG isolation and absence of cross-partition edges. |
| GNNs | Residual architectures, original-feature paths, configurable one to three layers, feature/embedding dropout, class weighting and validation checkpoint selection. Defaults use two layers and two sampling hops. |
| Per-model settings | `model_settings.py` resolves `model_overrides` consistently for training and explanations; tabular estimators accept constructor overrides. |
| Optimization | Resumable validation-only Optuna studies for GNNs and tabular models, expanded sampling/dropout/attention spaces, pruning, fingerprints, saved best configs and a focused PNA template. |
| Tabular devices | Foundation estimators automatically select available hardware; tree models default to CPU, with explicit overrides supported. |
| Explanations | Explain each selected dataset, resolve a feature by name or node ID, validate bounds and load the selected model's settings. |
| Benchmarks/plots | Benchmark discovery supports both new and legacy folders. Seed plots use actual available seeds. Breast cancer sample labels and ordering are retained. |
| Regression checks | Deterministic preprocessing, real synthetic Optuna integrations, and explanation combination tests. |

AML-specific GEO acquisition/harmonisation, annotation and differential-CpG
selection are cohort-specific and have no direct replacement here: breast cancer
continues to use its processed matrix, mapper and published seeds. AML fitted
models, generated results and validation-selected hyperparameter winners are not
breast cancer results. `model_overrides` starts empty, and the refined search has
no queued AML winners. Use breast cancer validation data to select settings.

## Inputs and configuration

Edit `config/config.yaml`. The default keeps the existing breast cancer scalar
selection (correlation 0.33, kNN 3000, 10000 negatives). To sweep the same ranges
as AML, use:

```yaml
corr_threshold: [0.2, 0.33, 0.5, 0.7]
knn_neighbours: [500, 1500, 3000, 4500]
size_negative: [5000, 10000, 15000]
```

`p_raw` points to `raw/`, containing the GSE66695 matrix, annotations and
seeds. Generic foundation checkpoints stay in `model/`.
The breast cancer seed columns remain `probeID`, `deltaBeta`, `P.Value`, `CHR`;
p-value filtering, platform filtering and probe-to-gene expansion are preserved.

## Run

Activate `nanodiag_biomarker_gnn`, then run from this directory:

```bash
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage tune --trials 50
python run_pipeline.py --stage train
python run_pipeline.py --stage xai
python run_pipeline.py --stage compare-predictions
```

Without `--folder`, stages discover all existing dataset folders. Preprocessing
also includes combinations selected in `config/config.yaml`, creating their
folders as needed. To operate on one folder, append:

```bash
--folder data_corr_0.33_knn_n_3000_n_neg_10000
```

`--stage all` runs all five stages in order. Relative `--folder` names are always resolved inside
`breast_cancer_analysis`, independent of the terminal working directory.
Training binds output paths to the selected dataset and rejects model output
paths outside it. `train_gnn/main.py` also accepts `--folder`. `--dry-run` prints commands without
changing files or running jobs. `--models PNA GCN` restricts tuning, training or
comparisons; XAI uses PNA and the first `test_gene_ids` value from its likely-positive
CSV. If that table is missing, it is reconstructed from saved test predictions.
An empty likely-positive table produces a clear error.

Each dataset stores its selected inputs in `preprocessing_config.yaml` and its
validation-selected model settings in `config.yaml`. Tuning updates only that
folder's config. Downstream stages require tuning to have created the config.
Preprocessing skips a completed preprocessing manifest and can rebuild an
incomplete existing folder. Logs are written to `<dataset>/logs/<stage>.log`.

The output layout is:

```text
data_corr_<corr>_knn_n_<knn>_n_neg_<negative>/
  preprocessing_config.yaml
  config.yaml
  preprocessed/
  generated/
    data/       # feature tables, prediction groups, metrics JSON/CSV
    plots/      # PDFs, PNGs and other figures
    models/     # fitted models, graph tensors and all .pt files
    optuna/     # independent study databases, trials and best exports
  evaluation/   # prediction-comparison tables, plots, reports and extracted groups
  logs/
```

Existing flat generated artifacts are moved into these subfolders when the
pipeline uses the dataset; existing destination files are never overwritten.
`artifact_migration.json` records moves and `config.before_artifact_layout.yaml`
backs up the previous dataset config. Readers retain support for flat legacy
study exports. Standalone scripts also accept `--config <dataset>/config.yaml`.

See [HYPERPARAMETER_OPTIMIZATION.md](HYPERPARAMETER_OPTIMIZATION.md) for study
settings. Completed matching studies can create dataset configs without trials:

```bash
python train_gnn/tune_models.py --export-existing
```

## Checks

```bash
python -m unittest discover -s preprocess_raw_data -p 'test_*.py'
python -m unittest discover -s train_gnn -p 'test_*.py'
```
