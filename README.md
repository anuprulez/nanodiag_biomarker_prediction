### GraphMeXplain: Prioritising DNA methylation biomarkers using graph neural networks and explainable AI

CpG site prioritisation from the unknown sites using already available markers by propagating information through a CpG-CpG site interaction network

### Environment

Install (via conda) the following:

```
conda env create -f env.yml
```

### Steps to run

### Breast cancer

#### Using raw datasets 

- Set `p_raw_data` to `true` in `config.yaml` (All required datasets will be downloaded from [Zenodo](https://zenodo.org/records/17191564/files/bc_raw_datasets_to_preprocess.zip?download=1)
- Run `breast_cancer_analysis/preprocess_raw_data/process_raw_illumina_arrays_bc.py` to preprocess arrays and soft label assignment
- Run `breast_cancer_analysis/train_gnn/main.py` for training Graph neural network
- Run `breast_cancer_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis

#### Using preprocessed datasets 

- Set `p_raw_data` to `false` and `download_preprocessed_data` to `true` in `config.yaml`. Preprocessed datasets will be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.19050630)
- Run `breast_cancer_analysis/train_gnn/main.py` for training Graph neural network
- Run `breast_cancer_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis



#### Training runs and cumulative plots

Run these commands from `breast_cancer_analysis` in the project environment.
The dataset must already contain a tuned `config.yaml`; use the preprocessing
and tuning stages first for a new dataset.

```bash
cd breast_cancer_analysis
python run_pipeline.py --stage train --folder data_corr_0.33_knn_n_3000_n_neg_10000
python run_pipeline.py --stage accumulate --folder data_corr_0.33_knn_n_3000_n_neg_10000
```

Each successful pipeline training invocation saves the next numbered archive
under that dataset's `runs/` directory. Numbering starts at `run1` and continues
from the highest existing run number. Failed training does not publish an archive.
The current outputs remain available in `generated/` for prediction comparison
and XAI; earlier archives are preserved.

```text
<data-folder>/
├── config.yaml
├── generated/                 # Current artifacts
├── logs/train.log             # Current training log
├── runs/
│   ├── run1/
│   │   ├── config.yaml        # Training config snapshot
│   │   ├── run.json           # Models trained in this invocation
│   │   ├── generated/         # Saved data, models, and plots
│   │   └── logs/train.log
│   └── run2/
└── accumulated_results/
    ├── data/                  # Metric summaries and included runs
    └── plots/                 # Cumulative PDF figures
```

`--stage accumulate` reads completed `runs/run<number>/` archives without
training or rerunning external prediction comparisons. It calls the cumulative
routine in `train_gnn/plot_aggregations.py`, which uses
`train_gnn/plot_gnn.py` for the average model-versus-metric heatmap. Outputs include:

- Average metric heatmap and radar plot across saved runs.
- One box plot per heatmap metric across runs, comparing all selected models:
  validation/test accuracy, macro/weighted/micro F1, weighted precision, and
  weighted recall. Individual run points and contributing run counts are shown.
  PNG/PDF figures are saved as `plots/boxplot_<metric>_across_runs.*`; their
  observations are exported to `data/accuracy_metrics_by_run.csv`. This applies
  to both breast cancer and AML accumulation.
- PNA likely-positive CpG overlap pie chart (PNG/PDF): unique CpGs predicted in
  exactly one, two, or more included runs, with the all-run intersection count.
  Multiple gene annotations for the same CpG count once per run. Supporting
  membership/count CSVs and a source-run JSON report are saved in `data/`.
  Runs without a PNA likely-positive export are reported and excluded.
- Per-model mean/std loss and accuracy curves, where histories are available.
  Different early-stopping lengths are supported; each epoch uses the runs
  with values for that epoch.
- `metrics_mean_std.csv`, per-history mean/std CSVs with contributing run counts,
  and `included_runs.json` recording the archives used for each model.

Use `--models PNA XGBoost` with training or accumulation to select models.
Accumulation otherwise uses the models in the tuned dataset config and includes
only runs whose manifests list each model. If no completed archives exist, it
reports that cumulative plots were skipped. Direct `train_gnn/main.py` calls
do not create these archives; use `run_pipeline.py --stage train`.

```bash
# Accumulate all configured dataset folders.
python run_pipeline.py --stage accumulate

# Preview commands without changing files.
python run_pipeline.py --stage accumulate --folder data_corr_0.33_knn_n_3000_n_neg_10000 --dry-run

# Compare current predictions and then generate cumulative plots from saved runs.
python run_pipeline.py --stage compare-predictions --folder data_corr_0.33_knn_n_3000_n_neg_10000
```

`compare-predictions` defaults to all tuned models; `--models` limits the selection.
Its cumulative output also goes to `accumulated_results/`. The `all` stage
already reaches cumulative plotting through `compare-predictions`; it does not
run the separate `accumulate` stage again. See the
[evaluation README](breast_cancer_analysis/evaluation/README.md) for external
prediction comparison details.


### AML

#### Training archives, accumulation, and saved-run comparison

From the repository root, run AML training through the pipeline to create a
numbered archive automatically:

```bash
cd aml_analysis
python run_pipeline.py --stage tune --trials 25 --folder data_corr_0.33_knn_n_3000_n_neg_10000/
python run_pipeline.py --stage train --folder data_corr_0.33_knn_n_3000_n_neg_10000/
```

After training succeeds, the pipeline copies the generated artifacts and training
log into `<data-folder>/runs/run<number>/` and prints `saved training run:` with
the archive path. Each archive contains `generated/`, `logs/train.log`,
`config.yaml`, and `run.json`. Numbering continues from the highest existing run
number; for a dataset with `run1`, the next successful training creates `run2`.
Current outputs remain in the dataset's `generated/` folder. Copying large
artifacts can take several minutes after model training finishes; wait for the
pipeline's final completion message. Failed training does not publish an archive.

```bash
# Cumulative plots and CpG overlap from completed saved runs.
python run_pipeline.py --stage accumulate --folder data_corr_0.33_knn_n_3000_n_neg_10000/

# Compare the saved run's PNA predictions.
python run_pipeline.py --stage compare-predictions \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000/ --run run1 --models PNA

# Compare current generated/ predictions instead.
python run_pipeline.py --stage compare-predictions \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000/ --models PNA
```

Accumulated outputs go to `<data-folder>/accumulated_results/`. Saved-run
comparisons go to `<data-folder>/evaluation/run1/`; current comparisons go to
`<data-folder>/evaluation/`. `--run 1` and `--run run1` are equivalent.
Direct `python train_gnn/main.py` calls do not create run archives. The same
archive behavior applies when using the pipeline in `breast_cancer_analysis`.

See the [AML pipeline README](aml_analysis/README.md) for training archives,
`--stage accumulate`, saved-run comparison with `--run`, and plot updates.

- Set `p_raw_data` to `true` in `config.yaml` (All required datasets will be downloaded from [Zenodo](https://zenodo.org/records/17202627/files/aml_raw_datasets.zip?download=1)
- Run `aml_analysis/preprocess_raw_data/preprocess_HumanMethylation450_dataset.py` to preprocess arrays and soft label assignment
- Run `aml_analysis/train_gnn/main.py` for training Graph neural network
- Run `aml_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis

#### Using preprocessed datasets (faster)

- Set `p_raw_data` to `false` and `download_preprocessed_data` to `true` in `config.yaml`.  Preprocessed datasets will be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.19050582)
- Run `aml_analysis/train_gnn/main.py` for training Graph neural network
- Run `aml_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis


### Model benchmarks

Select `tabpfn`, `tabiclv2`, `xgboost`, `catboost`, or a GNN with `model_type` in the existing config and run `main.py` in the same conda environment. Compare all nine architectures on the same fixed splits using the [benchmark scripts and instructions](breast_cancer_analysis/train_gnn/BENCHMARKS.md). The workflow produces validation-selected results, per-node predictions, and a combined CSV/Markdown report.

### Preprint

[Prioritizing DNA methylation biomarkers using graph neural networks and explainable AI](https://www.biorxiv.org/content/10.64898/2026.01.26.701692v1.abstract)

### Plots

#### AML 

Top-20 Likely positive predicted probes

![Violin plot](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/plots/violin_aml.png)

PNA: Confusion matrix

![Confusion matrix](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/plots/confusion_pna_aml.png)

xAI for "cg23281527_KLHDC7A"

![xAI](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/plots/plots_xai_aml.png)




### Compare predictions from a saved training run

Run from `breast_cancer_analysis`:

```bash
python run_pipeline.py --stage compare-predictions \
  --folder data_corr_0.33_knn_n_3000_n_neg_10000 --run run2 --models PNA
```

`--run 2` is equivalent to `--run run2`. The command reads the archived config
and prediction/model artifacts under `<dataset>/runs/run2/`; it does not retrain
models. Results go to `<dataset>/evaluation/run2/`. Without `--run`, comparisons
continue to use the current configured `generated/` artifacts and write to
`<dataset>/evaluation/`. The run flag is also available on
`evaluation/run_comparisons.py`. If `--models` is omitted with `--run`, the models
recorded in that run's manifest are selected. Missing runs or models absent
from the selected run raise an error rather than falling back to current files.
When `--folder` is omitted, the selected run must exist in each selected dataset.

## Select XAI targets (AML and Breast Cancer)

Run from either `aml_analysis` or `breast_cancer_analysis`. These flags are
identical in both pipelines:

```bash
# Highest-probability LP (default).
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --xai-group LP
# Highest-probability RN from pred_negatives.
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --xai-group RN
# Explicit graph node ID (replace with an ID for your dataset).
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --node-id 7980
# Explain several top predictions.
python run_pipeline.py --stage xai --folder data_corr_0.33_knn_n_3000_n_neg_10000 --xai-group RN --top-k 5
```

Automatic selection ranks `pred_probs` descending in the chosen prediction
export. Explicit node IDs override that selection. All per-node figures,
including the graph/bar/Δβ composite, are generated by the XAI stage.
