# Methylation biomarker analysis: breast cancer and AML

This repository provides parallel workflows for breast cancer (BC) and acute
myeloid leukemia (AML): methylation preprocessing, CpG graph construction,
NIAPU label propagation, graph-model training, prediction explanation, external
methylation comparison, and cross-cohort expression follow-up.

This overview consolidates the [BC README](breast_cancer_analysis/README.md) and
[AML README](aml_analysis/README.md). Consult those documents for disease-specific
configuration, statistical models and detailed output descriptions.

## Project organization

| Directory | Purpose |
| --- | --- |
| [breast_cancer_analysis/](breast_cancer_analysis/README.md) | BC pipeline, configuration and evaluation. |
| [aml_analysis/](aml_analysis/README.md) | AML pipeline, configuration and evaluation. |
| [post_evaluation_support/](post_evaluation_support/README.md) | Shared patient-classifier, expression, publication-plot and validation code. |

Each disease has its own `run_pipeline.py`. Run the commands below **inside the
chosen disease directory**. Relative dataset folders resolve under that directory;
absolute dataset paths are also accepted.

## Environment and inputs

Use the project ML environment with compatible PyTorch/PyG, sampling extensions,
NumPy, pandas, SciPy, scikit-learn, matplotlib, umap-learn, OmegaConf and the
selected model dependencies. Dependency lists are under each disease's
`train_gnn/` directory:

- `requirements-benchmark.txt`: benchmark/model dependencies.
- `requirements-sampling.txt`: graph sampling dependencies.
- `requirements-optuna.txt`: tuning dependencies.
- `requirements-plotting.txt`: plot composition dependencies.

Original methylation, annotation, seed and sample-metadata files must be present
at the configured paths before preprocessing. `config/config.yaml` controls
preprocessing; `config/optuna.yaml` defines search spaces and runtime defaults.
Preprocessing saves `<dataset>/preprocessing_config.yaml`; tuning creates the
dataset-specific `config.yaml`. Foundation-model checkpoints must be available
locally or accessible through the configured downloader.

Cross-cohort expression analysis additionally requires **R with limma and statmod**.
The optional environment below supports that analysis, rather than the entire
GNN workflow. Run from a disease directory:

```bash
conda env create -f ../post_evaluation_support/environment-cross-cohort.yml
conda activate nanodiag-cross-cohort
```

## Pipeline commands

Choose a disease directory from the repository root, then set the dataset:

```bash
cd breast_cancer_analysis
# For AML, use: cd aml_analysis  (from the repository root)
dataset_folder=data_corr_0.33_knn_n_3000_n_neg_10000
```

The main stage order is:

`preprocess → tune → train → xai → compare-predictions → accumulate → post-evaluation`

### 1. Preprocess, tune and train

```bash
python run_pipeline.py --stage preprocess --folder "$dataset_folder"
python run_pipeline.py --stage tune --folder "$dataset_folder" --models PNA --trials 25
python run_pipeline.py --stage train --folder "$dataset_folder" --models PNA
```

Tuning uses training/validation partitions; repeated tuning calls resume studies
and add trials. Each successful pipeline training invocation creates a new
`runs/run<number>/` archive containing configuration, generated artifacts and
training logs. Current artifacts remain under `generated/`.

The fixed-panel post-evaluation requires **five completed PNA runs**. For a new
dataset, execute the training command five times in total. Check existing archives
first: another training invocation creates another run. Direct calls to
`train_gnn/main.py` do not create pipeline archives.

### 2. Prepare the external methylation reference

Run once before external comparison or explanation figures that require it:

```bash
python evaluation/download_datasets.py
python evaluation/evaluate_datasets.py
python evaluation/validate_results.py
```

These helpers reuse downloaded files at their expected cache paths. See the
[BC evaluation README](breast_cancer_analysis/evaluation/README.md) or
[AML evaluation README](aml_analysis/evaluation/README.md) for details.

### 3. Explain and compare predictions

```bash
python run_pipeline.py --stage xai --folder "$dataset_folder" --xai-group LP --top-k 5
python run_pipeline.py --stage compare-predictions --folder "$dataset_folder" --models PNA
```

XAI uses PNA. `LP` selects likely-positive predictions; `RN` selects predicted
negatives. `--node-id` selects a particular graph node instead of automatic ranking.
For comparison of an archived run:

```bash
python run_pipeline.py --stage compare-predictions \
  --folder "$dataset_folder" --models PNA --run run1
```

### 4. Accumulate and run post-evaluation

After completing the required runs:

```bash
python run_pipeline.py --stage accumulate --folder "$dataset_folder" --models PNA
python run_pipeline.py --stage post-evaluation --folder "$dataset_folder" --models PNA
```

Accumulation rebuilds archived comparisons and run summaries without training.
Post-evaluation uses the exact frozen panel: **33 CpGs for BC** and **17 for AML**.
It verifies methylation effects, prepares genomic annotation and Reactome
analysis when resources are available, compares patient classifiers, and runs
cross-cohort methylation/expression analysis. Incompatible panel identities or
sizes fail explicitly.

The patient-classifier comparison uses Random Forest, XGBoost, TabPFN and TabICL,
with original-patient five-fold CV and final original-trained models applied to
the external cohort. It exports ROC, precision–recall and confusion-matrix plots.

To run only the complete cross-cohort expression workflow:

```bash
python run_pipeline.py --stage post-evaluation \
  --folder "$dataset_folder" --post-evaluation-analysis cross-cohort
```

Missing expression resources are downloaded, matrices prepared, limma fitted,
CpG/gene effects compared, results validated and publication figures exported.

## NIAPU rankings and UMAP

This separate stage provides both an exploratory patient-classifier comparison
using NIAPU likely positives and a four-panel UMAP comparison:

```bash
python run_pipeline.py --stage niapu-rankings --folder "$dataset_folder" --run run1
```

It is not automatically included in `--stage all`. The `naipu-rankings` spelling
is accepted as an alias. Omit `--models`; the patient-classifier suite is fixed.

### Likely-positive patient-classifier comparison

Reads **NIAPU class 2 only**, without a GNN probability filter, top-N cutoff or
five-run intersection. Multiple gene annotations are collapsed to unique CpGs.
Externally measured sites must satisfy all of:

- Absolute external delta beta ≥ 0.30.
- External Welch P < 0.01.
- BH FDR < 0.01 over the full shared external CpG test universe.
- Matching nonzero original/external methylation directions.

The selected panel size is determined by these criteria, rather than fixed at
33 or 17. Random Forest, XGBoost, TabPFN and TabICL classify disease versus healthy
samples using the same patient-comparison framework. To run this portion without
a trained PNA archive:

```bash
python run_pipeline.py --stage niapu-rankings --folder "$dataset_folder" --skip-umap
```

### Publication UMAP comparison

Requires one archived PNA run, rather than five runs or external methylation
resources. To regenerate only these figures:

```bash
python run_pipeline.py --stage naipu-rankings \
  --folder "$dataset_folder" --run run1 --umap-only
```

All panels use **all held-out test CpG–gene nodes**, without a likely-positive
filter. Nodes represent CpG/gene annotations, not patients; multiple nodes can
share a CpG. Labels and node IDs are verified against archived artifacts.

| Panel | Subplot title | Input representation |
| --- | --- | --- |
| A | NIAPU labels, NeDBIT features | Configured graph-derived NeDBIT features. |
| B | NIAPU labels, NeDBIT + DNA methylation features | Combined feature blocks. |
| C | NIAPU labels, DNA methylation features | Patient methylation measurements describing each node. |
| D | PNA labels, Final PNAConv representation | Saved final PNA message-passing output, colored by PNA predictions. |

Panel D uses the learned output after the final residual PNAConv and LayerNorm,
not raw combined features or class probabilities. It has 256 dimensions in the
reference `run1` models and is projected separately with UMAP. Panels A–C use
median-imputed, standardized feature columns with equal total variance per block;
panel D uses saved latent values without additional scaling.

UMAP parameters come from the selected archived run. Reference `run1` settings
for both diseases are **20 neighbors, min_dist=0.8, correlation distance, seed 32**.
Colors follow the existing training UMAPs: positive seed blue, likely positive
orange, weakly negative green, likely negative red and reliable negative purple.
Figures omit scores and subtitles; quality scores remain in a separate TSV.
Exports include PDF, editable SVG and 300-dpi PNG, with individual panel files.

## Datasets

Sample counts below describe the analysis sets documented in the disease READMEs.

| Role | BC | AML |
| --- | --- | --- |
| Original methylation | [GSE66695](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66695): 80 tumors, 40 normals. | [GSE58477](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58477): 62 AML; [GSE50660](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50660): 464 blood controls. |
| External methylation | [GSE226569](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226569): 694 tumors; [GSE101961](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE101961): 121 controls. | [GSE62298](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62298): 68 AML; [GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279): 656 controls. |
| Independent expression | [TCGA-BRCA HiSeqV2 via UCSC Xena](https://xenabrowser.net/datapages/?dataset=TCGA.BRCA.sampleMap%2FHiSeqV2&host=https%3A%2F%2Ftcga.xenahubs.net): selected 775 tumors, 84 normals. | [GSE15061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15061): 202 AML, 69 non-leukemic marrow samples. |

Cross-cohort plots use **X = expression log2FC** and **Y = original-cohort
methylation delta beta**. BC uses a patient-blocked limma model with paired-patient
sensitivity analysis. AML uses DS records only, excluding duplicate DQN3 records
and MDS specimens, with a group-plus-study-center primary model. AML expression
coverage is 16 of 17 frozen CpGs: cg07911080–PRHOXNB/URAD lacks an unambiguous
expression probe. AML expression controls are non-leukemic marrow, not necessarily
healthy volunteers.

## Downloads and reruns

Expected source files and compatible caches are reused. Missing external resources
are downloaded; UMAP-only execution uses archived local artifacts and does not
download an external cohort. Input/settings fingerprints govern reuse of model,
expression and UMAP caches. A final plot alone does not replace required sources.

Useful post-evaluation options:

| Option | Purpose |
| --- | --- |
| `--offline` | Require cached resources; fail if required caches are missing or incompatible. |
| `--refresh-expression` | Rebuild expression matrices and refit limma while reusing downloaded sources. |
| `--rscript /path/to/Rscript` | Choose an R installation. |
| `--pathway-gmt /path/to/ReactomePathways.gmt` | Supply a local pathway resource. |
| `--dry-run` | Print the planned commands without running stages or changing outputs. |

Changing the selected UMAP run replaces displayed figures/tables while retaining
per-run coordinate caches. Training always creates a new archive. Check
`<dataset>/logs/<stage>.log` and generated manifests when diagnosing failures.

## Output layout

Paths are relative to the selected disease dataset directory:

```text
preprocessing_config.yaml
config.yaml
preprocessed/
generated/{data,models,plots}/
runs/run<number>/{config.yaml,run.json,generated/,logs/}
evaluation/
accumulated_results/
  data/
  evaluation/
  plots/
  post_evaluation/
    REPORT.md
    manifest.json
    candidate_cohort_comparison.pdf
    methylation_expression/
      cross_cohort_comparison/{REPORT.md,manifest.json,validation.json,data/,plots/}
  niapu_comparison/
    REPORT.md
    UMAP_REPORT.md
    selection_manifest.json
    umap_manifest.json
    candidate_cohort_comparison.pdf
    data/
      selected_niapu_CpGs.tsv
      niapu_umap_coordinates.tsv
      niapu_umap_quality.tsv
    plots/niapu_umap_comparison.{pdf,svg,png}
    plots/niapu_umap_1.{pdf,svg,png}
    plots/niapu_umap_2.{pdf,svg,png}
    plots/niapu_umap_3.{pdf,svg,png}
    plots/niapu_umap_4.{pdf,svg,png}
    models/
    cache/
logs/<stage>.log
```

## Interpretation

External methylation labels contribute to candidate selection, so external
patient-classification results are exploratory and selection-biased. Original
patient CV is conditional on preprocessing/feature selection already performed.
Study, tissue composition and disease status can be confounded.

Cross-cohort delta beta versus log2FC is a comparison across features, not a
matched-patient methylation–expression correlation. Annotation-based gene mapping
provides regulatory hypotheses and does not establish causation. UMAP separation
is descriptive: NIAPU and NeDBIT share graph information, and PNA is trained to
predict NIAPU classes. Clearer separation does not establish independent biological
validity.

Use the disease READMEs and generated reports/manifests for exact sample handling,
coverage, source hashes, statistical assumptions and run-specific results.
