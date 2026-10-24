# AML methylation data preparation

Run with Python 3.10+ (standard library only), from any directory:

```bash
python /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/preprocess_raw_data/download_geo_datasets.py
python /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/preprocess_raw_data/harmonise_geo_datasets.py
```

Both commands accept `--data-dir`; harmonisation also accepts `--output-dir`.
Downloads are cached, checked for gzip integrity and recorded with source URLs,
file sizes and SHA256 checksums in `raw/raw/download_manifest.json`.
Use `--force` to replace cached downloads. Interrupted transfers never replace
completed files.

Sources:
- [GSE50660](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50660): peripheral blood controls (includes smoking variation).
- [GSE58477](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58477): only samples annotated `cell type: leukemic blast` are selected. Its normal bone marrow progenitor samples are excluded.

Both datasets must report platform GPL13534. Scripts use published processed
Series Matrix beta values, rather than raw IDAT intensities. They do not imply
identical original normalisation or perform raw-array quality control.

Outputs in `raw/harmonised/`:
- `combined_beta.tsv.gz`: one compressed tabular file with `ID_REF` followed by control GSM columns and AML GSM columns. Rows are shared `cg` CpG probes, aligned by ID in GSE50660 order, matching the breast cancer `p_arrays` input layout. Pandas can read the gzip directly with `sep="\t"`.
- `sample_metadata.tsv`: selected samples in exactly the matrix column order, with group, numeric label (healthy=0, AML=1), study, tissue and all supplied characteristics.
- `all_sample_metadata.tsv`: includes excluded samples and selection status.
- `harmonisation_report.json`: sample counts, shared/retained probe counts and processing choices.

Harmonisation retains the intersection of CpGs and removes rows missing a beta
value in any selected sample. Other nonnumeric values and numeric values outside
[0,1] cause an error; values are never clipped or imputed. Duplicate probe/sample
identifiers, mismatched metadata and truncated matrices also cause an error.
SQLite and streaming gzip keep the full combined matrix out of memory; allow
disk space for raw downloads, a temporary AML lookup database and the output.

Study and disease group are perfectly confounded, and peripheral blood controls
differ in cellular composition from bone marrow leukemic blasts. Automatic
ComBat or study centering cannot distinguish these effects from disease and is
therefore not applied. This is structural harmonisation of published betas,
not removal of technical or biological confounding. Downstream differential
methylation and predictive results need independent validation. Probe annotation,
SNP/cross-reactive/sex-chromosome filtering, seed selection and GNN training remain
separate later analysis steps; the breast cancer disease-specific seeds and
configuration have not been copied.

Run the fixture checks:

```bash
python -m unittest discover -s /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/preprocess_raw_data -p 'test_*.py'
```

## Annotate combined matrix identifiers

```bash
python /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/preprocess_raw_data/annotate_combined_beta.py
python /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/preprocess_raw_data/annotate_combined_beta.py --apply
```

The first command assesses coverage; `--apply` writes
`raw/harmonised/combined_beta_cpg_gene.tsv.gz`. The first column remains `ID_REF`
for compatibility, with values formatted `CpG_gene`. Other columns and all beta
values are preserved. The original combined matrix is retained.

The script uses the GPL13534 platform table's `UCSC_RefGene_Name` annotation,
extracted into `data/GPL13534_probe_annotation.tsv`. The cache is sufficient for
subsequent runs; without it, supply the downloaded GPL13534 SOFT family archive
with `--platform`. This is the legacy hg19 manufacturer/platform annotation,
not a new genome annotation or an experimentally established target-gene map.
Repeated gene symbols are deduplicated. Each CpG-gene pair gets its own row:
`cg12345678_GENE1;GENE2` becomes `cg12345678_GENE1` and
`cg12345678_GENE2`, with identical beta values. CpGs without a RefGene assignment
are excluded from the annotated matrix. The original CpG-only matrix retains
all probes. Expanded rows sharing a CpG should stay in the same partition when
splitting features for modelling to avoid leakage.

`cpg_gene_mapping.tsv` preserves the original ID, gene symbols, renamed ID,
chromosome, hg19 position and annotation status. The annotation report records
coverage and a SHA256 check confirming unchanged beta-value text. Gene symbols
may differ from symbols in the literature's supplementary table because the
annotation sources differ. Use the mapping table to join by original CpG ID.

## AML graph preprocessing, tuning and model stages

From `aml_analysis`, with `nanodiag_biomarker_gnn` active:

```bash
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage tune --trials 50
python run_pipeline.py --stage train
python run_pipeline.py --stage xai
python run_pipeline.py --stage compare-predictions
```

Without `--folder`, stages select all dataset folders. Preprocessing also creates
combinations configured in `config/config.yaml`. To select one folder, append:

```bash
--folder data_corr_0.33_knn_n_3000_n_neg_10000
```

`--dry-run` prints commands without changing files or running stages. `--stage all`
runs all five stages in order. `--models PNA GCN` selects models for tuning,
training or comparisons. XAI selects the first `test_gene_ids` from
`generated/data/pred_likely_pos_aml_PNA.csv`; missing groups are reconstructed from
saved predictions. Empty likely-positive groups produce a clear error.

Configuration is separated by stage:

- `config/config.yaml`: AML preprocessing inputs, seeds, filters, 530 input
  dimensions, splits and the existing 48 correlation/kNN/negative-count combinations.
- `config/optuna.yaml`: AML search spaces and `runtime_defaults` for training,
  XAI, plots and output templates.
- `<dataset>/preprocessing_config.yaml`: that folder's fixed preprocessing context.
- `<dataset>/config.yaml`: created after tuning, with only completed model winners,
  seed provenance and normal training artifact paths. Training, XAI and comparisons
  use this config. Tuning never writes model settings into the shared preprocessing config.

The existing completed AML studies have been exported into
`data_corr_0.33_knn_n_3000_n_neg_10000/config.yaml` for all eight tuned models.
Other folders receive their configs when tuning completes. A missing config
requires tuning first; no settings are borrowed from another dataset.

```text
<dataset>/
  preprocessing_config.yaml
  config.yaml
  preprocessed/
  generated/
    data/       # features, prediction tables, metrics JSON/CSV
    plots/      # PDF, PNG and other figures
    models/     # fitted models, graph tensors and all .pt files
    optuna/     # independent study databases, trials and winner exports
  evaluation/   # comparison tables, plots, reports and extracted groups
  logs/         # one log per pipeline stage
```

Existing flat generated artifacts are migrated without overwriting destinations.
Moves are recorded in `artifact_migration.json`; the old dataset config is backed
up as `config.before_artifact_layout.yaml`. Completed preprocessing manifests are
skipped; incomplete existing folders can be rebuilt. NeDBIT/APU executables must
remain on the selected Python environment's PATH.

AML inputs and seed semantics are retained: the harmonised, annotated beta matrix
and aligned sample metadata, 464 controls plus 62 AML samples, four NeDBIT features,
and `raw/aml_top200_cpg_gene_delta_beta.tsv` with 142 gene-annotated seed CpGs.
Seed weights use `deltaBeta_AML_minus_Normal`; manifests reject stale seed inputs.
The original preprocessing/model config and search config are backed up in
`config/config.before_stage_split.yaml` and `config/optuna.before_stage_split.yaml`.
Generic TabPFN/TabICL checkpoints are shared from `../breast_cancer_analysis/model/`;
fitted AML models remain separate.

All mappings of each CpG stay in one partition. AML sample grouping follows its
metadata. External comparisons continue using GSE40279 healthy blood and GSE62298
AML, and calculate AML-minus-healthy beta differences. No BC seed identities,
cohort dimensions or evaluation datasets are substituted.

The model predicts five propagated priority classes for CpG-gene nodes; it does
not classify patients. The cohort-confounding limitations above remain applicable.

Standalone training, XAI and comparison runners accept `--config <dataset>/config.yaml`.
Existing completed Optuna studies can be exported without additional trials:

```bash
python train_gnn/tune_models.py --export-existing --corr-threshold 0.33 --knn-neighbours 3000 --size-negative 10000
```

## Top 100 hypermethylated and 100 hypomethylated CpGs

Run `preprocess_raw_data/select_aml_differential_cpgs.py` with the
`nanodiag_biomarker_gnn` Python environment. It uses the original, unexpanded
`combined_beta.tsv.gz` with all complete shared CpGs and matches groups by sample
metadata (464 GSE50660 controls, 62 GSE58477 AML patients). It applies strict
control mean beta <0.1 or >0.9, then control sample SD <0.05 (`ddof=1`), and ranks
positive AML-minus-control differences descending and negative differences
ascending. Ties use CpG ID ascending. No additional SNP, chromosome, cross-reactive
or gene-annotation filters are added to these requested selection rules.

Outputs in `data/`:
- `aml_top200_cpg_gene_delta_beta.tsv`: 200 unique CpGs, hyper first then hypo,
  with CpG_ID, Gene, deltaBeta_AML_minus_Normal and CpG_Gene columns.
- `aml_top100_hyper_cpg_gene_delta_beta.tsv` and
  `aml_top100_hypo_cpg_gene_delta_beta.tsv`: the separate 100-CpG lists.
- `aml_top200_cpg_statistics.tsv`: means, SD, direction and rank for each selected CpG.
- `aml_eligible_cpg_statistics.tsv.gz`: all CpGs satisfying both control filters.
- `aml_top200_selection_report.json`: selection counts, sources and conventions.

The delta sign is AML minus control, opposite to the original literature-derived
`aml_cpg_gene_delta_beta.tsv` (control minus AML). The old seed file and GNN
configuration are not changed by this selection script. Gene annotation is added
after ranking; missing gene symbols remain blank and multiple gene symbols are
semicolon-separated, so the output remains exactly 200 unique CpGs. The requested
formula selection is applied to our complete-case harmonised matrix; it is not a
claim of an exact match to an unpublished original analysis script. The paper does
not specify the SD degrees-of-freedom convention in its methods; `--sd-ddof 0`
is available for sensitivity analysis.

## Improved PNA

The completed validation-selected experiment is in `pna_improvement/`, with the
checkpoint `best_model_PNA.pt`, test predictions, source graph snapshot, four
candidate validation histories, a before/after heatmap and `REPORT.md`.
The original comparison and model files remain unchanged.

The selected model uses two residual PNA layers with a direct original-feature
path, width 128, dropout 0.3, lr 0.0005, weight decay 0.005 and sampled neighbours
[25,10]. It uses square-root inverse-frequency loss weights computed only on
training nodes. Validation macro F1 selects the checkpoint. Both default AML configs now apply these settings to PNA through
`model_overrides.PNA`. Other model families retain their existing settings.
Configs without this override still load the original PNA architecture.

To train with these settings using the standard entry point:

```bash
conda activate nanodiag_biomarker_gnn
python /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/train_gnn/main.py --config /home/ubuntu/data/anup/nanodiag-biomarker/test-train/nanodiag_biomarker_prediction/aml_analysis/config/config_pna_improved.yaml
```

The improved config writes new results to `pna_improvement/generated/`.
To repeat the four-candidate search, run `train_gnn/tune_pna.py --output` with a
fresh directory. It snapshots the stored original graph, checks CpG/edge partition
separation, selects on validation macro F1 and evaluates only the winning model
on test. This remains a single-seed result using propagated targets.


The normal `run_pipeline.py --stage train` and `train_gnn/main.py` commands now
use improved PNA automatically. PNA training and explanation share the per-model
settings resolver. New default PNA artifacts go to
`${p_base}generated_files_pna_improved/`; the existing `generated/` results
are preserved. The separate completed experiment remains in `pna_improvement/`.
These changes affect model training; graph preprocessing and seed selection are
unchanged. Both default configs retain their existing correlation thresholds.

## Preprocessing parameter sweep

Edit only `config/config.yaml`. `corr_threshold`, `knn_neighbours`, and
`size_negative` accept a scalar or a list. Lists run every combination;
scalars select one dataset. `p_base` is constructed automatically from these
three values for preprocessing, validation, and training.

```yaml
corr_threshold: 0.5
knn_neighbours: 1500
size_negative: 5000
```

This selects `data_corr_0.5_knn_n_1500_n_neg_5000/`. No config files are
written inside dataset folders. The shared inputs and outputs are organized as:

```text
aml_analysis/
  config/config.yaml
  raw/
    harmonised/
    raw/                       # downloaded GEO files
    GPL13534_probe_annotation.tsv
    aml_top200_cpg_gene_delta_beta.tsv
    snp_7998probes.vh20151030.txt
    cross_reactive_probes.csv
  data_corr_0.5_knn_n_1500_n_neg_5000/
    preprocessed/
    generated/
```

Run in the project environment from `aml_analysis`:

```bash
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage train
```

Use `--stage all` to preprocess, validate, and train each selected combination.
Standalone preprocessing, validation, and training scripts read the same shared
config and construct the same folders. Logs are saved under `logs/` with the
parameter folder name. All raw inputs are shared across parameter combinations;
preprocessing recomputes the graphs, NeDBIT features, labels, and manifests in
`preprocessed/`, while model artifacts go into `generated/`.

Preprocessing skips a configuration whenever its `data_corr_*` folder already
exists, including empty or partially processed folders. Remove that folder to
rerun preprocessing for that combination. With `--stage all`, training still
runs for existing configurations; preprocessing and its validation are skipped.

Negative feature selection and CpG-level train/validation/test partitions are
reproducible from `SEED` for the same input data and selection method. The same
negative count always selects the same features and partitions across correlation
thresholds and KNN sizes. Canonical ordering also removes dependence on input
feature order. Negatives use one ranked pool, with seeded tie-breaking, so the
5,000-feature set is a subset of the 10,000-feature set, which is a subset of the
15,000-feature set. Separate seeded random generators isolate positive and
negative splits from selection and other random operations. CpG mappings always
stay together. Split membership may differ between different negative counts.
Existing data folders are still skipped; delete a folder to regenerate it using
this deterministic selection and splitting implementation.

## Optuna model optimization

See [HYPERPARAMETER_OPTIMIZATION.md](HYPERPARAMETER_OPTIMIZATION.md) for the
GNN, XGBoost, CatBoost and optional TabPFN/TabICL search spaces and saved outputs.
After preprocessing, run in the project environment:

```bash
python -m pip install -r train_gnn/requirements-optuna.txt
python run_pipeline.py --stage tune --trials 50
```

Each selected dataset/model has an independent, resumable validation-only study.
The editable search settings are in `config/optuna.yaml`.

## Explain each configured dataset

The explainer expands `corr_threshold`, `knn_neighbours`, and `size_negative`
from the shared config just like training. It processes each combination in
sequence, using the GNN named by `best_trained_model` and its per-model settings.
Models and prediction artifacts must already have been generated by training.
Each configuration writes explanations to its own `generated/` directory.

```bash
python train_gnn/run_xai_explainer.py
```

The default node ID remains 8140, or `xai_node_id` if set in the shared config.
Node IDs may change between negative sets. To explain the same CpG-gene feature
across configurations, supply its full name (it must exist in each selected
configuration):

```bash
python train_gnn/run_xai_explainer.py --node-name cg00000001_GENE
```

Replace the example with a real node name. Alternatively use `--node-id 123`.
The script reports an error for missing names or out-of-range node IDs. It
supports `--config` and the same single-combination overrides as preprocessing:

```bash
python train_gnn/run_xai_explainer.py \
  --corr-threshold 0.33 --knn-neighbours 3000 --size-negative 10000 --node-id 123
```

This script uses GNNExplainer and supports PNA, GCN, GraphSAGE, GATv2 and
GraphTransformer. It reports an error if `best_trained_model` selects a tabular
estimator.

Prediction comparisons preserve existing XAI-exported group CSVs and record their
source paths/hashes. They do not reapply a separate default selection rule to
existing exports. Folder arguments resolve relative to `aml_analysis` regardless
of shell cwd; training binds outputs to the selected dataset and rejects escapes.
The training entry point also accepts `--folder`.

### Compare dataset overlap

```bash
python run_pipeline.py --stage compare_data_overlap
```

Generates three comparisons:

- Vary `n_neighbours`, fixing `corr=0.33` and `num_negatives=10000`.
- Vary `num_negatives`, fixing `corr=0.33` and `n_neighbours=3000`.
- Vary `corr`, fixing `n_neighbours=3000` and `num_negatives=10000`.

Writes one PNG per comparison containing separate train, test, and validation bar charts to
`evaluation/data_overlap/`, along with `membership_regions.csv` and
`pairwise_overlap.csv` (intersection, union, and Jaccard similarity).
Each bar represents an exclusive membership region across parameter settings;
percentages use the union of distinct probe–gene IDs for that split.
IDs come from final retained nodes in `preprocessed/out_genes_aml_*.csv` and are
matched by probe–gene name, rather than local numeric ID.
Missing split files are reported in `skipped_datasets.csv`.
This stage runs explicitly and does not require tuned model configs.
