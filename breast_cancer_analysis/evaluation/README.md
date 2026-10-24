# External breast cancer evaluation: GSE226569 versus GSE101961

The downloaded cohorts contain **694 breast cancer samples** (GSE226569, FFPE
breast tumor tissue) and **121 cancer-free controls** (GSE101961, breast tissue).
Both use the Illumina 450K platform GPL13534.

There are **276,046 common CpGs**, including **206,489 gene-annotated CpGs** and
69,557 unannotated CpGs. Expanding the unique gene assignments produces
**227,589 CpG–gene comparison rows**. The tumor processed matrix contains 276,046
CpGs, all present in the control matrix's 482,420 CpGs. These are measured,
published processed values; no new selection by effect size was applied.

## Tables available now

- [cpg_gene_beta_comparison.tsv](results/cpg_gene_beta_comparison.tsv): one row per
  common annotated CpG–gene pair, both cohort means, sample SDs, valid sample
  counts, **Δβ = breast cancer mean − healthy mean**, chromosome and hg19 position.
- [shared_cpg_statistics.tsv](results/shared_cpg_statistics.tsv): every common
  CpG, including unannotated sites, with the same statistics and all mapped genes.
- [GSE226569_cpg_gene_statistics.tsv](results/GSE226569_cpg_gene_statistics.tsv)
  and [GSE101961_cpg_gene_statistics.tsv](results/GSE101961_cpg_gene_statistics.tsv):
  map every CpG in each cohort independently, including unmatched/unannotated
  sites. Each mapped gene has a separate row; unannotated sites have blank genes.
- `results/GSE*_samples.tsv` and `results/GSE*_metadata.json`: sample membership,
  source tissue, titles, and published characteristics.
- `results/analysis_report.json`, `results/validation_report.json`, and
  `raw/download_manifest.json`: counts, source URLs, checksums and validation.

Preview (CpG ID order, not a significance ranking):

| CpG–gene | Healthy mean β | Breast cancer mean β | Δβ (BC − healthy) |
| --- | ---: | ---: | ---: |
| cg00000289_ACTN1 | 0.694315 | 0.660180 | -0.034135 |
| cg00000321_SFRP1 | 0.265876 | 0.305831 | +0.039954 |
| cg00000714_TSEN34 | 0.113386 | 0.104164 | -0.009222 |
| cg00000721_LRRC16A | 0.946726 | 0.873492 | -0.073234 |
| cg00000807_KLHL29 | 0.849936 | 0.796063 | -0.053873 |

## Regenerate the external reference

From `breast_cancer_analysis`, with `nanodiag_biomarker_gnn` active:

```bash
python evaluation/download_datasets.py
python evaluation/evaluate_datasets.py
python evaluation/validate_results.py
```

These three scripts use the standard library (the downloader calls `curl`).
They accept `--raw-dir`; evaluation additionally accepts `--output-dir`, and
validation accepts `--results-dir`. Downloads reuse and validate cached files;
`--force` requests replacements. Interrupted downloads use `.part` files.

GSE226569's Series Matrix contains metadata only. Its supplementary
`GSE226569_Methy_ProcessedMatrix.csv.gz` has interleaved beta and detection-p-value
columns. Only beta columns are summarized; detection p-values are excluded, and
no extra detection-p-value threshold is applied. All 694 beta column names are
matched exactly to GEO sample titles and reordered to metadata order.
GSE101961 betas are read directly from its Series Matrix.

Gene mapping uses the same GPL13534 `UCSC_RefGene_Name` legacy hg19 annotation as
AML, cached independently here. Multiple genes are deduplicated and expanded.
The first run can reuse AML's platform cache; absent that cache, the downloader
fetches the platform annotation from NCBI. A beta value belongs to a CpG: multiple
gene assignments repeat that site's statistics rather than establish separate
measurements or gene-level average methylation.

Exact CpG IDs define the intersection. Missing beta values are omitted per CpG,
valid n is reported, SD uses n−1, and undefined statistics are blank. No imputation,
new normalization, SNP/sex-chromosome filter, batch correction or hypothesis test
is applied. Non-finite or out-of-range beta values and duplicate probes fail.


Pipeline comparisons run across all dataset folders, or one selected folder:

```bash
python run_pipeline.py --stage compare-predictions
python run_pipeline.py --stage compare-predictions --folder data_corr_0.33_knn_n_3000_n_neg_10000
```

Each dataset exports comparisons to its own `evaluation/` directory. Saved test
partition tables are read from `generated/data/`; prediction tensors come from
`generated/models/`. Legacy flat exports remain readable. Training writes the
likely-positive/negative groups into `generated/data/`, and XAI selects the first
`test_gene_ids` in the PNA likely-positive table.

## Compare likely-positive and negative predictions later

After training and test prediction artifacts have been saved, run:

```bash
python evaluation/compare_predictions.py \
  --data data_corr_0.33_knn_n_3000_n_neg_10000 \
  --model PNA
```

`compare_pna_predictions.py` is an equivalent entry point defaulting to PNA.
Other supported model names: GCN, GraphSAGE, GATv2, GraphTransformer, TabPFN,
TabICL, XGBoost, CatBoost. The training matrix and dataset default to the tuned dataset config, discovered
from `config/config.yaml`. Use `--config <dataset-folder>/config.yaml` to choose
a fixed dataset, or `--matrix` to override its matrix; `--external` and `--output` override reference/output paths.

The script first reads the current `generated/data/pred_negatives_<model>.csv`
and `pred_likely_pos_bc_<model>.csv` exports. These selections are authoritative,
including label-rule changes made in `run_xai_explainer.collect_pred_labels`.
Rows are validated against saved inference IDs, names, labels and probabilities;
float32 CSV rounding is allowed. Comparison copies these files into its evaluation
output and records their source paths and SHA256 hashes. It never overwrites the
XAI exports or applies the default masks to an existing selection. Only when both
exports are absent are groups reconstructed from tensors. After editing XAI
selection code, regenerate its prediction-group exports before comparing.
No GNN explanation run is required if the desired exports already exist. It checks ID/name agreement
with the saved test table before assigning predictions. Required files are:

```text
<dataset>/preprocessed/out_genes_bc.csv
<dataset>/preprocessed/combined_pos_neg_signals_bc.csv
<dataset>/generated/data/test_probe_genes_bc_<model>.csv
<dataset>/generated/models/test_ids_bc_<model>.pt
<dataset>/generated/models/true_labels_bc_<model>.pt
<dataset>/generated/models/pred_labels_bc_<model>.pt
<dataset>/generated/models/best_class_pred_probs_bc_<model>.pt
<training matrix from config p_arrays, currently raw/GSE66695_matrix_processed.txt.gz>
```

When CSV exports are absent, reconstruction uses these fallback rules with one-based labels (existing XAI exports always take precedence):

- Likely positive: predicted label 2, reference propagated label in 2–5.
- Negative: predicted label in 3–5, reference propagated label in 3–5.

Reference labels are propagated model targets, not experimentally validated
biomarker truth. Raw exported label tensors are zero-based; the extractor
converts them to the one-based convention above.

The script derives original sample groups from `BC_`/`NC_` identifiers (case
insensitive), checks exported beta values against the original raw matrix to
verify sample order, and computes original BC-minus-healthy effects. External
matching uses exact CpG ID, retaining predicted and external genes separately
and flagging gene concordance. Multiple genes count once per CpG in plots and
summary statistics. CpGs in both prediction groups are reported explicitly.
At least two finite matched CpGs per group are needed for this plotting workflow.

Outputs under `<dataset>/evaluation/` include:

- Extracted test predictions, all predictions with match flags, matching CpG–gene
  rows, and one-row-per-CpG tables for each group.
- Original versus external Δβ, effect shifts (external − original), direction
  agreement, Pearson/Spearman correlations, and unmatched features.
- Two plot sets as PNG and PDF, summary TSV, README and checksummed JSON report.
- Dataset and model suffixes distinguish outputs across the parameter sweep.

For every combination in `config/config.yaml`, using each dataset's tuned config:

```bash
python evaluation/run_comparisons.py --dry-run
python evaluation/run_comparisons.py --models PNA
```

Multiple models can follow `--models`. The runner supports `--config` and the
same scalar `--corr-threshold`, `--knn-neighbours`, `--size-negative` overrides as
training. It stops on missing artifacts; run it after the selected jobs finish.
Each dataset config must have been created by tuning before these commands run.

## Validation and interpretation

```bash
python -m unittest discover -s evaluation -p 'test_*.py'
```

Seven regression fixtures cover detection-column exclusion, sample reordering,
missing/invalid values, malformed matrix rejection, gene expansion, intersection,
effect sign, inference ID alignment, and a complete synthetic comparison with
plots. Real output validation checks every annotated pair, delta consistency,
sample counts, source checksums and independently recomputes three CpGs from
both original downloaded matrices.

These are descriptive cross-study comparisons. Disease and study are confounded;
tumor samples are FFPE. Batch, age, race, tumor purity and tissue cell composition
are unadjusted. External effects are an independent descriptive reference, not
patient classification or proof that predicted nodes are validated biomarkers.

Sources: [GSE226569](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226569),
[GSE101961](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE101961), and
[GPL13534 annotation](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/soft/GPL13534_family.soft.gz).

## Original-source audit: PBMC versus breast tissue

GSE237036, the original training cohort, measures **PBMCs** (GPL23976), whereas
GSE226569 and GSE101961 measure **breast tissue** (GPL13534). The effect comparison
is exploratory across different tissues and must not be presented as matched-tissue
validation. All 350 plotted CpGs were independently recomputed from original GEO
files; numerical errors were below 5e-13. See
[the source audit](results/original_gse_audit/REPORT.md), its per-sample observations
and the source beta distribution plots. The comparison figures now label the
tissues explicitly; the previous figures are preserved in the audit folder.

```bash
python evaluation/audit_original_gse.py
python evaluation/render_gse_audit.py
```
