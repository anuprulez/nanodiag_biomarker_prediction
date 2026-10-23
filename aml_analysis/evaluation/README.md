# GSE40279 versus GSE62298

Healthy: 656 samples. AML: 68 samples.
Shared CpGs: 470,043; CpG–gene rows: 393,997.

Full table: `results/cpg_gene_beta_comparison.tsv`. All shared CpGs, including unannotated probes: `results/shared_cpg_statistics.tsv`.

The preview below contains the first 10 CpG–gene pairs in CpG ID order, not a significance ranking.

| CpG_gene | Healthy GSE40279 (mean ± SD) | AML GSE62298 (mean ± SD) | Δβ (AML − healthy) |
|---|---:|---:|---:|
| cg00000029_RBL2 | 0.476097 ± 0.068449 | 0.552630 ± 0.220065 | +0.076533 |
| cg00000108_C3orf35 | 0.944313 ± 0.020977 | 0.936439 ± 0.012153 | -0.007874 |
| cg00000109_FNDC3B | 0.892913 ± 0.033777 | 0.901500 ± 0.024141 | +0.008587 |
| cg00000236_VDAC3 | 0.746412 ± 0.041170 | 0.737685 ± 0.152562 | -0.008728 |
| cg00000289_ACTN1 | 0.609753 ± 0.065780 | 0.708363 ± 0.081368 | +0.098610 |
| cg00000292_ATP2A1 | 0.826354 ± 0.024053 | 0.769176 ± 0.138833 | -0.057178 |
| cg00000321_SFRP1 | 0.312680 ± 0.067525 | 0.388763 ± 0.165339 | +0.076083 |
| cg00000622_NIPA2 | 0.007836 ± 0.005106 | 0.009879 ± 0.004786 | +0.002043 |
| cg00000658_MAN1B1 | 0.817863 ± 0.023315 | 0.817420 ± 0.028870 | -0.000442 |
| cg00000714_TSEN34 | 0.188988 ± 0.025302 | 0.278197 ± 0.141297 | +0.089209 |

From aml_analysis, run `python evaluation/download_datasets.py` to download/validate the matrices, then `python evaluation/evaluate_datasets.py` to regenerate the results. Run `python evaluation/validate_results.py` to validate output identities, counts, delta values, and independently recompute three CpGs from both source matrices.

Method: exact intersection of cg probe IDs; GPL13534 UCSC_RefGene_Name (legacy hg19) annotation reused from ../raw. Multiple gene assignments are expanded into unique CpG–gene pairs. Published beta values are unchanged. SD uses n−1; missing values are omitted per probe, with valid n reported. No imputation, filtering by effect, or statistical significance threshold was applied.

These are descriptive cross-study differences without batch, age, sex, or cell-composition adjustment. Whole blood controls and primary AML blood differ in cellular composition. GSE62298 is the requested AML GEO dataset; it is not TCGA.

Dataset URLs, checksums, row counts and policies are in results/analysis_report.json; sample metadata and cohort tables are retained in raw/ and results/.
