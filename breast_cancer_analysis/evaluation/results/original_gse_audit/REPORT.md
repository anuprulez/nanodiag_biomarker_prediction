# Audit of original GSE files and the PNA effect comparison

**Finding: the plotted numbers are correct; the cohorts measure different tissues.**

The original graph inputs are PBMC methylation from GSE237036. The external cohorts measure breast tumor and normal breast tissue. The original plot did not make this tissue mismatch sufficiently clear.

| Dataset | Role | Tissue | Platform | Samples |
| --- | --- | --- | --- | ---: |
| GSE237036 | Original case/control | PBMCs | GPL23976, EPIC 850K | 50 BC + 30 controls |
| GSE226569 | External cases | FFPE breast tumor | GPL13534, 450K | 694 |
| GSE101961 | External controls | Normal breast from cancer-free women | GPL13534, 450K | 121 |

## Independent numerical audit

- Downloaded the original GSE237036 processed matrix directly from NCBI, independently of the Zenodo training package.
- Recomputed all 350 plotted CpGs from the original three GEO matrices using an independent csv/numpy reader, rather than the evaluation summary functions.
- Matched all 80 PBMC sample column names to GEO sample descriptions and source labels. All patient/control assignments agree; mixed-case NC names are handled.
- Checked 28,000 PBMC beta values individually against the local training matrix: maximum discrepancy 1.11e-16.
- All six plotted mean/delta quantities agree with source recomputation to within 5e-13 (TSV rounding). Both deltas are consistently case minus control.
- Independently verified all 369 matched feature rows against GPL13534 gene annotation; these collapse to 350 unique CpGs.
- Tumor beta columns were separated from detection-p-value columns and matched to GEO titles. None of the 350 plotted CpGs had a tumor sample with detection p >0.01.
- No changes to beta values, group membership or delta signs were required. The updated plots make the tissue mismatch explicit.

Detailed numeric checks: [all_plotted_cpg_recalculation.tsv](all_plotted_cpg_recalculation.tsv).
Individual source observations: [plotted_cpg_source_beta_values.tsv.gz](plotted_cpg_source_beta_values.tsv.gz).
Sample identity audit: [sample_identity_audit.tsv](sample_identity_audit.tsv).
Machine-readable results and checksums: [audit_report.json](audit_report.json).

## What the weak results mean

| Prediction group | Plotted / predicted unique CpGs | Coverage | Pearson r | Same effect direction |
| --- | ---: | ---: | ---: | ---: |
| negative | 227 / 859 | 26.4% | -0.363 | 27.8% |
| likely_positive | 123 / 499 | 24.6% | 0.070 | 46.3% |

The likely-positive effects do not reproduce consistently in these breast tissue cohorts. This observation is real for the available common CpGs. Tissue mismatch, separate case/control studies, processing differences and tumor composition prevent attributing the discrepancy to a single cause.

Only about a quarter of the predicted CpGs appear in the comparison. Of the negative set, 396 are absent from the control beta matrix and another 236 appear in controls but not in the processed tumor matrix. For likely positives the corresponding counts are 303 and 73. The 850K/450K platform difference and the tumor matrix's reduced 276,046-CpG coverage limit which predictions can be checked. Missing sites are not replication failures; the plotted subset is not representative of all predictions.

## Source examples

| CpG | PBMC control mean | PBMC BC mean | PBMC Δβ | Normal breast mean | Tumor mean | Breast tissue Δβ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cg12018098 | 0.272905 | 0.236543 | -0.036362 | 0.194010 | 0.536609 | +0.342599 |
| cg02846790 | 0.885233 | 0.905883 | +0.020649 | 0.918369 | 0.538706 | -0.379664 |

[Individual-source beta distribution plots](source_beta_distributions.pdf). Red diamonds indicate means.

## Model interpretation

PNA's 86.17% internal test accuracy measures agreement with propagated node-priority labels. It does not measure patient diagnostic accuracy, effect-size prediction or external biomarker replication. The label named "negative" is a propagated priority class; it does not guarantee an externally negligible delta beta. No improved model performance claim should be interpreted as successful biological validation.

For validation of the PBMC signature, use an independent BC-versus-healthy PBMC cohort with compatible processing and prespecified evaluation criteria. Keep the requested breast cohorts as an exploratory cross-tissue analysis. Do not flip signs, relabel samples, filter away discordant sites or select new model settings based on this external result.

## Published processing differences

- GSE237036: Average Beta data from BeadStudio v3.2.
- GSE226569: GenomeStudio intensities processed with minfi; Average Beta supplementary matrix, interleaved with detection p-values.
- GSE101961: GenomeStudio/Partek with SWAN normalization, according to GEO metadata.
No batch or cell-composition correction was applied. Disease is confounded with study in the external case/control comparison; automatic study correction cannot safely isolate disease from study effects.

## Reproduce

```bash
python evaluation/audit_original_gse.py
python evaluation/render_gse_audit.py
```

Requires numpy/pandas and the three original matrices under evaluation/raw/. The GSE237036 metadata and beta matrix were downloaded for this audit; the other sources are retained from evaluation setup.

## Primary sources

[GSE237036 metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE237nnn/GSE237036/matrix/GSE237036_series_matrix.txt.gz),
[GSE237036 original beta matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE237nnn/GSE237036/suppl/GSE237036_matrix_processed.txt.gz),
[GSE226569 metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/GSE226569/matrix/GSE226569_series_matrix.txt.gz),
[GSE226569 beta matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/GSE226569/suppl/GSE226569_Methy_ProcessedMatrix.csv.gz),
[GSE101961 metadata and beta matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE101nnn/GSE101961/matrix/GSE101961_series_matrix.txt.gz).
