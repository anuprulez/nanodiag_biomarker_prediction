from pathlib import Path
import json,csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

root=Path(__file__).resolve().parents[1];audit=root/'evaluation/results/original_gse_audit';base=root/'evaluation/results';comparison=base/'prediction_comparison';suffix='PNA_corr_0.33_knn_n_3000_n_neg_10000'
report=json.loads((audit/'audit_report.json').read_text())
values=pd.read_csv(audit/'plotted_cpg_source_beta_values.tsv.gz',sep='\t')
original=values.loc[values.series.eq('GSE237036')].pivot(index='CpG_ID',columns='source_column',values='beta')
remaining=set(original.index);max_error=0.
with (root/'raw/GSE237036_matrix_processed.txt').open() as handle:
 header=next(handle).strip().split('\t')[1:]
 for line in handle:
  probe=line.split('\t',1)[0]
  if probe not in remaining:continue
  nums=np.array(line.rstrip().split('\t')[1:],dtype=float)
  max_error=max(max_error,float(np.max(np.abs(nums-original.loc[probe,header].to_numpy()))))
  remaining.remove(probe)
assert not remaining and max_error<1e-14
report['maximum_per_sample_error_GEO_vs_local_training_matrix']=max_error
report['training_sample_betas_checked']=len(original)*len(header)
# Independently verify all matched CpG-gene pairs against the platform annotation.
ann={}
with (root/'evaluation/raw/GPL13534_probe_annotation.tsv').open() as handle:
 for row in csv.DictReader(handle,delimiter='\t'):
  ann[row['ID']]=set(g.strip() for g in row['UCSC_RefGene_Name'].split(';') if g.strip())
coverage=[];gene_rows=0
controls=set(pd.read_csv(base/'GSE101961_cpg_gene_statistics.tsv',sep='\t',usecols=['CpG_ID']).CpG_ID)
tumors=set(pd.read_csv(base/'GSE226569_cpg_gene_statistics.tsv',sep='\t',usecols=['CpG_ID']).CpG_ID)
for group in ['negative','likely_positive']:
 matched=pd.read_csv(comparison/f'{group}_common_cpg_gene_matches_{suffix}.tsv',sep='\t')
 for _,row in matched.iterrows():assert row.Predicted_gene in ann[row.CpG_ID]
 gene_rows+=len(matched)
 all_pred=pd.read_csv(comparison/f'{group}_all_predictions_{suffix}.tsv',sep='\t');probes=set(all_pred.CpG_ID)
 coverage.append({'prediction_group':group,'predicted_unique_cpgs':len(probes),'plotted_common_cpgs':len(probes&controls&tumors),
                  'coverage_fraction':len(probes&controls&tumors)/len(probes),'absent_from_control_beta_matrix':len(probes-controls),
                  'present_in_control_but_absent_from_tumor_beta_matrix':len((probes&controls)-tumors)})
report['matched_gene_rows_independently_verified']=gene_rows;report['coverage']=coverage
(audit/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n')
# Source distributions for examples exhibiting sign reversals.
examples=['cg12018098','cg02846790']
labels=['PBMC control','PBMC BC patient','Normal breast','Breast tumor']
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
checks=pd.read_csv(audit/'all_plotted_cpg_recalculation.tsv',sep='\t').set_index('CpG_ID')
for ax,probe in zip(axes,examples):
 frame=values.loc[values.CpG_ID.eq(probe)].copy()
 frame['cohort']=np.select([frame.series.eq('GSE237036')&frame.group.eq('healthy_control'),frame.series.eq('GSE237036')&frame.group.eq('breast_cancer'),frame.series.eq('GSE101961')],labels[:3],default=labels[3])
 sns.boxplot(data=frame,x='cohort',y='beta',order=labels,showfliers=False,ax=ax,color='#a9c9e5')
 sns.stripplot(data=frame,x='cohort',y='beta',order=labels,ax=ax,size=1.8,alpha=.3,color='#333333',jitter=.2,rasterized=True)
 means=frame.groupby('cohort').beta.mean().reindex(labels)
 ax.scatter(range(4),means,marker='D',color='red',s=35,zorder=5)
 row=checks.loc[probe]
 ax.set_title(f'{probe}\nPBMC Δβ={row.Source_PBMC_delta:+.3f}; breast tissue Δβ={row.Source_breast_tissue_delta:+.3f}')
 ax.set_xticks(range(4),['PBMC control\nn=30','PBMC BC\nn=50','Normal breast\nn=121','Breast tumor\nn=694']);ax.set(xlabel='',ylabel='Published source beta',ylim=(-.02,1.02))
fig.suptitle('Source GSE beta values: effect directions differ between blood and breast tissue')
fig.tight_layout();fig.savefig(audit/'source_beta_distributions.pdf',bbox_inches='tight');fig.savefig(audit/'source_beta_distributions.png',dpi=200,bbox_inches='tight');plt.close(fig)
lines=['# Audit of original GSE files and the PNA effect comparison','',
       '**Finding: the plotted numbers are correct; the cohorts measure different tissues.**','',
       'The original graph inputs are PBMC methylation from GSE237036. The external cohorts measure breast tumor and normal breast tissue. The original plot did not make this tissue mismatch sufficiently clear.','',
       '| Dataset | Role | Tissue | Platform | Samples |','| --- | --- | --- | --- | ---: |',
       '| GSE237036 | Original case/control | PBMCs | GPL23976, EPIC 850K | 50 BC + 30 controls |',
       '| GSE226569 | External cases | FFPE breast tumor | GPL13534, 450K | 694 |',
       '| GSE101961 | External controls | Normal breast from cancer-free women | GPL13534, 450K | 121 |','',
       '## Independent numerical audit','',
       '- Downloaded the original GSE237036 processed matrix directly from NCBI, independently of the Zenodo training package.',
       '- Recomputed all 350 plotted CpGs from the original three GEO matrices using an independent csv/numpy reader, rather than the evaluation summary functions.',
       '- Matched all 80 PBMC sample column names to GEO sample descriptions and source labels. All patient/control assignments agree; mixed-case NC names are handled.',
       f'- Checked {report["training_sample_betas_checked"]:,} PBMC beta values individually against the local training matrix: maximum discrepancy {max_error:.3g}.',
       '- All six plotted mean/delta quantities agree with source recomputation to within 5e-13 (TSV rounding). Both deltas are consistently case minus control.',
       f'- Independently verified all {gene_rows} matched feature rows against GPL13534 gene annotation; these collapse to 350 unique CpGs.',
       '- Tumor beta columns were separated from detection-p-value columns and matched to GEO titles. None of the 350 plotted CpGs had a tumor sample with detection p >0.01.',
       '- No changes to beta values, group membership or delta signs were required. The updated plots make the tissue mismatch explicit.','',
       'Detailed numeric checks: [all_plotted_cpg_recalculation.tsv](all_plotted_cpg_recalculation.tsv).',
       'Individual source observations: [plotted_cpg_source_beta_values.tsv.gz](plotted_cpg_source_beta_values.tsv.gz).',
       'Sample identity audit: [sample_identity_audit.tsv](sample_identity_audit.tsv).',
       'Machine-readable results and checksums: [audit_report.json](audit_report.json).','',
       '## What the weak results mean','',
       '| Prediction group | Plotted / predicted unique CpGs | Coverage | Pearson r | Same effect direction |','| --- | ---: | ---: | ---: | ---: |']
for cov in coverage:
 group=next(x for x in report['groups'] if x['prediction_group']==cov['prediction_group'])
 lines.append(f'| {cov["prediction_group"]} | {cov["plotted_common_cpgs"]} / {cov["predicted_unique_cpgs"]} | {cov["coverage_fraction"]:.1%} | {group["pearson_r"]:.3f} | {group["same_direction_fraction"]:.1%} |')
lines+=['',
        'The likely-positive effects do not reproduce consistently in these breast tissue cohorts. This observation is real for the available common CpGs. Tissue mismatch, separate case/control studies, processing differences and tumor composition prevent attributing the discrepancy to a single cause.','',
        'Only about a quarter of the predicted CpGs appear in the comparison. Of the negative set, 396 are absent from the control beta matrix and another 236 appear in controls but not in the processed tumor matrix. For likely positives the corresponding counts are 303 and 73. The 850K/450K platform difference and the tumor matrix\'s reduced 276,046-CpG coverage limit which predictions can be checked. Missing sites are not replication failures; the plotted subset is not representative of all predictions.','',
        '## Source examples','',
        '| CpG | PBMC control mean | PBMC BC mean | PBMC Δβ | Normal breast mean | Tumor mean | Breast tissue Δβ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for probe in examples:
 row=checks.loc[probe]
 lines.append(f'| {probe} | {row.Source_PBMC_healthy_mean:.6f} | {row.Source_PBMC_BC_mean:.6f} | {row.Source_PBMC_delta:+.6f} | {row.Source_normal_breast_mean:.6f} | {row.Source_breast_tumor_mean:.6f} | {row.Source_breast_tissue_delta:+.6f} |')
lines+=['','[Individual-source beta distribution plots](source_beta_distributions.pdf). Red diamonds indicate means.','',
        '## Model interpretation','',
        'PNA\'s 86.17% internal test accuracy measures agreement with propagated node-priority labels. It does not measure patient diagnostic accuracy, effect-size prediction or external biomarker replication. The label named "negative" is a propagated priority class; it does not guarantee an externally negligible delta beta. No improved model performance claim should be interpreted as successful biological validation.','',
        'For validation of the PBMC signature, use an independent BC-versus-healthy PBMC cohort with compatible processing and prespecified evaluation criteria. Keep the requested breast cohorts as an exploratory cross-tissue analysis. Do not flip signs, relabel samples, filter away discordant sites or select new model settings based on this external result.','',
        '## Published processing differences','',
        '- GSE237036: Average Beta data from BeadStudio v3.2.',
        '- GSE226569: GenomeStudio intensities processed with minfi; Average Beta supplementary matrix, interleaved with detection p-values.',
        '- GSE101961: GenomeStudio/Partek with SWAN normalization, according to GEO metadata.',
        'No batch or cell-composition correction was applied. Disease is confounded with study in the external case/control comparison; automatic study correction cannot safely isolate disease from study effects.','',
        '## Reproduce','',
        '```bash','python evaluation/audit_original_gse.py','python evaluation/render_gse_audit.py','```','',
        'Requires numpy/pandas and the three original matrices under evaluation/raw/. The GSE237036 metadata and beta matrix were downloaded for this audit; the other sources are retained from evaluation setup.','',
        '## Primary sources','',
        '[GSE237036 metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE237nnn/GSE237036/matrix/GSE237036_series_matrix.txt.gz),',
        '[GSE237036 original beta matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE237nnn/GSE237036/suppl/GSE237036_matrix_processed.txt.gz),',
        '[GSE226569 metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/GSE226569/matrix/GSE226569_series_matrix.txt.gz),',
        '[GSE226569 beta matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/GSE226569/suppl/GSE226569_Methy_ProcessedMatrix.csv.gz),',
        '[GSE101961 metadata and beta matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE101nnn/GSE101961/matrix/GSE101961_series_matrix.txt.gz).','']
(audit/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({'sample_values_checked':report['training_sample_betas_checked'],'gene_rows_checked':gene_rows,'report':str(audit/'REPORT.md')},indent=2))
