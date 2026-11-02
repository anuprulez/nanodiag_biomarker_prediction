"""Audit AML panel direction without changing labels, models or candidate selection."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AML = ROOT / 'aml_analysis'
BASE = AML / 'data_corr_0.33_knn_n_3000_n_neg_10000'
OUT = BASE / 'accumulated_results/post_evaluation/methylation_expression/hypermethylation_audit'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(AML / 'raw/harmonised/sample_metadata.tsv', sep='\t')
    features = pd.read_csv(BASE / 'runs/run1/generated/data/df_nebit_dnam_features_aml_PNA.csv')
    beta = features[[str(i) for i in range(len(meta))]].to_numpy(float)
    disease = meta.label.to_numpy() == 1
    features['original_delta'] = beta[:, disease].mean(axis=1) - beta[:, ~disease].mean(axis=1)
    features['CpG_ID'] = features.name.str.split('_').str[0]
    features['direction'] = np.where(features.original_delta > 0, 'hyper', np.where(features.original_delta < 0, 'hypo', 'zero'))
    panel = pd.read_csv(BASE / 'accumulated_results/post_evaluation/data/fixed_candidate_panel.tsv', sep='\t')
    checked = panel.merge(features[['CpG_ID','original_delta']].drop_duplicates('CpG_ID'), on='CpG_ID', validate='one_to_one')
    assert len(checked) == 17
    assert np.allclose(checked.original_delta, checked.Internal_delta_beta_AML_minus_healthy, atol=1e-7)
    features[['name','CpG_ID','labels','original_delta','direction']].to_csv(OUT / 'graph_node_directions.tsv', sep='\t', index=False)
    features.groupby(['labels','direction']).size().unstack(fill_value=0).to_csv(OUT / 'niapu_class_directions.tsv', sep='\t')
    seeds = pd.read_csv(AML / 'raw/aml_top200_cpg_gene_delta_beta.tsv', sep='\t').drop_duplicates('CpG_ID')
    common = pd.read_csv(BASE / 'accumulated_results/evaluation/likely_positive_common_unique_cpgs_PNA_corr_0.33_knn_n_3000_n_neg_10000.tsv', sep='\t')
    rows = []
    def add(stage, delta, unit):
        d = np.asarray(delta)
        rows.append(dict(stage=stage, unit=unit, hyper=int((d>0).sum()), hypo=int((d<0).sum()), zero=int((d==0).sum())))
    add('Annotated seeds', seeds.deltaBeta_AML_minus_Normal, 'unique CpGs')
    add('Merged graph', features.drop_duplicates('CpG_ID').original_delta, 'unique CpGs')
    add('NIAPU likely-positive class 2', features.loc[features.labels.eq(2)].original_delta, 'CpG-gene nodes')
    add('Common PNA likely-positive candidates', common.Internal_delta_beta_AML_minus_healthy, 'unique CpGs')
    add('Frozen post-evaluation panel', panel.Internal_delta_beta_AML_minus_healthy, 'unique CpGs')
    pd.DataFrame(rows).to_csv(OUT / 'selection_direction_counts.tsv', sep='\t', index=False)
    for cutoff in [0,.1,.2,.25,.3]:
        selected = common.loc[common.Delta_beta_AML_minus_healthy.abs().ge(cutoff)]
        print(f'External absolute delta >= {cutoff}: {len(selected)}')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Internal delta range:', panel.Internal_delta_beta_AML_minus_healthy.min(), panel.Internal_delta_beta_AML_minus_healthy.max())
    print('External delta range:', panel.Delta_beta_AML_minus_healthy.min(), panel.Delta_beta_AML_minus_healthy.max())
    sensitivity = pd.read_csv(AML / 'evaluation/dataset_comparability/frozen_17_same_study_control_sensitivity.tsv', sep='\t')
    print('Same-study positive:', sensitivity.within_study_delta_beta.gt(0).sum(), 'FDR < .05:', sensitivity.targeted_BH_FDR_17.lt(.05).sum())

if __name__ == '__main__':
    main()
