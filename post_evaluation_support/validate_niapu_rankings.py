"""Numerical and source-isolation checks for the NIAPU patient comparison."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix


def main(out):
    out=Path(out);data=out/'data';audit=json.loads((out/'selection_manifest.json').read_text())
    nodes=pd.read_csv(data/'all_niapu_nodes.tsv',sep='\t')
    lp=pd.read_csv(data/'niapu_likely_positive_nodes.tsv',sep='\t')
    assert lp.niapu_class.eq(2).all()
    assert set(lp.node_name)==set(nodes.loc[nodes.niapu_class.eq(2),'node_name'])
    selected=pd.read_csv(data/'selected_niapu_CpGs.tsv',sep='\t');features=selected.CpG_ID.tolist()
    assert selected.CpG_ID.is_unique and set(features)<=set(lp.CpG_ID)
    assert not set(features)&set(nodes.loc[nodes.niapu_class.eq(1),'CpG_ID'])
    all_sites=pd.read_csv(data/'all_common_likely_positive_CpGs.tsv',sep='\t')
    internal=f'Internal_delta_beta_{audit["disease"]}_minus_healthy';external=f'Delta_beta_{audit["disease"]}_minus_healthy'
    expected=(all_sites[external].abs().ge(.30)&all_sites.External_p_value.lt(.01)&all_sites.External_FDR.lt(.01)&all_sites[internal].mul(all_sites[external]).gt(0))
    assert set(features)==set(all_sites.loc[expected,'CpG_ID'])
    universe=pd.read_csv(data/'external_full_test_universe.tsv.gz',sep='\t')
    expected_q=false_discovery_control(universe.External_p_value.fillna(1).to_numpy(),method='bh')
    valid=universe.External_p_value.notna()
    np.testing.assert_allclose(universe.loc[valid,'External_FDR'],expected_q[valid],atol=1e-12)
    original=pd.read_csv(data/'original_selected_beta.tsv.gz',sep='\t',index_col=0)
    ext=pd.read_csv(data/'external_selected_beta.tsv.gz',sep='\t',index_col=0)
    om=pd.read_csv(data/'original_patient_annotations.tsv',sep='\t',index_col=0).loc[original.index]
    em=pd.read_csv(data/'external_patient_annotations.tsv',sep='\t',index_col=0).loc[ext.index]
    assert not set(original.index)&set(ext.index)
    for matrix,meta,column in [(original,om,internal),(ext,em,external)]:
        delta=matrix.loc[meta.label.eq(1)].mean()-matrix.loc[meta.label.eq(0)].mean()
        np.testing.assert_allclose(selected[column],selected.CpG_ID.map(delta),atol=2e-9,rtol=0)
    predictions=pd.read_csv(data/'classifier_patient_predictions.tsv',sep='\t')
    metrics=pd.read_csv(data/'classifier_performance.tsv',sep='\t')
    models={'Random Forest','XGBoost','TabPFN3','TabICLv2'}
    assert set(predictions.model)==models and len(metrics)==8
    assert not predictions.duplicated(['sample_id','model','cohort']).any()
    for (model,cohort),p in predictions.groupby(['model','cohort']):
        meta=om if cohort=='original_patient_CV' else em
        assert set(p.sample_id)==set(meta.index)
        np.testing.assert_array_equal(p.label,p.sample_id.map(meta.label))
        assert set(p.fold)==({1,2,3,4,5} if cohort=='original_patient_CV' else {0})
        assert np.isfinite(p.probability).all() and p.probability.between(0,1).all()
        row=metrics.loc[metrics.model.eq(model)&metrics.cohort.eq(cohort)].iloc[0]
        predicted=p.probability.ge(.5).astype(int)
        np.testing.assert_allclose(row.accuracy_at_0_5,accuracy_score(p.label,predicted),atol=1e-12)
        np.testing.assert_allclose(row.balanced_accuracy_at_0_5,balanced_accuracy_score(p.label,predicted),atol=1e-12)
        np.testing.assert_array_equal([row.TN,row.FP,row.FN,row.TP],confusion_matrix(p.label,predicted,labels=[0,1]).ravel())
    result=dict(status='passed',selected_CpGs=len(features),likely_positive_unique_CpGs=lp.CpG_ID.nunique(),models=sorted(models),
                checks=['NIAPU class 2 only; seeds excluded','All and only CpGs meeting the exact fixed criterion',
                        'BH FDR independently verified over full external universe','Original/external sample contrasts independently reproduced',
                        'No original/external GSM overlap','Unique five-fold OOF and external predictions per model',
                        'Accuracy, balanced accuracy and confusion counts independently recomputed'])
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
