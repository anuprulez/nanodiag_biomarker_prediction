#!/usr/bin/env python3
"""Freeze the validation-selected PNA, evaluate test once, and optionally install settings."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.loader import NeighborLoader

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from preprocessing_grid import preprocessing_configs
from seed_inputs import validate_seed_manifest
from utils import choose_model
from train_model_sub_graph import val_evaluate, test_evaluate, validate_predefined_splits
from tune_models import seed_all

TRAIN_KEYS=('SEED','batch_size','hidden_dim','grad_clip_norm','learning_rate','weight_decay','dropout',
            'gnn_layers','class_weight_power','feature_dropout','embedding_dropout','neighbors_spread',
            'n_epo','lr_patience','early_stopping_patience','checkpoint_selection_metric','pna_variant')


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True,help='Directory containing PNA best_config.yaml and best_model.pt')
    parser.add_argument('--config',type=Path,default=HERE.parent/'config/config.yaml')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args();study=args.study.resolve();config_path=args.config.resolve()
    os.chdir(HERE)
    source=OmegaConf.load(config_path)
    best=OmegaConf.load(study/'best_config.yaml')
    winner=json.loads((study/'best_params.json').read_text())
    baseline_root=study.parents[2]
    if baseline_root.name!='generated':raise ValueError('Expected generated/optuna/<tag>/PNA study layout')
    output=args.output.resolve() if args.output else baseline_root/'pna_accuracy_improved'
    if output.exists() and any(output.iterdir()):raise ValueError('Output exists; do not repeat test evaluation or overwrite results')
    validate_seed_manifest(best)
    if best.model_type!='PNA' or winner['validation_metric']!='accuracy':raise ValueError('Expected accuracy-selected PNA')
    original=json.loads((baseline_root/'all_metrics_PNA.json').read_text())
    if winner['validation_score']<=original['best_val_acc']:raise ValueError('No validation-accuracy improvement; keep current model')
    graph=study/'prepared/data_bc_PNA.pt'
    data=torch.load(graph,map_location='cpu',weights_only=False)
    baseline=torch.load(baseline_root/'data_bc_PNA.pt',map_location='cpu',weights_only=False)
    for field in ('x','y','edge_index','train_mask','val_mask','test_mask'):
        if not torch.equal(getattr(data,field),getattr(baseline,field)):
            raise ValueError(f'Search and original model data differ: {field}')
    del baseline
    validate_predefined_splits(data,pd.read_csv(best.p_out_genes,sep=r'\s+',header=None),best)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(int(best.SEED))
    model=choose_model(best,data,'PNA',False).to(device)
    model.load_state_dict(torch.load(study/'best_model.pt',map_location=device,weights_only=True))
    counts=torch.bincount(data.y[data.train_mask],minlength=best.num_classes).float()
    weights=(counts.sum()/(len(counts)*counts)).pow(float(best.get('class_weight_power',1.)))
    weights/=weights.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=weights.to(device))
    def loader(split):
        return NeighborLoader(data,input_nodes=getattr(data,split+'_mask'),num_neighbors=list(best.neighbors_spread),
                              batch_size=int(best.batch_size),shuffle=False,num_workers=int(best.num_workers),subgraph_type='bidirectional')
    # Reproduce the frozen validation result before accessing the test partition.
    seed_all(int(best.SEED))
    val_loss,val_acc,_,val_f1=val_evaluate(loader('val'),model,criterion,device)
    if not np.isclose(val_acc,winner['validation_score'],rtol=0,atol=1e-10):
        raise ValueError(f'Frozen checkpoint validation differs: {val_acc} vs {winner["validation_score"]}')
    output.mkdir(parents=True,exist_ok=True)
    selection={'model':'PNA','trial':winner['trial'],'validation_accuracy':float(val_acc),
               'validation_macro_f1':float(val_f1),'checkpoint_sha256':digest(study/'best_model.pt'),
               'selection':'Highest validation accuracy across the fixed search; test not used to choose configuration',
               'source_study':str(study),'graph_sha256':digest(graph)}
    (output/'selection_frozen_before_test.json').write_text(json.dumps(selection,indent=2)+'\n')
    seed_all(int(best.SEED))
    loss,acc,pred,truth,probs,confidence,ids,embed,embed_bn=test_evaluate(loader('test'),model,criterion,device)
    ids=np.asarray(ids,dtype=int)
    if len(set(ids))!=len(ids) or set(ids)!=set(data.test_mask.nonzero().view(-1).tolist()):raise ValueError('Test IDs do not match held-out partition')
    if not np.array_equal(np.asarray(truth),data.y[ids].numpy()):raise ValueError('Test labels not aligned')
    assert np.isclose(acc,accuracy_score(truth,pred))
    history=json.loads((study/f'trial_{winner["trial"]:05d}/validation_history.json').read_text())
    chosen=max(history,key=lambda item:item['accuracy'])
    metrics={'best_epoch':chosen['epoch'],'best_val_acc':float(val_acc),'best_val_macro_f1':float(val_f1),
             'completed_epochs':len(history),'checkpoint_selection_metric':'accuracy',
             'te_acc':[float(acc)],'te_loss':[float(loss)],
             'te_f1_macro':float(f1_score(truth,pred,average='macro')),
             'te_f1_weighted':float(f1_score(truth,pred,average='weighted')),
             'te_f1_micro':float(f1_score(truth,pred,average='micro')),
             'te_precision':float(precision_score(truth,pred,average='weighted',zero_division=0)),
             'te_recall':float(recall_score(truth,pred,average='weighted',zero_division=0)),
             'val_acc':[item['accuracy'] for item in history],
             'val_macro_f1':[item['macro_f1'] for item in history],
             'selection':selection}
    for prefix,values in [('pred_labels',pred),('true_labels',truth),('all_class_pred_probs',probs),
                           ('best_class_pred_probs',confidence),('test_ids',ids.tolist()),
                           ('embed_conv',embed),('embed_batch_norm',embed_bn)]:
        torch.save(values,output/f'{prefix}_bc_PNA.pt')
    for split in ('train','val','test'):
        shutil.copy2(study/f'prepared/{split}_probe_genes_bc_PNA.csv',output/f'{split}_probe_genes_bc_PNA.csv')
    shutil.copy2(graph,output/'data_bc_PNA.pt')
    (output/'models').mkdir(exist_ok=True)
    shutil.copy2(study/'best_model.pt',output/'models/best_model_bc_PNA.pt')
    (output/'all_metrics_PNA.json').write_text(json.dumps(metrics,indent=2)+'\n')
    baseline_table=pd.read_csv(baseline_root/'Heatmap_avg_edges_9913451_epochs_100.csv')
    new_row={column:metrics[column] if column not in ('model','te_acc') else ('PNA_improved' if column=='model' else acc)
             for column in baseline_table.columns}
    comparison=pd.concat([baseline_table,pd.DataFrame([new_row])],ignore_index=True)
    comparison.to_csv(output/'model_comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig,ax=plt.subplots(figsize=(13,7))
    sns.heatmap(comparison.set_index('model').astype(float),annot=True,fmt='.4f',cmap='YlGnBu',vmin=0,vmax=1,ax=ax)
    ax.set_title('Original models and validation-selected PNA (held-out test evaluation)')
    fig.tight_layout();fig.savefig(output/'Heatmap_PNA_improved.pdf');fig.savefig(output/'Heatmap_PNA_improved.png',dpi=220);plt.close(fig)
    settings={key:OmegaConf.to_container(best,resolve=False)[key] for key in TRAIN_KEYS if key in best}
    settings['p_generated_files']='${p_base}generated/pna_accuracy_improved/'
    # Save a self-contained runnable configuration while preserving original model artifacts.
    final_config=copy.deepcopy(best)
    for key in list(final_config):
        if key.startswith('p_') and isinstance(final_config[key],str):
            value=final_config[key]
            old=str(study/'final')+'/'
            if value.startswith(old):final_config[key]=str(output)+'/'+value[len(old):]
    OmegaConf.save(final_config,output/'selected_config.yaml')
    if args.install:
        shutil.copy2(config_path,output/'config_before_pna_improvement.yaml')
        source.model_overrides.PNA=OmegaConf.create(settings)
        text=config_path.read_text();start=text.index('model_overrides:');end=text.index('\nsplit_ratios:',start)
        block=OmegaConf.to_yaml(OmegaConf.create({'model_overrides':OmegaConf.to_container(source.model_overrides,resolve=False)})).rstrip()
        config_path.write_text(text[:start]+block+text[end:])
        from model_settings import apply_model_settings
        current=apply_model_settings(OmegaConf.load(config_path),'PNA')
        for key,value in settings.items():
            if key!='p_generated_files':assert OmegaConf.to_container(current,resolve=False)[key]==value
    best_other=baseline_table.loc[baseline_table.model.isin(['GCN','GraphSAGE','GATv2','GraphTransformer']),'te_acc'].max()
    report=dict(selection,original_PNA_test_accuracy=original['te_acc'][0],improved_PNA_test_accuracy=float(acc),
                best_other_GNN_test_accuracy=float(best_other),outperforms_other_GNNs=bool(acc>best_other),
                original_PNA_validation_accuracy=original['best_val_acc'],improved_PNA_validation_accuracy=float(val_acc),
                test_macro_f1=metrics['te_f1_macro'],config_installed=args.install,
                limitations='Single seed on propagated CpG-node targets; not patient diagnosis or independent biomarker validation. The baseline test set was previously inspected; it was not used by this search.')
    (output/'improvement_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Breast cancer PNA accuracy search','',
           f'Selected by validation accuracy only: trial {winner["trial"]}, epoch {chosen["epoch"]}.',
           f'Validation accuracy: {original["best_val_acc"]:.4%} → {val_acc:.4%}.',
           f'Test accuracy: original PNA {original["te_acc"][0]:.4%}; selected PNA {acc:.4%}; best other GNN {best_other:.4%}.',
           f'Outperforms the other plotted GNNs: {acc>best_other}. Test macro F1: {metrics["te_f1_macro"]:.6f}.','',
           'The selected checkpoint was frozen before its single test evaluation. No test-based candidate selection was performed.',
           'Original artifacts are preserved. This folder contains the selected model, graph, predictions, split tables, metrics, runnable config and comparison heatmap.',
           'See the adjacent Optuna study for trial histories and best parameters.','',report['limitations']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
