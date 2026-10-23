#!/usr/bin/env python3
"""Export AML-equivalent prediction groups aligned by saved inference node IDs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def extract(data,model,output,generated_dir=None):
    import torch
    generated=Path(generated_dir) if generated_dir is not None else Path(data)/'generated'; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    def load(prefix):
        path=generated/f'{prefix}_bc_{model}.pt'
        values=torch.load(path,map_location='cpu',weights_only=False)
        return np.asarray(values.detach().cpu() if hasattr(values,'detach') else values).reshape(-1)
    ids=load('test_ids'); truth=load('true_labels'); predicted=load('pred_labels'); probabilities=load('best_class_pred_probs')
    if not len(ids)==len(truth)==len(predicted)==len(probabilities) or len(set(ids))!=len(ids):
        raise ValueError('Prediction lengths differ or duplicate inference IDs')
    if not np.isin(truth,range(5)).all() or not np.isin(predicted,range(5)).all():
        raise ValueError('Expected zero-based five-class labels')
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Invalid saved probabilities')
    node_path=Path(data)/'preprocessed/out_genes_bc.csv'
    # NeDBIT exports ID, feature name, and optionally a seed weight. Supplying
    # two names for three fields makes pandas silently use IDs as an index.
    nodes=pd.read_csv(node_path,sep=r'\s+',header=None)
    if nodes.shape[1] not in (2,3):
        raise ValueError(f'{node_path}: expected ID, feature name and optional seed weight')
    nodes=nodes.iloc[:,:2].copy()
    nodes.columns=['id','name']
    if nodes.isna().any().any():raise ValueError('Missing graph IDs or names')
    numeric_ids=pd.to_numeric(nodes['id'],errors='raise')
    if not np.array_equal(numeric_ids.to_numpy(),np.arange(len(nodes))):
        raise ValueError('Graph IDs must be unique contiguous integers starting at zero')
    nodes['id']=numeric_ids.astype(int)
    if nodes['name'].duplicated().any():raise ValueError('Duplicate graph feature names')
    lookup=nodes.set_index('id')['name']
    names=lookup.loc[ids].to_numpy()
    split=pd.read_csv(generated/f'test_probe_genes_bc_{model}.csv')
    if set(split['test_gene_ids'])!=set(ids):raise ValueError('Inference IDs differ from saved test partition')
    split_names=split.set_index('test_gene_ids')['test_gene_names']
    if not np.array_equal(split_names.loc[ids].to_numpy(),names):raise ValueError('Test table names disagree with node IDs')
    frame=pd.DataFrame({'test_gene_ids':ids.astype(int),'test_gene_names':names,
                        'labels':truth.astype(int)+1,'pred_labels':predicted.astype(int)+1,'pred_probs':probabilities})
    groups={'negative':frame.loc[frame['labels'].isin([3,4,5]) & frame['pred_labels'].isin([3,4,5])],
            'likely_positive':frame.loc[frame['labels'].isin([2,3,4,5]) & frame['pred_labels'].eq(2)]}
    for group,filename in [('negative',f'pred_negatives_{model}.csv'),('likely_positive',f'pred_likely_pos_bc_{model}.csv')]:
        groups[group].to_csv(output/filename,sep='\t',index=False)
    frame.to_csv(output/f'all_test_predictions_{model}.tsv',sep='\t',index=False)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',default='PNA')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    groups=extract(args.data,args.model,args.output)
    print({group:len(frame) for group,frame in groups.items()})

if __name__=='__main__':main()
