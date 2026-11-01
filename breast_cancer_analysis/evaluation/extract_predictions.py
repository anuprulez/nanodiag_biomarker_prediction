#!/usr/bin/env python3
"""Export AML-equivalent prediction groups aligned by saved inference node IDs."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artifact_layout import artifact_file


def prediction_frame(data,model,generated_dir=None):
    import torch
    generated=Path(generated_dir) if generated_dir is not None else Path(data)/'generated'
    def load(prefix):
        path=artifact_file(generated,f'{prefix}_bc_{model}.pt')
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
    split=pd.read_csv(artifact_file(generated,f'test_probe_genes_bc_{model}.csv'))
    if set(split['test_gene_ids'])!=set(ids):raise ValueError('Inference IDs differ from saved test partition')
    split_names=split.set_index('test_gene_ids')['test_gene_names']
    if not np.array_equal(split_names.loc[ids].to_numpy(),names):raise ValueError('Test table names disagree with node IDs')
    frame=pd.DataFrame({'test_gene_ids':ids.astype(int),'test_gene_names':names,
                        'labels':truth.astype(int)+1,'pred_labels':predicted.astype(int)+1,'pred_probs':probabilities})
    return frame


def extract(data,model,output,generated_dir=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame=prediction_frame(data,model,generated_dir)
    groups={'negative':frame.loc[frame['labels'].isin([3,4,5]) & frame['pred_labels'].isin([3,4,5])],
            'likely_positive':frame.loc[frame['labels'].isin([2,3,4,5]) & frame['pred_labels'].eq(2)]}
    for group,filename in [('negative',f'pred_negatives_{model}.csv'),('likely_positive',f'pred_likely_pos_bc_{model}.csv')]:
        groups[group]=groups[group].sort_values('pred_probs',ascending=False,kind='stable').copy()
        groups[group].to_csv(output/filename,sep='\t',index=False)
    frame.to_csv(output/f'all_test_predictions_{model}.tsv',sep='\t',index=False)
    return groups



def comparison_groups(data,model,output,generated_dir=None):
    """Use current exported selections, including user-defined XAI label rules."""
    generated=Path(generated_dir) if generated_dir is not None else Path(data)/'generated'
    paths={group:artifact_file(generated,filename) for group,filename in [
        ('negative',f'pred_negatives_{model}.csv'),
        ('likely_positive',f'pred_likely_pos_bc_{model}.csv')]}
    present={group:path.is_file() for group,path in paths.items()}
    if not any(present.values()):
        return extract(data,model,output,generated_dir),[]
    if not all(present.values()):
        raise ValueError('Incomplete prediction-group exports; regenerate both groups using collect_pred_labels before comparison')
    reference=prediction_frame(data,model,generated_dir).set_index('test_gene_ids')
    groups={}
    for group,path in paths.items():
        frame=pd.read_csv(path,sep='\t')
        required={'test_gene_ids','test_gene_names','labels','pred_labels','pred_probs'}
        if not required<=set(frame):raise ValueError(f'{path}: missing prediction columns')
        if frame.test_gene_ids.duplicated().any() or frame.test_gene_names.duplicated().any():
            raise ValueError(f'{path}: duplicate exported inference IDs or names')
        if not frame.test_gene_ids.isin(reference.index).all():raise ValueError(f'{path}: exported IDs are outside the current test partition')
        expected=reference.loc[frame.test_gene_ids]
        for column in ['test_gene_names','labels','pred_labels']:
            if not np.array_equal(frame[column].to_numpy(),expected[column].to_numpy()):
                raise ValueError(f'{path}: exported {column} differs from current predictions; regenerate XAI prediction exports')
        if not np.allclose(frame.pred_probs.to_numpy(float),expected.pred_probs.to_numpy(float),rtol=float(np.finfo(np.float32).eps),atol=1e-8):
            raise ValueError(f'{path}: exported probabilities differ from current predictions; regenerate XAI prediction exports')
        groups[group]=frame.sort_values('pred_probs',ascending=False,kind='stable').copy()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for group,path in paths.items():
        destination=output/path.name
        if destination.resolve()!=path.resolve():
            groups[group].to_csv(destination,sep='\t',index=False)
    return groups,list(paths.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',default='PNA')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    groups=extract(args.data,args.model,args.output)
    print({group:len(frame) for group,frame in groups.items()})

if __name__=='__main__':main()
