#!/usr/bin/env python3
"""Validation-only PNA tuning on an immutable copy of the existing graph.

All candidates use fixed node partitions. Only the validation-selected checkpoint
is evaluated on test, once, after candidate selection has been saved to disk.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import coalesce
from gnn_network import ResidualPNA, GPNA

HERE = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_json(path, value):
    temporary = Path(str(path) + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def loaders(data, neighbors, batch_size, split):
    return NeighborLoader(data, input_nodes=getattr(data, split + '_mask'), num_neighbors=list(neighbors),
                          batch_size=batch_size, shuffle=split == 'train', num_workers=0,
                          subgraph_type='bidirectional')


@torch.no_grad()
def evaluate(model, loader, seed):
    model.eval()
    ids, targets, logits = [], [], []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for batch in loader:
            batch = batch.to(DEVICE)
            n = batch.batch_size
            out = model(batch.x, batch.edge_index)[0][:n]
            ids.extend(batch.n_id[:n].cpu().tolist())
            targets.extend(batch.y[:n].cpu().tolist())
            logits.append(out.cpu())
    probabilities = torch.cat(logits).softmax(-1).numpy()
    prediction = probabilities.argmax(1)
    metrics = {'accuracy': float(accuracy_score(targets, prediction)),
               'macro_f1': float(f1_score(targets, prediction, labels=np.arange(5), average='macro', zero_division=0))}
    return metrics, ids, targets, prediction, probabilities


def train_candidate(data, config, directory, epochs, patience):
    seed_all(int(config.SEED))
    model = ResidualPNA(config, data).to(DEVICE)
    counts = torch.bincount(data.y[data.train_mask], minlength=5).float()
    if (counts == 0).any(): raise ValueError('Training partition lacks a class')
    power = float(config.class_weight_power)
    weights = (counts.sum() / (5 * counts)).pow(power)
    weights /= weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=4)
    train_loader = loaders(data, config.neighbors_spread, config.batch_size, 'train')
    val_loader = loaders(data, config.neighbors_spread, config.batch_size, 'val')
    best = (-1., -1.)
    best_epoch, waiting, history = 0, 0, []
    directory.mkdir()
    OmegaConf.save(config, directory / 'config.yaml')
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0., 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.x, batch.edge_index)[0][:batch.batch_size]
            loss = criterion(output, batch.y[:batch.batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            total_loss += float(loss.detach()) * batch.batch_size
            total += batch.batch_size
        validation, *_ = evaluate(model, val_loader, int(config.SEED))
        scheduler.step(validation['macro_f1'])
        row = dict(epoch=epoch, train_loss=total_loss / total, **validation)
        history.append(row)
        key = (validation['macro_f1'], validation['accuracy'])
        if key > best:
            best = key; best_epoch = epoch; waiting = 0
            torch.save(model.state_dict(), directory / 'best_model.pt')
        else: waiting += 1
        save_json(directory / 'history.json', history)
        print(f'{directory.name} epoch={epoch} loss={row["train_loss"]:.4f} val_acc={key[1]:.4f} val_macro_f1={key[0]:.4f} best_epoch={best_epoch}', flush=True)
        if waiting >= patience: break
    result = dict(trial=directory.name, validation_macro_f1=best[0], validation_accuracy=best[1],
                  best_epoch=best_epoch, completed_epochs=len(history), seconds=time.perf_counter()-started,
                  class_weights=weights.tolist(), config=OmegaConf.to_container(config, resolve=True))
    save_json(directory / 'result.json', result)
    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE.parent / 'config/config.yaml')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'pna_improvement')
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    args.config = args.config.resolve(); args.output = args.output.resolve()
    os.chdir(HERE)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory to preserve previous experiments')
    args.output.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(args.config)
    config.model_type = 'PNA'
    source = Path(config.p_torch_data).resolve()
    # Snapshot the graph before tuning: other preprocessing jobs cannot change it.
    graph = args.output / 'data_aml_PNA.pt'
    source_hash = digest(source); shutil.copy2(source, graph)
    if digest(graph) != source_hash or digest(source) != source_hash: raise ValueError('Source graph changed during copy')
    data = torch.load(graph, weights_only=False).cpu()
    if data.num_node_features != int(config.gene_dim): raise ValueError('Unexpected input dimensions')
    split_names = ['train', 'val', 'test']
    codes = torch.full((data.num_nodes,), -1, dtype=torch.long)
    node_names = np.full(data.num_nodes, '', dtype=object)
    for code, split in enumerate(split_names):
        mask = getattr(data, split + '_mask')
        if (codes[mask] != -1).any(): raise ValueError('Overlapping node partitions')
        codes[mask] = code
        table_path = Path(config[f'p_{split}_probe_genes']).resolve()
        table = pd.read_csv(table_path)
        ids = table[split + '_gene_ids'].to_numpy()
        if set(ids) != set(mask.nonzero().view(-1).tolist()): raise ValueError('Split table disagrees with stored graph')
        node_names[ids] = table[split + '_gene_names'].to_numpy()
        shutil.copy2(table_path, args.output / table_path.name)
    if (codes < 0).any(): raise ValueError('Unassigned nodes')
    a, b = data.edge_index
    if not torch.equal(codes[a], codes[b]): raise ValueError('Cross-partition graph edges')
    for i, split in enumerate(split_names):
        probes = {name.split('_',1)[0] for name in node_names[codes.numpy()==i]}
        for j in range(i):
            if probes & {name.split('_',1)[0] for name in node_names[codes.numpy()==j]}: raise ValueError('CpG leakage')
    snapshot = dict(graph_sha256=source_hash, source_graph=str(source), nodes=data.num_nodes,
                    edges=data.num_edges, inputs=data.num_node_features,
                    partitions={s:int(getattr(data,s+'_mask').sum()) for s in split_names})
    save_json(args.output / 'input_snapshot.json', snapshot)
    # Validation-only reference from existing fitted PNA. No test evaluation here.
    baseline_path = Path(config.p_torch_model).resolve()
    shutil.copy2(baseline_path, args.output / 'original_pna.pt')
    seed_all(int(config.SEED))
    baseline = GPNA(config, data).to(DEVICE)
    baseline.load_state_dict(torch.load(args.output/'original_pna.pt', map_location=DEVICE, weights_only=True))
    baseline_val, *_ = evaluate(baseline, loaders(data, config.neighbors_spread, config.batch_size, 'val'), int(config.SEED))
    save_json(args.output/'baseline_validation.json', baseline_val)
    print('Original PNA validation:', baseline_val, flush=True)
    del baseline
    candidates = [
        dict(name='residual2_balanced',pna_layers=2,hidden_dim=64,learning_rate=.001,dropout=.2,weight_decay=.001,class_weight_power=.5,neighbors_spread=[25,10]),
        dict(name='residual1_balanced',pna_layers=1,hidden_dim=64,learning_rate=.001,dropout=.2,weight_decay=.001,class_weight_power=.5,neighbors_spread=[25]),
        dict(name='residual2_regularized',pna_layers=2,hidden_dim=128,learning_rate=.0005,dropout=.3,weight_decay=.005,class_weight_power=.5,neighbors_spread=[25,10]),
        dict(name='residual2_unweighted',pna_layers=2,hidden_dim=64,learning_rate=.001,dropout=.2,weight_decay=.001,class_weight_power=0.,neighbors_spread=[25,10]),
    ]
    results = []
    for options in candidates:
        options = dict(options); name = options.pop('name')
        current = copy.deepcopy(config)
        for key,value in options.items():current[key]=value
        current.pna_variant='residual';current.batch_size=256
        results.append(train_candidate(data,current,args.output/name,args.epochs,args.patience))
        save_json(args.output/'validation_results.json',results)
    selected = max(results,key=lambda row:(row['validation_macro_f1'],row['validation_accuracy']))
    # Commit selection before accessing any test results for these candidates.
    save_json(args.output/'selected.json',selected)
    winner = args.output/selected['trial']
    winning_config=OmegaConf.load(winner/'config.yaml')
    seed_all(int(winning_config.SEED))
    model=ResidualPNA(winning_config,data).to(DEVICE)
    model.load_state_dict(torch.load(winner/'best_model.pt',map_location=DEVICE,weights_only=True))
    metrics,ids,truth,prediction,probabilities=evaluate(model,loaders(data,winning_config.neighbors_spread,winning_config.batch_size,'test'),int(winning_config.SEED))
    metrics.update(classification_report=classification_report(truth,prediction,labels=np.arange(5),output_dict=True,zero_division=0),
                   confusion_matrix=confusion_matrix(truth,prediction,labels=np.arange(5)).tolist())
    save_json(args.output/'test_metrics.json',metrics)
    frame=pd.DataFrame({'node_id':ids,'name':node_names[ids],'true':truth,'predicted':prediction})
    for cls in range(5):frame[f'probability_{cls}']=probabilities[:,cls]
    frame.to_csv(args.output/'test_predictions_PNA.csv',index=False)
    shutil.copy2(winner/'best_model.pt',args.output/'best_model_PNA.pt')
    shutil.copy2(winner/'config.yaml',args.output/'selected_config.yaml')
    from artifact_layout import migrate_generated
    migrate_generated(args.output)
    print('Selected',selected['trial'],'validation',selected['validation_macro_f1'],'test',metrics,flush=True)


if __name__=='__main__':main()
