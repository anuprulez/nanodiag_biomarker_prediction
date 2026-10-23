#!/usr/bin/env python3
"""Optuna model search on fixed breast cancer train/validation partitions; no test evaluation."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from preprocessing_grid import add_parameter_arguments, load_preprocessing_config, preprocessing_configs
from model_settings import apply_model_settings
from seed_inputs import validate_seed_manifest

GNN_MODELS = ('PNA', 'GCN', 'GraphSAGE', 'GATv2', 'GraphTransformer')
MODELS = GNN_MODELS + ('XGBoost', 'CatBoost', 'TabPFN', 'TabICL')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def suggest_parameters(trial, search, model):
    space = dict(search.spaces.GNN if model in GNN_MODELS else search.spaces[model])
    if model in ('GATv2', 'GraphTransformer'):
        space.update(search.spaces.get('Attention', {}))
    if model in GNN_MODELS and model in search.spaces:
        space.update(search.spaces[model])
    parameters = {}
    for name, spec in space.items():
        kind = spec['type']
        if kind == 'categorical':
            choices = list(spec['choices'])
            if name == 'neighbors_spread':
                # Optuna categorical storage requires primitive choices.
                choices = [json.dumps(list(choice), separators=(',', ':')) for choice in choices]
            parameters[name] = trial.suggest_categorical(name, choices)
        elif kind in ('int', 'float'):
            options = {key: spec[key] for key in ('log', 'step') if key in spec}
            function = trial.suggest_int if kind == 'int' else trial.suggest_float
            parameters[name] = function(name, spec['low'], spec['high'], **options)
        else:
            raise ValueError(f'Unknown search distribution: {kind}')
    if model in ('GATv2', 'GraphTransformer') and parameters['hidden_dim'] % parameters['heads']:
        import optuna
        raise optuna.TrialPruned('hidden_dim must be divisible by heads')
    return parameters


def trial_config(base, parameters, model):
    current = copy.deepcopy(base)
    if model in GNN_MODELS:
        parameters = dict(parameters)
        fanout = parameters.pop('neighbor_fanout', None)
        sampled_hops = parameters.pop('neighbors_spread', None)
        if isinstance(sampled_hops, str):
            sampled_hops = json.loads(sampled_hops)
        if sampled_hops is not None and (not isinstance(sampled_hops, (list, tuple)) or
                not sampled_hops or any(isinstance(value, bool) or not isinstance(value, int) or
                                        (value != -1 and value < 1) for value in sampled_hops)):
            raise ValueError('neighbors_spread choices must be nonempty lists of positive integers or -1')
        current = OmegaConf.merge(current, parameters)
        # Sample as many hops as the number of message-passing layers.
        hops = list(sampled_hops if sampled_hops is not None else base.neighbors_spread)
        if fanout is not None:
            hops = [int(fanout), max(1, int(fanout) // 2), max(1, int(fanout) // 4)]
        if not hops:
            raise ValueError('neighbors_spread must not be empty')
        current.neighbors_spread = (hops + [hops[-1]] * int(current.gnn_layers))[:int(current.gnn_layers)]
    else:
        current.estimator_params = OmegaConf.merge(current.get('estimator_params', {}), parameters)
    return current


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gnn_objective(trial, config, data, directory, metric):
    import optuna
    import torch
    from torch_geometric.loader import NeighborLoader
    from utils import choose_model
    from train_model_sub_graph import train_one_epoch, val_evaluate

    seed_all(int(config.SEED))
    device = torch.device(config.get('hpo_device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = choose_model(config, data, str(config.model_type), True).to(device)
    counts = torch.bincount(data.y[data.train_mask], minlength=int(config.num_classes)).float()
    if (counts == 0).any():
        raise ValueError('Training partition lacks a class')
    power = float(config.get('class_weight_power', 1.0))
    weights = (counts.sum() / (len(counts) * counts)).pow(power)
    weights /= weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate),
                                  weight_decay=float(config.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=int(config.lr_patience))
    def loader(split):
        return NeighborLoader(data, input_nodes=getattr(data, split + '_mask'),
                              num_neighbors=list(config.neighbors_spread), batch_size=int(config.batch_size),
                              shuffle=split == 'train', num_workers=int(config.get('num_workers', 0)),
                              subgraph_type='bidirectional')
    train_loader, val_loader = loader('train'), loader('val')
    best, waiting, history = -float('inf'), 0, []
    try:
        for epoch in range(1, int(config.n_epo) + 1):
            train_one_epoch(train_loader, model, optimizer, criterion, device, config.grad_clip_norm)
            # Validation sampling has its own repeatable random state.
            devices = [device.index or 0] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(config.SEED))
                loss, accuracy, _, macro_f1 = val_evaluate(val_loader, model, criterion, device)
            score = float(macro_f1 if metric == 'macro_f1' else accuracy)
            if not np.isfinite(score):
                raise ValueError('Non-finite validation metric')
            scheduler.step(score)
            history.append(dict(epoch=epoch, loss=float(loss), accuracy=float(accuracy), macro_f1=float(macro_f1)))
            if score > best:
                best, waiting = score, 0
                torch.save(model.state_dict(), directory / 'model.pt')
                trial.set_user_attr('best_epoch', epoch)
            else:
                waiting += 1
            trial.report(best, epoch)
            write_json(directory / 'validation_history.json', history)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if waiting >= int(config.early_stopping_patience):
                break
        return best
    finally:
        del model, optimizer, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def tabular_objective(trial, config, dataset, directory, metric):
    import joblib
    from benchmark_tabular import make_model, fit_model, predict, default_tabular_device
    from benchmark_common import scores
    seed_all(int(config.SEED))
    x, scaler = dataset.features(str(config.get('tabular_preprocessing', 'original')))
    train, val = dataset.mask('train'), dataset.mask('val')
    options = SimpleNamespace(
        seed=int(config.SEED), device=default_tabular_device(config.model_type, config.get('tabular_device')),
        threads=int(config.get('tabular_threads', 4)), tree_iterations=int(config.get('tabular_iterations', 500)),
        patience=int(config.get('tabular_patience', 30)), foundation_estimators=int(config.get('foundation_estimators', 4)),
        tabpfn_model_path=config.get('tabpfn_model_path'), tabicl_model_path=config.get('tabicl_model_path'),
        model_parameters=OmegaConf.to_container(config.get('estimator_params', OmegaConf.create({})), resolve=True))
    name = str(config.model_type)
    model = make_model(name, int(config.get('tabular_depth', 6)), options)
    try:
        fit_model(name, model, x, dataset.y, train, val, options.patience)
        predictions, _ = predict(model, x[val])
        validation = scores(dataset.y[val], predictions)
        trial.set_user_attr('validation', validation)
        if name == 'XGBoost':
            trial.set_user_attr('best_iteration', int(model.best_iteration))
        elif name == 'CatBoost':
            trial.set_user_attr('best_iteration', int(model.get_best_iteration()))
        joblib.dump(dict(model=model, scaler=scaler, columns=dataset.columns,
                         preprocessing=str(config.get('tabular_preprocessing', 'original'))), directory / 'model.joblib')
        return float(validation[metric])
    finally:
        del model
        gc.collect()


def input_fingerprint(config, search):
    """Refuse to resume a study after data, training settings, or search changes."""
    from benchmark_common import sha256, versions
    import optuna
    fields = ('p_out_genes', 'p_out_links', 'p_nedbit_features', 'p_out_gene_rankings',
              'p_combined_pos_neg_signals', 'p_split_manifest', 'p_seeds')
    payload = dict(inputs={key: sha256(config[key]) for key in fields},
                   config=OmegaConf.to_container(config, resolve=True),
                   search=OmegaConf.to_container(search, resolve=True), versions=versions(), optuna_version=optuna.__version__)
    # Adding more trials does not change the objective or search space.
    payload['search'].pop('n_trials', None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def optimize_model(config, search, name, n_trials, timeout=None, study_tag=None):
    import optuna
    base = apply_model_settings(config, name)
    # Flatten overrides so the winning config cannot reapply stale defaults.
    base.pop('model_overrides', None)
    output = Path(base.p_generated_files) / 'optuna'
    if study_tag is not None:
        if not study_tag or Path(study_tag).name != study_tag or study_tag in ('.', '..'):
            raise ValueError('study_tag must be a single directory name')
        output /= study_tag
    directory = (output / name).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprint(base, search)
    study = optuna.create_study(study_name=name, storage='sqlite:///' + str(directory / 'study.sqlite3'),
                               direction='maximize', load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=int(base.SEED),
                                                                n_startup_trials=int(search.startup_trials)),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=int(search.startup_trials),
                                                                 n_warmup_steps=int(search.pruning_warmup_epochs)))
    previous = study.user_attrs.get('input_fingerprint')
    if previous is not None and previous != fingerprint:
        raise ValueError(f'{directory}: inputs/settings changed; use a new study directory or restore the original settings')
    study.set_user_attr('input_fingerprint', fingerprint)
    # Persist the seeded sampler as well as trials, avoiding repeated suggestions
    # when a fresh process resumes a study during its random startup phase.
    import joblib
    sampler_path = directory / 'sampler.joblib'
    if sampler_path.exists():
        study.sampler = joblib.load(sampler_path)

    def save_sampler(study, trial):
        temporary = sampler_path.with_suffix('.part')
        joblib.dump(study.sampler, temporary)
        temporary.replace(sampler_path)
    if not study.trials:
        for candidate in search.get('initial_trials', {}).get(name, []):
            study.enqueue_trial(OmegaConf.to_container(candidate, resolve=True))
    # Intermediate feature exports are isolated from ordinary model artifacts.
    prepared = copy.deepcopy(base)
    prepared.p_generated_files = str(directory / 'prepared') + '/'
    Path(prepared.p_generated_files).mkdir(exist_ok=True)
    if name in GNN_MODELS:
        import torch
        import preprocess_data
        from train_model_sub_graph import validate_predefined_splits
        import pandas as pd
        preprocess_data.read_files(prepared)
        data = torch.load(prepared.p_torch_data, weights_only=False).cpu()
        validate_predefined_splits(data, pd.read_csv(prepared.p_out_genes, sep=r'\s+', header=None), prepared)
        dataset = None
    else:
        from train_tabular import load_training_dataset
        dataset = load_training_dataset(prepared)
        data = None

    def objective(trial):
        parameters = suggest_parameters(trial, search, name)
        current = trial_config(base, parameters, name)
        trial_directory = directory / f'trial_{trial.number:05d}'
        trial_directory.mkdir()
        if name in GNN_MODELS:
            return gnn_objective(trial, current, data, trial_directory, str(search.metric))
        return tabular_objective(trial, current, dataset, trial_directory, str(search.metric))

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, gc_after_trial=True,
                       callbacks=[save_sampler])
    finally:
        study.trials_dataframe().to_csv(directory / 'trials.csv', index=False)
    if input_fingerprint(base, search) != fingerprint:
        raise RuntimeError('Inputs changed during optimization; discard this study')
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f'No completed trials for {name}; inspect {directory}')
    winner = study.best_trial
    best_config = trial_config(base, winner.params, name)
    best_config.checkpoint_selection_metric = str(search.metric)
    # Export absolute input paths and a dedicated final-training output directory.
    final = directory / 'final'
    best_config.p_generated_files = str(final) + '/'
    resolved = OmegaConf.create(OmegaConf.to_container(best_config, resolve=True))
    for key, value in list(resolved.items()):
        if key.startswith('p_') and isinstance(value, str) and value:
            resolved[key] = str(Path(value).resolve()) + ('/' if value.endswith('/') else '')
    OmegaConf.save(resolved, directory / 'best_config.yaml')
    extension = 'pt' if name in GNN_MODELS else 'joblib'
    shutil.copy2(directory / f'trial_{winner.number:05d}' / f'model.{extension}', directory / f'best_model.{extension}')
    best_parameters = dict(winner.params)
    if 'neighbors_spread' in best_parameters:
        best_parameters['neighbors_spread'] = list(best_config.neighbors_spread)
    write_json(directory / 'best_params.json', dict(model=name, trial=winner.number,
               validation_metric=str(search.metric), validation_score=winner.value, parameters=best_parameters,
               input_fingerprint=fingerprint, test_evaluated=False))
    print(f'{name}: best validation {search.metric}={winner.value:.5f}; results: {directory}', flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE.parent / 'config/config.yaml')
    parser.add_argument('--search-space', type=Path, default=HERE.parent / 'config/optuna.yaml')
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--study-tag', help='Separate study group under generated/optuna/')
    parser.add_argument('--trials', type=int, help='Additional trials per model and dataset')
    parser.add_argument('--timeout', type=float, help='Seconds per model and dataset')
    add_parameter_arguments(parser)
    args = parser.parse_args()
    config = load_preprocessing_config(args.config.resolve(), args)
    search = OmegaConf.load(args.search_space.resolve())
    if search.metric not in ('macro_f1', 'accuracy'):
        parser.error('metric must be macro_f1 or accuracy')
    n_trials = args.trials if args.trials is not None else int(search.n_trials)
    if n_trials < 1 or (args.timeout is not None and args.timeout <= 0):
        parser.error('trials and timeout must be positive')
    models = args.models if args.models is not None else list(search.models)
    if not models or any(name not in MODELS for name in models):
        parser.error(f'Choose models from {MODELS}')
    os.chdir(HERE)
    for current in preprocessing_configs(config):
        validate_seed_manifest(current)
        for name in dict.fromkeys(models):
            optimize_model(current, search, name, n_trials, args.timeout, args.study_tag)


if __name__ == '__main__':
    main()
