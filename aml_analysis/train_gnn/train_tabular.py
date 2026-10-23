"""Tabular estimators behind the same config.yaml/main.py entry point as GNNs."""
from pathlib import Path
from types import SimpleNamespace
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, precision_recall_fscore_support

from benchmark_common import Dataset, scores, versions, write_json
from benchmark_tabular import make_model, fit_model, predict, TABICL_V2, default_tabular_device
from pretrained_checkpoints import CHECKPOINTS, file_digest

TABULAR_MODELS = {
    'tabpfn': 'TabPFN', 'tabicl': 'TabICL', 'tabiclv2': 'TabICL',
    'xgboost': 'XGBoost', 'catboost': 'CatBoost',
}


def load_training_dataset(config):
    import preprocess_data

    # Use the same name-aligned raw inputs as GNN preprocessing, without building
    # or sampling millions of graph edges for a model that does not consume them.
    features, labels = preprocess_data.merge_features(config)
    genes = pd.read_csv(config.p_out_genes, sep=r'\s+', header=None)
    names = genes[1].astype(str).to_numpy()
    manifest = pd.read_csv(config.p_split_manifest)
    if not manifest['feature_name'].is_unique or len(set(names)) != len(names):
        raise ValueError('Feature names and split manifest must be unique')
    splits = manifest.set_index('feature_name')['split'].reindex(names).to_numpy()
    if not np.isin(splits, ['train', 'val', 'test']).all():
        raise ValueError('Every feature needs a train, val or test assignment')
    probes = pd.Series(names).str.split('_').str[0].to_numpy()
    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(probes[splits == first]) & set(probes[splits == second]):
            raise ValueError(f'CpG leakage between {first} and {second}')
    x = features.to_numpy(dtype=np.float32)
    label_values = np.asarray(labels)
    if not np.isin(label_values, np.arange(1, 6)).all():
        raise ValueError('Expected propagated integer labels 1 through 5')
    y = label_values.astype(np.int64) - 1
    if not np.isfinite(x).all():
        raise ValueError('Tabular features must be finite')
    ned = len(config.keep_feature_names.split(','))
    frame = pd.DataFrame({'name': names, 'labels': y + 1})
    dataset = Dataset(frame, x, y, splits, list(features.columns), ned)
    train, val, test = [dataset.mask(s) for s in ('train', 'val', 'test')]
    if not all(mask.any() for mask in (train, val, test)) or set(y[train]) != set(range(5)):
        raise ValueError('Need nonempty partitions and all five classes in training')
    return dataset


def train_tabular_model(config):
    name = TABULAR_MODELS[str(config.model_type).lower()]
    started = time.perf_counter()
    dataset = load_training_dataset(config)
    x, y = dataset.x, dataset.y
    names = dataset.frame['name'].to_numpy()
    ned = dataset.ned
    train, val, test = [dataset.mask(s) for s in ('train', 'val', 'test')]
    variant = config.get('tabular_preprocessing', 'original')
    x, scaler = dataset.features(variant)
    # Export the same split tables and normalized features as graph training.
    for split in ('train', 'val', 'test'):
        mask = dataset.mask(split)
        pd.DataFrame({f'{split}_gene_ids': np.flatnonzero(mask),
                      f'{split}_gene_names': names[mask]}).to_csv(
            getattr(config, f'p_{split}_probe_genes'), index=False)
    normalized = pd.DataFrame(x)
    normalized['labels'] = y
    normalized.to_csv(config.p_nedbit_dnam_features_norm, index=False)
    # Defaults are estimator-specific. GNN learning rate, hops, heads, width and
    # epoch settings do not need changing when selecting a tabular estimator.
    options = SimpleNamespace(
        model_parameters=dict(config.get("estimator_params", {})),
        seed=int(config.SEED), device=default_tabular_device(name, config.get('tabular_device')),
        threads=int(config.get('tabular_threads', 4)),
        tree_iterations=int(config.get('tabular_iterations', 500)),
        patience=int(config.get('tabular_patience', 30)),
        foundation_estimators=int(config.get('foundation_estimators', 4)),
        tabpfn_model_path=config.get('tabpfn_model_path', str(Path(config.p_model) / CHECKPOINTS['TabPFN']['filename'])),
        tabicl_model_path=config.get('tabicl_model_path', str(Path(config.p_model) / CHECKPOINTS['TabICL']['filename'])),
    )
    print(f'Initialize model: {name}; preprocessing={variant}', flush=True)
    print(f'Train={train.sum()}, validation={val.sum()}, test={test.sum()}', flush=True)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    model = make_model(name, int(config.get('tabular_depth', 6)), options)
    checkpoint = Path(model.model_path).resolve() if name in CHECKPOINTS else None
    if checkpoint is not None:
        print(f'Pretrained checkpoint: {checkpoint}', flush=True)
    print('Fitting on training nodes only ...', flush=True)
    fit_model(name, model, x, y, train, val, options.patience)
    val_predictions, val_probs = predict(model, x[val])
    validation = scores(y[val], val_predictions)
    print(f'Validation accuracy: {validation["accuracy"]:.4f}', flush=True)
    # Persist fitted model before test inference; no train+validation refit.
    model_path = Path(config.p_torch_model).with_suffix('.joblib')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(model=model, scaler=scaler, variant=variant, model_type=name,
                    feature_columns=dataset.columns, nedbit_columns=ned,
                    versions=versions(), seed=options.seed,
                    tabicl_checkpoint=TABICL_V2 if name == 'TabICL' else None,
                    checkpoint_path=str(checkpoint) if checkpoint is not None else None,
                    checkpoint_sha256=file_digest(checkpoint) if checkpoint is not None else None)
    joblib.dump(artifact, model_path)
    if name in ('XGBoost', 'CatBoost'):
        model.save_model(str(model_path.with_suffix('.json' if name == 'XGBoost' else '.cbm')))
    test_predictions, test_probs = predict(model, x[test])
    test_scores = scores(y[test], test_predictions)
    ids = np.flatnonzero(test).tolist()
    outputs = {
        'p_test_loader_ids': ids,
        'p_true_labels': y[test].tolist(),
        'p_pred_labels': test_predictions.tolist(),
        'p_best_class_pred_probs': test_probs.max(axis=1).tolist(),
        'p_all_class_pred_probs': test_probs.tolist(),
    }
    for field, values in outputs.items():
        path = Path(getattr(config, field))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, path)
    precision, recall, _, _ = precision_recall_fscore_support(
        y[test], test_predictions, average='weighted', zero_division=0)
    best_epoch = int(model.best_iteration) + 1 if name == 'XGBoost' else (
        int(model.get_best_iteration()) + 1 if name == 'CatBoost' else None)
    metrics = dict(
        model_type=str(config.model_type), preprocessing=variant,
        checkpoint_path=artifact['checkpoint_path'], checkpoint_sha256=artifact['checkpoint_sha256'],
        best_epoch=best_epoch, best_val_acc=validation['accuracy'],
        validation=validation, test=test_scores, model_path=str(model_path),
        te_f1_macro=test_scores['macro_f1'],
        te_f1_weighted=float(precision_recall_fscore_support(y[test], test_predictions, average='weighted', zero_division=0)[2]),
        te_f1_micro=test_scores['accuracy'], te_precision=float(precision), te_recall=float(recall),
        tr_loss=[], tr_acc=[], val_loss=[float(log_loss(y[val], val_probs, labels=np.arange(5)))],
        val_acc=[validation['accuracy']], te_loss=[float(log_loss(y[test], test_probs, labels=np.arange(5)))],
        te_acc=[test_scores['accuracy']], versions=versions(), seconds=time.perf_counter() - started,
    )
    output = Path(config.p_plot)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f'all_metrics_{config.model_type}.json', metrics)
    predictions = pd.DataFrame({'node_id': ids, 'name': names[test], 'true': y[test], 'predicted': test_predictions})
    for cls in range(5):
        predictions[f'probability_{cls}'] = test_probs[:, cls]
    predictions.to_csv(output / f'test_predictions_{config.model_type}.csv', index=False)
    import plot_gnn
    plot_gnn.plot_features(x[test], y[test], config.model_type, config,
                           f'UMAP of input features: {config.model_type}', 'test_before_GNN')
    plot_gnn.plot_confusion_matrix(y[test], test_predictions, config.model_type, config)
    print(f'Test accuracy: {test_scores["accuracy"]:.4f}; macro F1: {test_scores["macro_f1"]:.4f}', flush=True)
    print(f'Saved fitted model: {model_path}', flush=True)
    return metrics
