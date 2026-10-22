"""Fixed-split TabPFN, TabICL, XGBoost and CatBoost benchmark.

Selects hyperparameters/preprocessing on validation; never refits with validation
labels, and evaluates test once per selected family. See BENCHMARKS.md.
"""
import argparse
import gc
import importlib
import inspect
from pathlib import Path
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from benchmark_common import fresh_output, input_snapshot, load_dataset, scores, sha256, versions, write_json

from pretrained_checkpoints import default_checkpoint, ensure_checkpoint

MODELS = ('TabPFN', 'TabICL', 'XGBoost', 'CatBoost')
TABICL_V2 = 'tabicl-classifier-v2-20260212.ckpt'


def make_model(name, depth, args):
    if name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=args.tree_iterations, max_depth=depth,
                             learning_rate=.05, reg_lambda=10., subsample=.8,
                             colsample_bytree=.8, objective='multi:softprob', num_class=5,
                             tree_method='hist', device='cuda' if args.device == 'cuda' else 'cpu',
                             eval_metric='mlogloss', early_stopping_rounds=args.patience,
                             n_jobs=args.threads, random_state=args.seed)
    if name == 'CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=args.tree_iterations, depth=depth,
                                  learning_rate=.05, l2_leaf_reg=10., loss_function='MultiClass',
                                  eval_metric='MultiClass', random_seed=args.seed,
                                  task_type='GPU' if args.device == 'cuda' else 'CPU',
                                  thread_count=args.threads, allow_writing_files=False, verbose=False)
    if name == 'TabPFN':
        from tabpfn import TabPFNClassifier
        checkpoint = ensure_checkpoint('TabPFN', args.tabpfn_model_path)
        return TabPFNClassifier(n_estimators=args.foundation_estimators, device=args.device,
                                random_state=args.seed, model_path=str(checkpoint))
    if name == 'TabICL':
        from tabicl import TabICLClassifier
        parameters = inspect.signature(TabICLClassifier).parameters
        if 'checkpoint_version' not in parameters:
            raise RuntimeError('Installed tabicl does not support explicit v2 checkpoints; install tabicl>=2.0')
        checkpoint = ensure_checkpoint('TabICL', args.tabicl_model_path)
        return TabICLClassifier(n_estimators=args.foundation_estimators, device=None if args.device == 'auto' else args.device,
                                random_state=args.seed, model_path=str(checkpoint),
                                checkpoint_version=TABICL_V2, n_jobs=args.threads)
    raise ValueError(name)


def fit_model(name, model, x, y, train, val, patience):
    if name == 'XGBoost':
        model.fit(x[train], y[train], eval_set=[(x[val], y[val])], verbose=False)
    elif name == 'CatBoost':
        model.fit(x[train], y[train], eval_set=(x[val], y[val]),
                  early_stopping_rounds=patience, use_best_model=True, verbose=False)
    else:
        model.fit(x[train], y[train])


def predict(model, x):
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if probabilities.shape != (len(x), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid probability output')
    if not np.allclose(probabilities.sum(axis=1), 1., atol=1e-4):
        raise ValueError('Class probabilities do not sum to one')
    aligned = np.zeros((len(x), 5))
    aligned[:, classes] = probabilities
    return aligned.argmax(axis=1), aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--variants', nargs='+', choices=['original', 'ranked_regularized'],
                        default=['original', 'ranked_regularized'])
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--tree-iterations', type=int, default=500)
    parser.add_argument('--depths', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--foundation-estimators', type=int, default=4)
    parser.add_argument('--tabpfn-model-path', default=str(default_checkpoint('TabPFN')), help='Local requested TabPFN v3 checkpoint')
    parser.add_argument('--tabicl-model-path', default=str(default_checkpoint('TabICL')), help='Local requested TabICL v2 checkpoint')
    parser.add_argument('--save-foundation-models', action='store_true', help='Also joblib-dump fitted foundation estimators (potentially large)')
    args = parser.parse_args()
    if min(args.threads, args.tree_iterations, args.patience, args.foundation_estimators, *args.depths) < 1:
        parser.error('Iteration counts, depths, threads and patience must be positive')
    fresh_output(args.output)
    dataset = load_dataset(args.data)
    snapshot = input_snapshot(args.data)
    metadata = dict(seed=args.seed, feature_columns=dataset.columns, device=args.device,
                    split_counts={s: int(dataset.mask(s).sum()) for s in ('train', 'val', 'test')},
                    versions=versions(), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    tabicl_checkpoint=TABICL_V2, tabpfn_checkpoint=args.tabpfn_model_path)
    write_json(args.output / 'config.json', metadata)
    write_json(args.output / 'input_hashes.json', snapshot)
    train, val, test = [dataset.mask(s) for s in ('train', 'val', 'test')]
    validation_results, final, failures = [], [], []
    # Each family is isolated so missing optional dependencies cannot masquerade as results.
    for name in args.models:
        start = time.perf_counter()
        selected = None
        try:
            importlib.import_module({'TabPFN': 'tabpfn', 'TabICL': 'tabicl', 'XGBoost': 'xgboost', 'CatBoost': 'catboost'}[name])
            for variant in args.variants:
                x, scaler = dataset.features(variant)
                depths = args.depths if name in ('XGBoost', 'CatBoost') else [None]
                for depth in depths:
                    np.random.seed(args.seed)
                    model = make_model(name, depth, args)
                    fit_start = time.perf_counter()
                    fit_model(name, model, x, dataset.y, train, val, args.patience)
                    fit_seconds = time.perf_counter() - fit_start
                    infer_start = time.perf_counter()
                    prediction, _ = predict(model, x[val])
                    val_seconds = time.perf_counter() - infer_start
                    validation = scores(dataset.y[val], prediction)
                    row = dict(model=name, variant=variant, depth=depth, validation=validation,
                               fit_seconds=fit_seconds, validation_seconds=val_seconds,
                               parameters={k: str(v) if (not isinstance(v, (str, int, float, bool, type(None))) or isinstance(v, float) and not np.isfinite(v)) else v
                                           for k, v in model.get_params().items()})
                    if name == 'XGBoost':
                        row['best_iteration'] = int(model.best_iteration)
                    elif name == 'CatBoost':
                        row['best_iteration'] = int(model.get_best_iteration())
                    checkpoint = getattr(model, 'model_path_', None)
                    if checkpoint is None and name in ('TabPFN', 'TabICL'):
                        checkpoint = getattr(model, 'model_path', None)
                    if checkpoint and isinstance(checkpoint, (str, Path)) and Path(checkpoint).is_file():
                        row['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256(checkpoint)}
                    validation_results.append(row)
                    key = (validation['accuracy'], validation['macro_f1'])
                    print(f'{name} {variant} depth={depth}: val_acc={key[0]:.4f} macro_f1={key[1]:.4f}', flush=True)
                    if selected is None or key > selected[0]:
                        selected = (key, row, model, scaler, variant)
                    write_json(args.output / 'validation_results.json', validation_results)
                    del model
                    gc.collect()
            _, row, model, scaler, variant = selected
            # Rebuild deterministic feature transform; no fitting/refitting of the selected model.
            x, _ = dataset.features(variant)
            infer_start = time.perf_counter()
            prediction, probabilities = predict(model, x[test])
            test_seconds = time.perf_counter() - infer_start
            result = dict(row, test=scores(dataset.y[test], prediction), test_seconds=test_seconds,
                          seconds=time.perf_counter() - start, status='complete')
            frame = pd.DataFrame({'node_id': np.flatnonzero(test), 'name': dataset.frame.loc[test, 'name'].to_numpy(),
                                  'true': dataset.y[test], 'predicted': prediction})
            for cls in range(5):
                frame[f'probability_{cls}'] = probabilities[:, cls]
            frame.to_csv(args.output / f'{name}_test_predictions.csv', index=False)
            joblib.dump(scaler, args.output / f'{name}_scaler.joblib')
            if name == 'XGBoost':
                model.save_model(str(args.output / f'{name}_model.json'))
            elif name == 'CatBoost':
                model.save_model(str(args.output / f'{name}_model.cbm'))
            elif args.save_foundation_models:
                joblib.dump(model, args.output / f'{name}_model.joblib')
            write_json(args.output / f'{name}_selected.json', result)
            final.append(result)
            print(f'{name}: selected={variant}, test_acc={result["test"]["accuracy"]:.4f}', flush=True)
        except Exception as error:
            failures.append(dict(model=name, status='failed', error=f'{type(error).__name__}: {error}'))
            (args.output / f'{name}_error.log').write_text(traceback.format_exc())
            print(f'{name} FAILED: {error}; see {name}_error.log', file=sys.stderr, flush=True)
        finally:
            selected = None
            if 'model' in locals():
                del model
            gc.collect()
        write_json(args.output / 'summary.json', dict(results=final, failures=failures, config=metadata,
                   input_hashes=snapshot, selection='validation accuracy; macro F1 tie break; no train+validation refit'))
    if input_snapshot(args.data) != snapshot:
        (args.output / 'summary.json').rename(args.output / 'invalid_summary.json')
        raise RuntimeError('Input files changed during benchmark; discard this run')
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(main())
