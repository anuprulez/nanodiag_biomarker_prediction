"""Train each model in config.model_type and export its individual artifacts."""
import copy
import os
from pathlib import Path
import traceback
import sys

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from seed_inputs import validate_seed_manifest
from dataset_configs import downstream_configs
from preprocessing_grid import preprocessing_configs, add_parameter_arguments, load_preprocessing_config
from model_settings import apply_model_settings


GNN_MODELS = {"pna", "gcn", "graphsage", "gatv2", "graphtransformer"}
TABULAR_NAMES = {"tabpfn", "tabicl", "tabiclv2", "xgboost", "catboost"}
# Files written by preprocessing/training; each must have a distinct path per model.
MODEL_OUTPUTS = (
    "p_dnam_features", "p_nedbit_dnam_features", "p_nedbit_dnam_features_norm",
    "p_test_probe_genes", "p_train_probe_genes", "p_val_probe_genes",
    "p_torch_data", "p_torch_embed", "p_torch_embed_batch_norm", "p_torch_model",
    "p_true_labels", "p_pred_labels", "p_all_class_pred_probs",
    "p_best_class_pred_probs", "p_test_loader_ids",
)


def model_configs(config):
    """Copy unresolved config, then bind interpolations to ONE model per run."""
    requested = config.model_type
    is_list = OmegaConf.is_list(requested) or isinstance(requested, list)
    names = list(requested) if is_list else [requested]
    if not names or any(not isinstance(name, str) or name.lower() not in
                        GNN_MODELS | TABULAR_NAMES for name in names):
        raise ValueError("model_type must be a model name or nonempty list of PNA, GCN, "
                         "GraphSAGE, GATv2, GraphTransformer, TabPFN, TabICL, XGBoost, CatBoost")
    keys = ["tabicl" if name.lower() == "tabiclv2" else name.lower() for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError("model_type contains duplicate models (including aliases)")
    configs, owners = [], {}
    for name in names:
        name = "TabICL" if name.lower() in {"tabicl", "tabiclv2"} else name
        current = apply_model_settings(config, name)
        if current.get('config_scope')=='dataset':
            dataset_root=Path(current.p_base).resolve()
            for field in (*MODEL_OUTPUTS,'p_generated_files','p_plot','p_data','p_model'):
                if field in current and not Path(str(current[field])).expanduser().resolve().is_relative_to(dataset_root):
                    raise ValueError(f'{name}.{field} points outside the selected dataset folder: {current[field]}')
        for field in MODEL_OUTPUTS:
            if field not in current:
                continue
            path = Path(str(current[field])).expanduser().resolve()
            if path in owners:
                raise ValueError(f"Output collision: {name}.{field} and {owners[path]} use {path}; "
                                 "include ${model_type} in output filenames")
            owners[path] = f"{name}.{field}"
        configs.append(current)
    return configs, is_list


def train_one_model(config):
    model_type = str(config.model_type)
    if model_type.lower() in TABULAR_NAMES:
        from train_tabular import train_tabular_model
        return train_tabular_model(config)
    import preprocess_data
    import train_model_sub_graph
    labels = preprocess_data.read_files(config)
    return train_model_sub_graph.train_gnn_model(config, labels, model_type)


def run_training(config_path="../config/config.yaml", args=None):
    configs = downstream_configs(config_path, args)
    if args is not None and getattr(args,'models',None):
        for config in configs:
            names=list(args.models)
            if config.get('config_scope')=='dataset' and any(name not in config.model_type for name in names):
                raise ValueError('Requested model has no tuned parameters in this dataset config')
            config.model_type=names
    results = [run_training_config(config) for config in configs]
    return results[0] if len(results) == 1 else results


def run_training_config(config):
    validate_seed_manifest(config)
    configs, is_list = model_configs(config)  # Validate ALL names/paths before writing files.
    results, failures = {}, {}
    for current in configs:
        name = str(current.model_type)
        print(f"\n=== Training {name} ===", flush=True)
        for field in MODEL_OUTPUTS:
            if field in current:
                Path(str(current[field])).expanduser().parent.mkdir(parents=True, exist_ok=True)
        for field in ('p_plot','p_data','p_model'):
            Path(current[field]).mkdir(parents=True, exist_ok=True)
        try:
            # Reset randomness so list order cannot change model initialization/sampling.
            import random
            import numpy as np
            import torch
            random.seed(int(current.SEED))
            np.random.seed(int(current.SEED))
            torch.manual_seed(int(current.SEED))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(current.SEED))
            results[name] = train_one_model(current)
            if current.get('artifact_layout')=='split_v1':
                from evaluation.extract_predictions import extract
                extract(current.p_base,name,current.p_data,generated_dir=current.p_generated_files)
        except Exception as error:
            if not is_list:
                raise
            failures[name] = f"{type(error).__name__}: {error}"
            log = Path(current.p_data) / f"training_error_{name}.log"
            log.write_text(traceback.format_exc())
            print(f"{name} FAILED: {error}; see {log}", flush=True)
        finally:
            import gc
            gc.collect()
            if 'torch' in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
    # Use only results from this invocation; stale metrics files are never mixed in.
    from plot_gnn import plot_model_comparison
    plot_model_comparison(results, [str(c.model_type) for c in configs], configs[0])
    if failures:
        raise RuntimeError(f"Training failed for {failures}; remaining models were attempted")
    return results if is_list else results[str(configs[0].model_type)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config/config.yaml")
    parser.add_argument("--models", nargs="+", help="Train only these tuned models")
    parser.add_argument("--folder", type=Path, help="Dataset folder relative to aml_analysis, or an absolute path")
    add_parameter_arguments(parser)
    args = parser.parse_args()
    config_path = args.config.resolve()
    # Existing configs express data/model paths relative to train_gnn.
    os.chdir(HERE)
    run_training(config_path, args)
