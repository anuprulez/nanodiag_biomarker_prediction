"""Expand AML preprocessing parameters into isolated, reproducible runs."""
from itertools import product
from pathlib import Path
import math

from omegaconf import OmegaConf

PARAMETERS = ('corr_threshold', 'knn_neighbours', 'size_negative')
ANALYSIS_ROOT = Path(__file__).resolve().parent


def preprocessing_configs(config):
    """Accept scalars or lists; lists select a Cartesian parameter sweep."""
    if config.get("config_scope") in ("dataset", "dataset_preprocessing"):
        return [OmegaConf.create(OmegaConf.to_container(config, resolve=False))]
    values = []
    for key in PARAMETERS:
        value = config[key]
        is_list = OmegaConf.is_list(value) or isinstance(value, (list, tuple))
        choices = list(value) if is_list else [value]
        if not choices:
            raise ValueError(f'{key} must not be empty')
        for choice in choices:
            if isinstance(choice, bool) or not isinstance(choice, (int, float)):
                raise ValueError(f'{key} requires numeric values')
            if not math.isfinite(choice):
                raise ValueError(f'{key} requires finite values')
            if key == 'corr_threshold':
                if not 0 <= choice <= 1:
                    raise ValueError('corr_threshold must be between 0 and 1')
            elif choice <= 0 or int(choice) != choice:
                raise ValueError(f'{key} requires positive integers')
        values.append(list(dict.fromkeys(choices)))
    runs = []
    for corr, knn, negative in product(*values):
        current = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        current.corr_threshold = float(corr)
        current.knn_neighbours = int(knn)
        current.size_negative = int(negative)
        dataset_root = Path(str(current.get('p_dataset_root', ANALYSIS_ROOT))).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = ANALYSIS_ROOT / 'train_gnn' / dataset_root
        dataset_root = dataset_root.resolve()
        current.p_base = str(dataset_root / f'data_corr_{corr:g}_knn_n_{int(knn)}_n_neg_{int(negative)}') + '/'
        runs.append(current)
    return runs


def prepare_run_directories(config):
    for key in ('p_base_files', 'p_generated_files'):
        if key in config:
            Path(config[key]).mkdir(parents=True, exist_ok=True)


def add_parameter_arguments(parser):
    parser.add_argument('--corr-threshold', type=float)
    parser.add_argument('--knn-neighbours', type=int)
    parser.add_argument('--size-negative', type=int)


def load_preprocessing_config(path, args=None):
    config = OmegaConf.load(path)
    if args is not None:
        for key in PARAMETERS:
            value = getattr(args, key, None)
            if value is not None:
                config[key] = value
    return config
