"""Shared fixed-split data validation, preprocessing and metrics for benchmarks."""
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

GRAPH_FEATURES = {'degree', 'ring', 'NetRank', 'NetShort', 'HeatDiff', 'InfoDiff'}
INPUT_FILES = ('df_nebit_dnam_features_bc_PNA.csv', 'node_split_manifest_bc.csv',
               'out_genes_bc.csv', 'data_bc_PNA.pt')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_file(directory, name):
    """Resolve inputs/artifacts in either a flat dataset or the organized layout."""
    directory = Path(directory)
    for folder in (directory, directory / 'preprocessed', directory / 'generated', directory / 'base_files', directory / 'generated_files'):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'{name} not found under {directory} (root, preprocessed or generated)')


def input_snapshot(directory):
    return {name: sha256(dataset_file(directory, name)) for name in INPUT_FILES}


def versions():
    result = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'torch-geometric',
                 'xgboost', 'catboost', 'tabpfn', 'tabicl'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def scores(y, prediction):
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=np.arange(5), average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                confusion_matrix=confusion_matrix(y, prediction, labels=np.arange(5)).tolist())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory is nonempty; choose a new run directory: {path}')
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class Dataset:
    frame: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    splits: np.ndarray
    columns: list
    ned: int

    def mask(self, split):
        return self.splits == split

    def features(self, variant):
        x = self.x.copy()
        if variant == 'original':
            scaler = MinMaxScaler().fit(x[self.mask('train'), :self.ned])
            x[:, :self.ned] = scaler.transform(x[:, :self.ned])
        elif variant == 'ranked_regularized':
            for split in ('train', 'val', 'test'):
                mask = self.mask(split)
                x[mask, :self.ned] = np.column_stack([
                    rankdata(x[mask, j]) / (mask.sum() + 1) for j in range(self.ned)])
            scaler = StandardScaler().fit(x[self.mask('train')])
            x = scaler.transform(x)
        else:
            raise ValueError(f'Unknown feature variant: {variant}')
        return x.astype(np.float32), scaler


def load_dataset(directory):
    directory = Path(directory)
    frame = pd.read_csv(dataset_file(directory, INPUT_FILES[0]))
    manifest = pd.read_csv(dataset_file(directory, INPUT_FILES[1]))
    genes = pd.read_csv(dataset_file(directory, INPUT_FILES[2]), sep=r'\s+', header=None)
    if not {'name', 'labels'} <= set(frame.columns):
        raise ValueError('Feature CSV must contain name and labels columns')
    if not frame['name'].is_unique or not manifest['feature_name'].is_unique:
        raise ValueError('Duplicate feature names in features or split manifest')
    if genes[0].tolist() != list(range(len(frame))) or genes[1].tolist() != frame['name'].tolist():
        raise ValueError('Feature row order does not match graph node IDs')
    splits = manifest.set_index('feature_name')['split'].reindex(frame['name']).to_numpy()
    if not np.isin(splits, ['train', 'val', 'test']).all():
        raise ValueError('Missing or invalid split assignments')
    probes = frame['name'].str.split('_').str[0]
    for first, second in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(probes[splits == first]) & set(probes[splits == second]):
            raise ValueError(f'CpG leakage between {first} and {second}')
    values = frame['labels'].to_numpy(dtype=float)
    if not np.isin(values, np.arange(1, 6)).all():
        raise ValueError('Expected original integer labels 1 through 5')
    y = values.astype(np.int64) - 1
    for split in ('train', 'val', 'test'):
        if not (splits == split).any():
            raise ValueError(f'Empty {split} partition')
    if set(y[splits == 'train']) != set(range(5)):
        raise ValueError('Training partition must contain all five classes')
    columns = frame.drop(columns=['name', 'labels']).columns.tolist()
    ned = sum(c in GRAPH_FEATURES for c in columns)
    if not 0 < ned < len(columns) or not all(c in GRAPH_FEATURES for c in columns[:ned]):
        raise ValueError('Expected leading NeDBIT columns followed by methylation columns')
    x = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Input features contain non-finite values')
    return Dataset(frame, x, y, splits, columns, ned)
