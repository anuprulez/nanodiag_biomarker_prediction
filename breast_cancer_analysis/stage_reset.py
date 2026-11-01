"""Reset only the selected stage's outputs inside its dataset folder."""
from pathlib import Path
import re
import shutil
from artifact_layout import artifact_kind, DATA_KEYS, MODEL_KEYS
from train_gnn.model_settings import apply_model_settings

XAI_PREFIXES = ('df_xai_', 'Feature_importance_', 'Explanation_', 'Explaination_')


def remove_output(folder, path):
    folder, path = Path(folder).resolve(), Path(path)
    if path.resolve() == folder or not path.resolve().is_relative_to(folder):
        raise ValueError(f'Refusing to reset output outside dataset: {path}')
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def reset_stage_outputs(folder, stage, config, models=None):
    """Preserve raw inputs, tuned settings, and upstream artifacts for each stage."""
    folder = Path(folder).resolve()
    if stage == 'preprocess':
        remove_output(folder, config.p_base_files)
        return
    if stage == 'compare-predictions':
        remove_output(folder, folder / 'evaluation')
        return
    if stage not in ('train', 'xai'):
        return  # Tuning resets its selected study directories via --fresh.
    selected = ['PNA'] if stage == 'xai' else (models or list(config.model_type))
    if stage == 'train' and any(name not in config.model_type for name in selected):
        raise ValueError('Requested model has no tuned parameters in this dataset config')
    roots = set()
    exact = set()
    xai_exact = set()
    for name in selected:
        current = apply_model_settings(config, name)
        roots.add(Path(current.p_generated_files))
        for key in ('p_data', 'p_model', 'p_plot'):
            if key in current: roots.add(Path(current[key]))
        for key in DATA_KEYS | MODEL_KEYS:
            if key in current:
                path = Path(current[key])
                if path.parent in roots: exact.add(path)
        if 'p_pred_likely_pos_no_training_genes_probes' in current:
            xai_exact.add(Path(current.p_pred_likely_pos_no_training_genes_probes))
    token = re.compile(r'(?:^|_)(' + '|'.join(re.escape(str(n)) for n in selected) + r')(?:_|\.|$)', re.I)
    for root in sorted(roots):
        if not root.resolve().is_relative_to(folder):
            raise ValueError(f'Generated output root outside dataset: {root}')
        if not root.is_dir(): continue
        # Only inspect flat exports and the three standard artifact directories.
        # Optuna studies, including their trial checkpoints, are upstream inputs.
        paths = set(root.iterdir())
        for kind in ('data', 'models', 'plots'):
            if (root / kind).is_dir(): paths.update((root / kind).iterdir())
        for path in paths:
            if not path.is_file(): continue
            is_xai = path.name.startswith(XAI_PREFIXES)
            if stage == 'xai':
                remove = is_xai or path in xai_exact
            else:
                remove = (path in exact or token.search(path.name) or
                          path.name.startswith(('model_comparison', 'Heatmap_avg_edges_'))) and artifact_kind(path.name)
            if remove: remove_output(folder, path)
