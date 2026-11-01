"""Shared generated/{data,plots,models} paths and legacy artifact migration."""
from pathlib import Path
from omegaconf import OmegaConf

LAYOUT='split_v1'
DATA_KEYS=set('p_dnam_features p_nedbit_dnam_features p_nedbit_dnam_features_norm p_test_probe_genes p_train_probe_genes p_val_probe_genes p_pred_likely_pos p_pred_likely_pos_no_training_genes_probes p_pred_negatives'.split())
MODEL_KEYS=set('p_torch_data p_torch_embed p_torch_embed_batch_norm p_true_labels p_torch_model p_pred_labels p_all_class_pred_probs p_best_class_pred_probs p_test_loader_ids'.split())


def apply_artifact_layout(config):
    if config.get('artifact_layout')!=LAYOUT or 'p_generated_files' not in config:return config
    root=str(config.p_generated_files).rstrip('/')+'/'
    raw=OmegaConf.to_container(config,resolve=False)
    config.p_data=root+'data/';config.p_plot=root+'plots/';config.p_model=root+'models/'
    for key in DATA_KEYS|MODEL_KEYS:
        if key not in raw:continue
        value=str(raw[key])
        for prefix in ('${p_generated_files}','${p_model}','${p_data}','${p_plot}'):
            value=value.replace(prefix,'')
        filename=value.rsplit('/',1)[-1]
        config[key]=('${p_data}' if key in DATA_KEYS else '${p_model}')+filename
    return config


def artifact_kind(name):
    path=Path(name);suffix=path.suffix.lower()
    if suffix in {'.pdf','.png','.svg','.jpg','.jpeg','.eps'}:return 'plots'
    if suffix in {'.pt','.pth','.ckpt','.joblib','.pkl','.cbm'} or (suffix=='.json' and (path.name.startswith(('best_model','model_')) or path.stem.endswith('_model'))):return 'models'
    if suffix in {'.csv','.tsv','.json','.txt','.npy','.npz','.log'}:return 'data'
    return None


def artifact_file(root,name):
    """Prefer the split layout, retaining read compatibility with flat exports."""
    root=Path(root);kind=artifact_kind(name)
    candidates=[root/kind/name] if kind else []
    candidates += [root/name]
    for folder in ('data','models','plots'):
        candidate=root/folder/name
        if candidate not in candidates:candidates.append(candidate)
    return next((p for p in candidates if p.is_file()),candidates[0])


def migrate_generated(root):
    """Move flat artifacts without overwriting an existing destination."""
    root=Path(root)
    if not root.is_dir():return []
    plan=[]
    for source in sorted(root.iterdir()):
        if not source.is_file():continue
        kind=artifact_kind(source.name)
        if not kind:continue
        target=root/kind/source.name
        if target.exists():
            raise FileExistsError(f'Artifact migration conflict: {source} and {target}; neither was overwritten')
        plan.append((source,target))
    for source,target in plan:
        target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    for kind in ('data','plots','models'):(root/kind).mkdir(exist_ok=True)
    return [(str(source),str(target)) for source,target in plan]


def ensure_artifact_directories(config):
    for key in ('p_data','p_plot','p_model'):
        if key in config:Path(config[key]).mkdir(parents=True,exist_ok=True)
    for key in DATA_KEYS|MODEL_KEYS:
        if key in config:Path(config[key]).parent.mkdir(parents=True,exist_ok=True)
