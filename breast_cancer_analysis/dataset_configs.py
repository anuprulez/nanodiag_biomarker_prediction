"""Stage-specific configuration: preprocessing -> tuning -> frozen dataset config.

Legacy relative paths use train_gnn as their base; emitted dataset configs use
absolute root paths and keep model-name interpolations for per-model artifacts.
"""
from pathlib import Path
import json
from omegaconf import OmegaConf
from preprocessing_grid import load_preprocessing_config, preprocessing_configs
from seed_inputs import seed_signature
from artifact_layout import LAYOUT, apply_artifact_layout

ROOT=Path(__file__).resolve().parent
PATH_BASE=ROOT/'train_gnn'


def resolve_dataset_folder(folder):
    """Relative folder names always refer to this analysis, independent of cwd."""
    path=Path(folder).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def bind_dataset_paths(config,folder):
    """Bind a dataset snapshot's output paths to the folder containing it."""
    folder=Path(folder).resolve()
    previous=OmegaConf.to_container(config,resolve=False)
    def generated_root(value):
        path=Path(value).expanduser()
        absolute=(path if path.is_absolute() else PATH_BASE/path).resolve()
        if absolute.is_relative_to(folder):return str(absolute)+'/'
        parts=path.parts
        suffix=Path(*parts[parts.index('generated')+1:]) if 'generated' in parts else Path('.')
        return str(folder/'generated'/suffix)+'/'
    generated=generated_root(str(config.get('p_generated_files',str(folder/'generated')+'/')))
    config.p_base=str(folder)+'/'
    config.p_base_files=str(folder/'preprocessed')+'/'
    config.p_generated_files=generated
    for key,value in previous.items():
        if key in {'p_base','p_base_files','p_generated_files'} or not key.startswith('p_') or not isinstance(value,str):continue
        if '${p_base_files}' in value or (Path(value).is_absolute() and 'preprocessed' in Path(value).parts):
            config[key]='${p_base_files}'+value.replace('${p_base_files}','').rsplit('/',1)[-1]
    for values in config.get('model_overrides',{}).values():
        if 'p_generated_files' in values:
            values.p_generated_files=generated_root(str(values.p_generated_files))
    config.artifact_layout=LAYOUT
    return apply_artifact_layout(config)


def tuning_source(config_path, search, args=None):
    source=load_preprocessing_config(config_path,args)
    if source.get('config_scope')=='dataset':
        raise ValueError('Tune from the preprocessing config, not a dataset training config')
    runtime=search.runtime_defaults if 'runtime_defaults' in search else OmegaConf.load(ROOT/'config/optuna.yaml').runtime_defaults
    defaults=OmegaConf.create(OmegaConf.to_container(runtime,resolve=False))
    return OmegaConf.merge(defaults,source)


def downstream_configs(config_path,args=None):
    if args is not None and getattr(args,'folder',None):
        config_path=resolve_dataset_folder(args.folder)/'config.yaml'
    source=load_preprocessing_config(config_path,args)
    if source.get('config_scope')=='dataset':
        # Do not rebuild p_base or change the preprocessing combination of a tuned snapshot.
        for key in ('corr_threshold','knn_neighbours','size_negative'):
            value=getattr(args,key,None) if args is not None else None
            if value is not None and value!=OmegaConf.load(config_path)[key]:
                raise ValueError(f'{key} cannot override a tuned dataset config')
        return [bind_dataset_paths(source,Path(config_path).resolve().parent)]
    if source.get('config_scope')!='preprocessing':
        return preprocessing_configs(source)  # Legacy standalone exported configs.
    configs=[]
    for selected in preprocessing_configs(source):
        path=Path(selected.p_base)/'config.yaml'
        if not path.is_absolute():path=PATH_BASE/path
        if not path.is_file():
            raise FileNotFoundError(f'Dataset config missing: {path.resolve()}. Run the tuning stage first.')
        saved=OmegaConf.load(path)
        if saved.get('config_scope')!='dataset':raise ValueError(f'{path}: expected a tuned dataset config')
        for key in ('corr_threshold','knn_neighbours','size_negative'):
            if saved[key]!=selected[key]:raise ValueError(f'{path}: preprocessing selection differs at {key}')
        configs.append(bind_dataset_paths(saved,path.resolve().parent))
    return configs


def dataset_path(config):
    path=Path(config.p_base)
    return (path if path.is_absolute() else PATH_BASE/path).resolve()/'config.yaml'


def write_tuned_dataset(config,directory):
    """Add one completed model's validation winner to its own dataset snapshot."""
    directory=Path(directory)
    result=json.loads((directory/'best_params.json').read_text())
    best=OmegaConf.load(directory/'best_config.yaml');name=result['model']
    target=dataset_path(config);target.parent.mkdir(parents=True,exist_ok=True)
    seed_path=Path(config.p_seeds)
    current_seed=seed_signature(seed_path if seed_path.is_absolute() else PATH_BASE/seed_path)
    if target.exists():
        saved=OmegaConf.load(target)
        if saved.get('config_scope')!='dataset' or dict(saved.get('seed_provenance',{}))!=current_seed:
            raise ValueError(f'{target}: existing config belongs to different seeds or an unsupported schema')
        for key in ('corr_threshold','knn_neighbours','size_negative','gene_dim'):
            if saved[key]!=config[key]:raise ValueError(f'{target}: dataset settings differ at {key}')
    else:
        saved=OmegaConf.create(OmegaConf.to_container(config,resolve=False))
        saved.pop('dataset_model_overrides',None)
        saved.model_overrides={};saved.model_type=[];saved.tuning_results={}
        saved.config_scope='dataset';saved.seed_provenance=current_seed
        # Bind dataset and input roots, leaving ${model_type} interpolations intact.
        for key in ('p_base','p_raw','tabpfn_model_path','tabicl_model_path'):
            if key in saved and saved[key]:
                path=Path(str(saved[key])).expanduser()
                resolved=(path if path.is_absolute() else PATH_BASE/path).resolve()
                saved[key]=str(resolved)+('/' if key in ('p_base','p_raw') else '')
        # Custom input paths may not be relative to p_raw.
        for key in ('p_arrays','p_mapper','p_platform','p_seeds','p_snp_probes','p_cross_reactive_probes','p_sample_metadata'):
            if key in saved:
                path=Path(str(saved[key])).expanduser()
                saved[key]=str((path if path.is_absolute() else PATH_BASE/path).resolve())
    if name in ('PNA','GCN','GraphSAGE','GATv2','GraphTransformer'):
        keys=(set(result['parameters'])-{'neighbor_fanout'}) | {'SEED','n_epo','neighbors_spread','lr_patience','early_stopping_patience','checkpoint_selection_metric','class_weight_power','batch_size','hidden_dim','grad_clip_norm','learning_rate','weight_decay','dropout','gnn_layers','feature_dropout','embedding_dropout','heads'}
        values={key:OmegaConf.to_container(best[key],resolve=True) if OmegaConf.is_config(best[key]) else best[key] for key in keys if key in best}
    else:
        values={'estimator_params':OmegaConf.to_container(best.estimator_params,resolve=True)}
    saved.model_overrides[name]=values
    if name not in saved.model_type:saved.model_type.append(name)
    if saved.get('best_trained_model') not in saved.model_type:saved.best_trained_model=saved.model_type[0]
    saved.tuning_results[name]={'validation_metric':result.get('validation_metric'),
        'validation_score':result.get('validation_score'),'trial':result.get('trial'),
        'input_fingerprint':result.get('input_fingerprint'),'directory':str(directory.resolve())}
    saved.artifact_layout='split_v1'
    apply_artifact_layout(saved)
    temporary=target.with_suffix('.yaml.part');OmegaConf.save(saved,temporary);temporary.replace(target)
    print(f'{name}: saved tuned settings to {target}',flush=True)
    return target
