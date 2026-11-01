#!/usr/bin/env python3
"""Run stages across dataset folders, or select one with --folder."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
from training_runs import archive_training_run, run_name, comparison_run
import sys
from omegaconf import OmegaConf
from preprocessing_grid import preprocessing_configs
from artifact_layout import LAYOUT,apply_artifact_layout,migrate_generated
from dataset_configs import resolve_dataset_folder,bind_dataset_paths
from stage_reset import reset_stage_outputs

HERE=Path(__file__).resolve().parent
PATTERN=re.compile(r'^data_corr_([0-9.eE+-]+)_knn_n_(\d+)_n_neg_(\d+)$')
STAGES=('preprocess','tune','train','xai','compare-predictions','accumulate','post-evaluation','niapu-rankings')
SCRIPTS={'preprocess':'preprocess_raw_data/process_raw_illumina_arrays_bc.py',
         'tune':'train_gnn/tune_models.py','train':'train_gnn/main.py',
         'accumulate':'evaluation/accumulate_runs.py','xai':'train_gnn/run_xai_explainer.py','compare-predictions':'evaluation/run_comparisons.py',
         'post-evaluation':'evaluation/post_evaluation.py','niapu-rankings':'evaluation/niapu_rankings.py'}


def absolute(path):
    path=Path(path).expanduser()
    return (path if path.is_absolute() else HERE/'train_gnn'/path).resolve()


def selected_folders(source,folder=None,include_configured=False):
    if folder is not None:
        return [resolve_dataset_folder(folder)]
    if source.get('config_scope')=='dataset':return [absolute(source.p_base)]
    root=absolute(source.get('p_dataset_root','..'))
    folders={p.resolve() for p in root.glob('data_corr_*_knn_n_*_n_neg_*') if p.is_dir() and PATTERN.fullmatch(p.name)}
    # Include non-standard dataset names that carry a dataset config.
    for p in root.glob('*/config.yaml'):
        if p.parent==HERE/'config':continue
        if OmegaConf.load(p).get('config_scope')=='dataset':folders.add(p.parent.resolve())
    if include_configured:
        folders.update(absolute(c.p_base) for c in preprocessing_configs(source))
    if not folders:raise FileNotFoundError(f'No dataset folders under {root}; run --stage preprocess first')
    return sorted(folders)


def preprocessing_for_folder(source,folder):
    saved=folder/'preprocessing_config.yaml'
    if saved.is_file():config=OmegaConf.load(saved)
    else:
        config=OmegaConf.create(OmegaConf.to_container(source,resolve=False))
        snapshot=folder/'config.yaml'
        if snapshot.is_file():
            existing=OmegaConf.load(snapshot)
            # Reuse the dataset's frozen inputs and preprocessing context.
            for key in list(config):
                if key in existing:config[key]=OmegaConf.to_container(existing[key],resolve=False) if OmegaConf.is_config(existing[key]) else existing[key]
        match=PATTERN.fullmatch(folder.name)
        if match:
            config.corr_threshold=float(match[1]);config.knn_neighbours=int(match[2]);config.size_negative=int(match[3])
        else:
            runs=preprocessing_configs(config)
            if len(runs)!=1:raise ValueError(f'{folder}: cannot infer preprocessing parameters; provide a dataset config or a standard folder name')
            config=runs[0]
    config.config_scope='dataset_preprocessing';config.p_base=str(folder)+'/'
    return config


def prepare_dataset(folder,dry_run=False):
    path=folder/'config.yaml'
    if not path.is_file():raise FileNotFoundError(f'{path} missing; run --stage tune --folder {folder} first')
    config=OmegaConf.load(path)
    if config.get('config_scope')!='dataset':raise ValueError(f'{path}: expected a tuned dataset config')
    bind_dataset_paths(config,folder)
    if not dry_run:
        roots={folder/'generated'}
        for values in config.get('model_overrides',{}).values():
            if 'p_generated_files' in values:roots.add(absolute(values.p_generated_files))
        moved=[]
        for root in roots:moved.extend(migrate_generated(root))
        if moved:
            report=folder/'artifact_migration.json'
            history=json.loads(report.read_text()) if report.exists() else []
            report.write_text(json.dumps(history+moved,indent=2)+'\n')
            print(f'{folder.name}: organized {len(moved)} existing artifacts',flush=True)
        for kind in ('data','plots','models'):(folder/'generated'/kind).mkdir(parents=True,exist_ok=True)
        backup=path.with_name('config.before_artifact_layout.yaml')
        if not backup.exists():backup.write_bytes(path.read_bytes())
        temporary=path.with_suffix('.yaml.part');OmegaConf.save(config,temporary);temporary.replace(path)
    return config


def first_likely_positive_node(folder,config):
    from train_gnn.model_settings import apply_model_settings
    from evaluation.extract_predictions import extract
    current=apply_model_settings(config,'PNA')
    path=Path(current.p_pred_likely_pos)
    if not path.is_file():
        # Training saves prediction tensors; groups can be reconstructed before XAI.
        extract(folder,'PNA',Path(current.p_data),generated_dir=current.p_generated_files)
    with path.open(newline='') as source:
        header=source.readline();source.seek(0)
        reader=csv.DictReader(source,delimiter='\t' if '\t' in header else ',');row=next(reader,None)
        if row is None:raise ValueError(f'{path}: no likely-positive test nodes available for XAI')
        if 'test_gene_ids' not in row:raise ValueError(f'{path}: missing test_gene_ids column')
        value=int(row['test_gene_ids'])
        if value<0:raise ValueError(f'{path}: invalid negative node ID')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    from train_gnn.xai_selection import add_xai_selection_arguments
    add_xai_selection_arguments(parser)
    parser.add_argument('--node-id', type=int, help='Graph node ID for --stage xai; overrides top prediction selection')
    parser.add_argument('--stage',choices=['all',*STAGES,'naipu-rankings','compare-preprocess'],default='all')
    parser.add_argument('--folder',type=Path,help='One dataset directory; default: all dataset folders')
    parser.add_argument('--folders',nargs='+',type=Path,help='Dataset folders for cross-folder accuracy heatmaps')
    parser.add_argument('--config',type=Path,default=HERE/'config/config.yaml')
    parser.add_argument('--models',nargs='+',help='Models for tuning, training or prediction comparison')
    parser.add_argument('--run',help='Archived run for prediction comparison or NIAPU UMAP; UMAP default: run1')
    parser.add_argument('--trials',type=int)
    parser.add_argument('--timeout',type=float)
    parser.add_argument('--study-tag')
    parser.add_argument('--search-space',type=Path)
    parser.add_argument('--random-panels', type=int, default=30, help='Matched backgrounds for post-evaluation')
    parser.add_argument('--background-size', type=int, default=800, help='Sampled CpG background for post-evaluation')
    parser.add_argument('--pathway-gmt', type=Path, help='Local Reactome-compatible gene-set GMT for post-evaluation')
    parser.add_argument('--offline', action='store_true', help='Post-evaluation: use cached resources only, including expression')
    sys.path.insert(0,str(HERE.parent))
    from post_evaluation_support.cross_cohort_pipeline import add_arguments
    add_arguments(parser)
    parser.add_argument('--dry-run',action='store_true',help='Print commands without changing configs, files or running stages')
    umap_options=parser.add_mutually_exclusive_group()
    umap_options.add_argument('--umap-only',action='store_true',help='NIAPU stage: create four UMAP panels without patient classifiers')
    umap_options.add_argument('--skip-umap',action='store_true',help='NIAPU stage: run patient classifiers without requiring a trained PNA run')
    args=parser.parse_args()
    if args.stage=='naipu-rankings':args.stage='niapu-rankings'
    if (args.umap_only or args.skip_umap) and args.stage!='niapu-rankings':parser.error('--umap-only/--skip-umap require --stage niapu-rankings')
    if args.node_id is not None and args.node_id < 0:parser.error('--node-id must be nonnegative')
    if args.top_k < 1:parser.error('--top-k must be at least 1')
    if args.folders and args.stage != 'compare-preprocess':
        parser.error('--folders is supported only with --stage compare-preprocess')
    if args.folders and args.folder:
        parser.error('Choose --folder or --folders')
    if args.stage == 'compare-preprocess':
        if args.run is not None:parser.error('compare-preprocess always uses run1; omit --run')
        comparison_folders = args.folders or ([args.folder] if args.folder else sorted(HERE.glob('data_corr_*_knn_n_*_n_neg_*')))
        command = [sys.executable, '-u', str(HERE/'evaluation/compare_heatmaps.py'),
                   '--folders', *[str(folder) for folder in comparison_folders]]
        if args.models: command += ['--models', *args.models]
        print(' '.join(command), flush=True)
        if not args.dry_run: subprocess.run(command, cwd=HERE, check=True)
        return
    source=OmegaConf.load(args.config.resolve())
    if args.run is not None:
        if args.stage not in ('compare-predictions','niapu-rankings'):parser.error('--run is only supported with --stage compare-predictions or niapu-rankings')
        try:args.run=run_name(args.run)
        except ValueError as error:parser.error(str(error))
    folders=selected_folders(source,args.folder,args.stage in ('preprocess','all'))
    if args.stage=='compare-predictions' and args.folder is None and source.get('config_scope')!='dataset':
        ready=[]
        for folder in folders:
            if (folder/'config.yaml').is_file():ready.append(folder)
            else:print(f'Skipping untuned dataset for prediction comparison: {folder}',flush=True)
        folders=ready
        if not folders:raise FileNotFoundError('No tuned dataset configs available for prediction comparison')
    if args.stage=='post-evaluation' and args.folder is None and source.get('config_scope')!='dataset':
        folders=[folder for folder in folders if (folder/'config.yaml').is_file() and
                 list((folder/'accumulated_results/evaluation').glob('likely_positive_common_unique_cpgs_PNA_*.tsv'))]
        if not folders:raise FileNotFoundError('No five-run PNA evaluation available; run compare-predictions/accumulate first')
    if args.stage=='niapu-rankings' and args.folder is None:
        folders=[folder for folder in folders if (folder/'preprocessed/out_gene_rankings_bc.csv').is_file()
                 and ((folder/'config.yaml').is_file() or (folder/'preprocessing_config.yaml').is_file())]
        if not folders:raise FileNotFoundError('No NIAPU rankings available; run --stage preprocess first')
    fresh=args.folder is None and source.get('config_scope')!='dataset'
    stages=tuple(s for s in STAGES if s not in ('accumulate','post-evaluation','niapu-rankings')) if args.stage=='all' else (args.stage,)
    env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg')
    for folder in folders:
        for stage in stages:
            # Clear this stage's log before config validation, cleanup or XAI lookup.
            if not args.dry_run:
                logs=folder/'logs';logs.mkdir(parents=True,exist_ok=True)
                (logs/f'{stage}.log').write_text('')
            extra=[]
            if stage in ('preprocess','tune'):
                current=preprocessing_for_folder(source,folder)
                config_path=folder/'preprocessing_config.yaml'
                if stage=='preprocess' and not fresh and Path(current.p_preprocessing_manifest).is_file():
                    print(f'Skipping completed preprocessing: {folder}',flush=True);continue
                if not args.dry_run:
                    folder.mkdir(parents=True,exist_ok=True)
                    if fresh:reset_stage_outputs(folder,stage,current,args.models)
                    temporary=config_path.with_suffix('.yaml.part');OmegaConf.save(current,temporary);temporary.replace(config_path)
                if stage=='tune':
                    if fresh:extra.append('--fresh')
                    for option,value in [('--study-tag',args.study_tag),('--search-space',args.search_space.resolve() if args.search_space else None),('--trials',args.trials),('--timeout',args.timeout)]:
                        if value is not None:extra.extend([option,str(value)])
                    if args.models:extra+=['--models',*args.models]
            elif stage=='niapu-rankings':
                config_path=folder/'config.yaml'
                if not config_path.is_file():config_path=folder/'preprocessing_config.yaml'
                if not config_path.is_file():raise FileNotFoundError(f'{config_path} missing; run preprocessing first')
                if args.models:raise ValueError('niapu-rankings uses its fixed four patient classifiers; omit --models')
                if args.run:extra+=['--run',args.run]
                if args.umap_only:extra.append('--umap-only')
                if args.skip_umap:extra.append('--skip-umap')
            elif stage=='post-evaluation':
                config_path=folder/'config.yaml'
                if not config_path.is_file():raise FileNotFoundError(f'{config_path} missing')
                extra += ['--models', *(args.models or ['PNA']), '--random-panels', str(args.random_panels),
                          '--background-size', str(args.background_size)]
                if args.pathway_gmt:extra += ['--pathway-gmt', str(args.pathway_gmt.resolve())]
                if args.offline:extra.append('--offline')
                extra += ['--post-evaluation-analysis',args.post_evaluation_analysis]
                if args.rscript:extra += ['--rscript',args.rscript]
                if args.refresh_expression:extra.append('--refresh-expression')
            else:
                current=prepare_dataset(folder,True);config_path=folder/'config.yaml'
                if stage=='compare-predictions' and args.run:
                    comparison_run(folder,args.run)  # Validate before resetting outputs.
                    extra+=['--run',args.run]
                if fresh and not args.dry_run and not args.run:reset_stage_outputs(folder,stage,current,args.models)
                if not args.dry_run:current=prepare_dataset(folder)
                if stage=='train' and args.models:extra+=['--models',*args.models]
                if stage=='xai':
                    extra+=['--model','PNA','--xai-group',args.xai_group,'--top-k',str(args.top_k)]
                    if args.node_id is not None:extra+=['--node-id',str(args.node_id)]
                if stage in ('compare-predictions','accumulate') and args.models:extra+=['--models',*args.models]
                if stage=='compare-predictions' and not args.dry_run:(folder/'evaluation').mkdir(exist_ok=True)
            command=[sys.executable,'-u',str(HERE/SCRIPTS[stage]),'--config',str(config_path),*extra]
            print(' '.join(command),flush=True)
            if args.dry_run:continue
            logs=folder/'logs';logs.mkdir(exist_ok=True);log=logs/f'{stage}.log'
            with log.open('w') as output:
                result=subprocess.run(command,cwd=HERE,env=env,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:raise SystemExit(f'{stage} failed for {folder.name}; see {log}')
            if stage=='train':
                run=archive_training_run(folder,current,log,args.models)
                print(f'{folder.name}: saved training run: {run}',flush=True)
            if stage=='preprocess':
                subprocess.run([sys.executable,str(HERE/'preprocess_raw_data/validate_bc_outputs.py'),'--config',str(config_path)],cwd=HERE,env=env,check=True)
            print(f'{folder.name}: {stage} complete; log: {log}',flush=True)



if __name__=='__main__':main()
