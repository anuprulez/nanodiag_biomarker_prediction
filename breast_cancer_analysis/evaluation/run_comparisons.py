#!/usr/bin/env python3
"""Run prediction comparisons for all parameter combinations in the shared config."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from omegaconf import OmegaConf
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from preprocessing_grid import add_parameter_arguments
from dataset_configs import downstream_configs
from train_gnn.model_settings import apply_model_settings
from training_runs import run_name, comparison_run, archived_generated_root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE.parent/'config/config.yaml')
    parser.add_argument('--models',nargs='+',help='Default: all tuned models, or models recorded in --run')
    parser.add_argument('--run',help='Saved run number or run<number>; default: current generated/')
    parser.add_argument('--dry-run',action='store_true')
    add_parameter_arguments(parser)
    args=parser.parse_args(); config_path=args.config.resolve()
    os.chdir(BASE.parent/'train_gnn')
    for config in downstream_configs(config_path,args):
        folder=Path(config.p_base).resolve()
        run=None
        selected_config=folder/'config.yaml'
        available=list(config.model_type)
        if args.run is not None:
            run, config, available=comparison_run(folder,args.run)
            selected_config=run/'config.yaml'
        models=args.models or available
        for model in models:
            model='TabICL' if model.lower() in ('tabicl','tabiclv2') else model
            if config.get('config_scope')=='dataset' and model not in config.model_type:
                raise ValueError(f'{model} has no tuned settings in {config.p_base}config.yaml')
            if model not in available:
                raise ValueError(f'{model} was not trained in {run or folder}')
            current=apply_model_settings(config,model)
            generated=archived_generated_root(folder,run,current,model) if run else Path(current.p_generated_files).resolve()
            output=folder/'evaluation'/run.name if run else folder/'evaluation'
            command=[sys.executable,str(BASE/'compare_predictions.py'),'--data',str(Path(config.p_base).resolve()),
                     '--config',str(selected_config),
                     '--matrix',str(Path(current.p_arrays).resolve()),
                     '--generated-dir',str(generated),
                     '--output',str(output),'--model',model]
            print(' '.join(command),flush=True)
            if not args.dry_run:subprocess.run(command,check=True)
        if args.dry_run:
            print(f'Cumulative plots from {config.p_base}runs/run*/generated -> {config.p_base}accumulated_results',flush=True)
        else:
            from train_gnn.plot_aggregations import plot_cumulative_runs
            plot_cumulative_runs(config, models)

if __name__=='__main__':main()
