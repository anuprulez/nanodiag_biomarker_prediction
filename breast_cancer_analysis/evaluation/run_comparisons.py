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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE.parent/'config/config.yaml')
    parser.add_argument('--models',nargs='+',help='Default: best_trained_model in config')
    parser.add_argument('--dry-run',action='store_true')
    add_parameter_arguments(parser)
    args=parser.parse_args(); config_path=args.config.resolve()
    os.chdir(BASE.parent/'train_gnn')
    for config in downstream_configs(config_path,args):
        models=args.models or [str(config.best_trained_model)]
        for model in models:
            model='TabICL' if model.lower() in ('tabicl','tabiclv2') else model
            if config.get('config_scope')=='dataset' and model not in config.model_type:
                raise ValueError(f'{model} has no tuned settings in {config.p_base}config.yaml')
            current=apply_model_settings(config,model)
            command=[sys.executable,str(BASE/'compare_predictions.py'),'--data',str(Path(config.p_base).resolve()),
                     '--config',str(Path(config.p_base)/'config.yaml'),
                     '--matrix',str(Path(current.p_arrays).resolve()),
                     '--generated-dir',str(Path(current.p_generated_files).resolve()),
                     '--output',str(Path(current.p_base)/'evaluation'),'--model',model]
            print(' '.join(command),flush=True)
            if not args.dry_run:subprocess.run(command,check=True)

if __name__=='__main__':main()
