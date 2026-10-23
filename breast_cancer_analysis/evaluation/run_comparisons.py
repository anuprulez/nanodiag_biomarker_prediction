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
from preprocessing_grid import preprocessing_configs,add_parameter_arguments,load_preprocessing_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE.parent/'config/config.yaml')
    parser.add_argument('--models',nargs='+',help='Default: best_trained_model in config')
    parser.add_argument('--dry-run',action='store_true')
    add_parameter_arguments(parser)
    args=parser.parse_args(); config_path=args.config.resolve()
    os.chdir(BASE.parent/'train_gnn')
    source=load_preprocessing_config(config_path,args)
    models=args.models or [str(source.best_trained_model)]
    for config in preprocessing_configs(source):
        for model in models:
            model='TabICL' if model.lower() in ('tabicl','tabiclv2') else model
            command=[sys.executable,str(BASE/'compare_predictions.py'),'--data',str(Path(config.p_base).resolve()),
                     '--matrix',str(Path(config.p_arrays).resolve()),'--model',model]
            print(' '.join(command),flush=True)
            if not args.dry_run:subprocess.run(command,check=True)

if __name__=='__main__':main()
