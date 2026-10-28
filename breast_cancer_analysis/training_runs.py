"""Immutable snapshots of successful pipeline training invocations."""
import json
from pathlib import Path
import re
import shutil
import tempfile
from omegaconf import OmegaConf


def completed_runs(folder):
    return sorted((p for p in (Path(folder)/'runs').glob('run*')
                   if p.is_dir() and re.fullmatch(r'run[1-9][0-9]*', p.name)
                   and (p/'run.json').is_file()), key=lambda p: int(p.name[3:]))


def archive_training_run(folder, config, log, models=None):
    dataset = Path(folder)
    folder = dataset/'runs'
    folder.mkdir(exist_ok=True)
    selected = list(models or config.model_type)
    # A lock keeps simultaneous archive publication from claiming the same number.
    import fcntl
    with (folder/'.training_runs.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        numbers = [int(p.name[3:]) for p in folder.glob('run*')
                   if re.fullmatch(r'run[1-9][0-9]*', p.name)]
        target = folder/f'run{max(numbers, default=0)+1}'
        temporary = Path(tempfile.mkdtemp(prefix='.training-run-', dir=folder))
        try:
            shutil.copytree(dataset/'generated', temporary/'generated')
            from train_gnn.model_settings import apply_model_settings
            root = (dataset/'generated').resolve()
            for model in selected:
                model_root = Path(apply_model_settings(config, model).p_generated_files).resolve()
                if not model_root.is_relative_to(root):
                    shutil.copytree(model_root, temporary/'generated'/'custom'/model)
            (temporary/'logs').mkdir()
            shutil.copy2(log, temporary/'logs/train.log')
            OmegaConf.save(config, temporary/'config.yaml')
            (temporary/'run.json').write_text(json.dumps({'models': selected}, indent=2)+'\n')
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return target


def run_name(value):
    """Accept a positive run number or run<number>, never a filesystem path."""
    text = str(value)
    if text.isdigit():
        text = 'run'+text
    if not re.fullmatch(r'run[1-9][0-9]*', text):
        raise ValueError('Run must be a positive number or run<number>, e.g. 2 or run2')
    return text


def comparison_run(folder, value):
    """Load the archived settings and manifest without rebinding output roots."""
    folder = Path(folder).resolve()
    run = folder/'runs'/run_name(value)
    if not (run/'run.json').is_file() or not (run/'config.yaml').is_file():
        raise FileNotFoundError(f'Completed training run not found: {run}')
    config = OmegaConf.load(run/'config.yaml')
    config.p_base = str(folder)+'/'
    manifest = json.loads((run/'run.json').read_text())
    return run, config, manifest['models']


def archived_generated_root(folder, run, current, model):
    """Mirror archive_training_run's standard and custom-root storage rules."""
    original = Path(current.p_generated_files).resolve()
    generated = Path(folder).resolve()/'generated'
    if original.is_relative_to(generated):
        return run/'generated'/original.relative_to(generated)
    return run/'generated'/'custom'/model
