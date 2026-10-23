"""Validate shared raw breast cancer inputs and create parameter-specific output folders."""
from pathlib import Path
from preprocessing_grid import prepare_run_directories

INPUT_KEYS = (
    "p_arrays", "p_platform", "p_snp_probes", "p_cross_reactive_probes",
    "p_mapper", "p_seeds",
)


def prepare_preprocessing_inputs(config):
    prepare_run_directories(config)
    missing = [f"{key}: {Path(config[key]).resolve()}"
               for key in INPUT_KEYS if not Path(config[key]).is_file()]
    if missing:
        raise FileNotFoundError("Missing breast cancer raw inputs:\n" + "\n".join(missing))
    print(f"breast cancer preprocessing directory: {Path(config.p_base).resolve()}", flush=True)
