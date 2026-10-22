"""Bind per-model AML settings without changing other model families."""
import copy
from omegaconf import OmegaConf


def apply_model_settings(config, model_name):
    current = copy.deepcopy(config)
    name = 'TabICL' if str(model_name).lower() in {'tabicl', 'tabiclv2'} else str(model_name)
    current.model_type = name
    overrides = current.get('model_overrides', {})
    for key, values in overrides.items():
        if str(key).lower() == name.lower():
            current = OmegaConf.merge(current, values)
            current.model_type = name
            break
    return current
