"""Bind per-model breast cancer settings without changing other model families."""
import copy
from omegaconf import OmegaConf
from artifact_layout import apply_artifact_layout


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
    return apply_artifact_layout(current)
