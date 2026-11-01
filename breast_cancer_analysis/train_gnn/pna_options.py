"""Validated PNA-specific architecture controls; defaults preserve existing models."""


def pna_conv_options(config, hidden):
    def names(key, default, allowed):
        value = config.get(key, default)
        result = value.split(",") if isinstance(value, str) else list(value)
        if not result or len(set(result)) != len(result) or not set(result) <= allowed:
            raise ValueError(f"Invalid {key}: {value}")
        return result

    towers = int(config.get("pna_towers", 1))
    pre = int(config.get("pna_pre_layers", 1))
    post = int(config.get("pna_post_layers", 1))
    if towers < 1 or hidden % towers or pre < 1 or post < 1:
        raise ValueError("PNA requires positive depths and towers dividing hidden_dim")
    return dict(
        aggregators=names("pna_aggregators", "mean,min,max,std",
                          {"sum", "mean", "min", "max", "var", "std"}),
        scalers=names("pna_scalers", "identity,amplification,attenuation",
                      {"identity", "amplification", "attenuation", "linear", "inverse_linear"}),
        towers=towers, pre_layers=pre, post_layers=post,
        divide_input=bool(config.get("pna_divide_input", False)),
    )
