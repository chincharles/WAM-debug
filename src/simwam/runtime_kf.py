"""Explicit KF factory; no unknown KF keys flow into the upstream constructor."""
def create_simwam_kf_grpo(*, kf, **kwargs):
    import os
    os.environ["SIMWAM_KF_OFFLINE"] = "1"
    from omegaconf import OmegaConf
    from .runtime_grpo import create_simwam_grpo
    from .models.wan22.simwam_kf import SimWAMKF
    config = OmegaConf.to_container(kf,resolve=True) if OmegaConf.is_config(kf) else dict(kf)
    from .kf.contracts import validate_kf_keys
    validate_kf_keys(config)
    model = create_simwam_grpo(model_class=SimWAMKF,**kwargs)
    model.setup_kf(config)
    return model
