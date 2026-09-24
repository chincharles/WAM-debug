"""Hydra composition with the released FlowGRPO task as parent."""
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from .contracts import validate_config

def compose_config(stage='rl',overrides=()):
    root=Path(__file__).resolve().parents[3]
    with initialize_config_dir(version_base='1.3',config_dir=str(root/'configs')):
        cfg=compose(config_name='train_grpo',overrides=[f'task=navsim_kf_{"warmup" if stage=="warmup" else "grpo"}',*overrides])
    validate_config(OmegaConf.to_container(cfg,resolve=True))
    return cfg

def dry_run_environment():
    """Only dry-run may resolve absent external resources to visible placeholders."""
    import os
    for key in ('NAVSIM_LOG_PATH','NAVSIM_SENSOR_BLOBS_PATH','NAVSIM_METRIC_CACHE_PATH','NAVSIM_DEVKIT_ROOT'):
        os.environ.setdefault(key,f'/UNSET/{key}')
