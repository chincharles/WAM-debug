"""Load model I/O dependencies only when requested, not for CPU block imports."""
__all__ = ['ModelConfig','hash_model_file','load_state_dict','wan_video_dit_from_diffusers',
           'wan_video_dit_state_dict_converter','wan_video_vae_state_dict_converter']
def __getattr__(name):
    import importlib
    if name not in __all__:raise AttributeError(name)
    module='io' if name in __all__[:3] else 'state_dict_converters'
    return getattr(importlib.import_module(f'{__name__}.{module}'),name)
