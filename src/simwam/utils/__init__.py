from .fs import ensure_dir
__all__ = ["ensure_dir", "save_mp4"]

def save_mp4(*args, **kwargs):
    from .video_io import save_mp4 as implementation
    return implementation(*args, **kwargs)
