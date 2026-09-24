"""CPU metadata audit of real files. Does not claim GPU execution succeeded."""
import _bootstrap
import argparse
import json
import os
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['base', 'c0'], default='base')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = {'status': 'failed', 'kind': a.kind, 'gpu_model_executed': False, 'checks': {}, 'errors': []}
    root = Path(os.environ.get('DIFFSYNTH_MODEL_BASE_PATH', '/missing'))
    try:
        from simwam.models.wan22.helpers.io import hash_model_file
        from simwam.models.wan22.helpers.loader import WAN22_MODEL_REGISTRY
        expected = {r['model_name']: r['model_hash'] for r in WAN22_MODEL_REGISTRY}
        paths = {'wan_video_vae': str(root / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors')}
        if a.kind == 'base':
            from download_models import validate_layout
            validate_layout(root)
            paths['wan_video_dit'] = sorted(str(p) for p in (root / 'Wan-AI/Wan2.2-TI2V-5B').glob('diffusion_pytorch_model*.safetensors'))
            paths['wan_video_text_encoder'] = str(root / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors')
        for name, path in paths.items():
            actual = hash_model_file(path)
            result['checks'][name] = {'path': path, 'tensor_structure_hash': actual, 'expected': expected[name]}
            if actual != expected[name]: raise ValueError(f'{name}: weights do not match this loader registry')
        if a.kind == 'c0':
            import torch
            path = Path(os.environ.get('SIMWAM_IL_CHECKPOINT', '/missing'))
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            for key in ('mot', 'proprio_encoder'):
                if not isinstance(payload.get(key), dict) or not payload[key]: raise ValueError(f'C0 missing/nonempty state dict required: {key}')
            result['checks']['c0'] = {'path': str(path), 'top_level_keys': sorted(payload)}
        result['status'] = 'passed'
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        result['traceback'] = traceback.format_exc()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))
    if result['errors']: raise SystemExit(1)

if __name__ == '__main__': main()
