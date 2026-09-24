"""Explicit, resumable HF downloads; training entry points stay offline.

A first online run resolves repository revisions and records them before downloads.
Re-runs reuse that lock. --verify is offline and checks every recorded SHA256.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

SPECS = {
    'Wan-AI/Wan2.2-TI2V-5B': ['diffusion_pytorch_model*.safetensors', 'diffusion_pytorch_model.safetensors.index.json', 'config.json', 'Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth'],
    'Wan-AI/Wan2.1-T2V-1.3B': ['google/umt5-xxl/*'],
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def convert_common(root):
    """Lossless CPU format conversion; preserve all tensor names/dtypes/values.

    Existing outputs must match the downloaded source; never replace different weights.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    source_dir = root / 'Wan-AI/Wan2.2-TI2V-5B'
    target_dir = root / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors'
    target_dir.mkdir(parents=True, exist_ok=True)
    for stem in ('Wan2.2_VAE', 'models_t5_umt5-xxl-enc-bf16'):
        source = source_dir / (stem + '.pth')
        target = target_dir / (stem + '.safetensors')
        print(f'CPU format conversion: {source} -> {target}', flush=True)
        # weights_only avoids executing arbitrary pickle objects. Older non-zip .pth
        # files cannot use mmap, so use the standard CPU loader for those files.
        import zipfile
        state = torch.load(source, map_location='cpu', weights_only=True,
                           mmap=True if zipfile.is_zipfile(source) else False)
        if isinstance(state, dict) and len(state) == 1:
            for key in ('state_dict', 'module', 'model_state'):
                if key in state: state = state[key]; break
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError(f'Expected flat tensor state dict: {source}')
        if target.exists():
            with safe_open(target, framework='pt', device='cpu') as existing:
                if set(existing.keys()) != set(state): raise ValueError(f'Existing converted file differs: {target}')
                for key, tensor in state.items():
                    saved = existing.get_tensor(key)
                    if saved.dtype != tensor.dtype or saved.shape != tensor.shape or not torch.equal(saved, tensor):
                        raise ValueError(f'Existing converted tensor differs: {target}: {key}')
        else:
            # Separate shared storage without changing tensor values.
            converted = {k: v.contiguous().clone() for k, v in state.items()}
            temporary = target.with_suffix('.safetensors.tmp')
            save_file(converted, str(temporary))
            temporary.replace(target)
            del converted
        del state


def validate_layout(root):
    dit = root / 'Wan-AI/Wan2.2-TI2V-5B'
    index = json.loads((dit / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards: raise ValueError('Empty DiT index')
    for name in shards:
        if Path(name).name != name or not (dit / name).is_file():
            raise ValueError(f'Missing/invalid DiT shard: {name}')
    actual = {p.name for p in dit.glob('diffusion_pytorch_model*.safetensors')}
    if actual != shards: raise ValueError(f'DiT shard mismatch: expected {shards}, got {actual}')
    common = root / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors'
    tok = root / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'
    for p in [common / 'Wan2.2_VAE.safetensors', common / 'models_t5_umt5-xxl-enc-bf16.safetensors', tok / 'tokenizer_config.json']:
        if not p.is_file() or p.stat().st_size == 0: raise ValueError(f'Missing/empty file: {p}')
    if not any((tok / n).is_file() for n in ('tokenizer.json', 'spiece.model')):
        raise ValueError('Missing tokenizer vocabulary')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=os.environ.get('DIFFSYNTH_MODEL_BASE_PATH'))
    p.add_argument('--plan', action='store_true', help='Print paths; no network or writes')
    p.add_argument('--verify', action='store_true', help='Verify recorded hashes offline')
    a = p.parse_args()
    if not a.root or '/path/to' in a.root: p.error('Set --root to an actual model directory')
    root = Path(a.root).expanduser().resolve()
    if a.plan:
        print(json.dumps({'root': str(root), 'repositories': SPECS}, indent=2)); return
    lock_path = root / 'simwam-models.lock.json'
    if a.verify:
        lock = json.loads(lock_path.read_text())
        if not lock.get('files'): raise ValueError('Download has not completed')
        validate_layout(root)
        for name, digest in lock['files'].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha256(path) != digest:
                raise ValueError(f'Checksum mismatch: {name}')
        print('All model file checksums verified; real model loading still required.'); return
    # The download command explicitly overrides the offline training environment.
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import HfApi, snapshot_download
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock['specs'] != SPECS: raise ValueError('Download lock specification differs. Preserve/rename simwam-models.lock.json and rerun; downloaded files can be reused.')
    else:
        api = HfApi()
        lock = {'specs': SPECS, 'revisions': {repo: api.model_info(repo).sha for repo in SPECS}}
        root.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    for repo, patterns in SPECS.items():
        snapshot_download(repo_id=repo, revision=lock['revisions'][repo],
                          allow_patterns=patterns, local_dir=str(root / repo), max_workers=4)
    convert_common(root)
    validate_layout(root)
    files = []
    for repo in [*SPECS, 'DiffSynth-Studio/Wan-Series-Converted-Safetensors']:
        files += [p for p in (root / repo).rglob('*') if p.is_file() and '.cache' not in p.parts]
    lock['files'] = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
    lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    print(f'Download complete: {lock_path}. Run --verify, then real-model smoke.')

if __name__ == '__main__': main()
