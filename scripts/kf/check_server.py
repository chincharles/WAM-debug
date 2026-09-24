"""Real imports and optional synthetic accelerator kernels, without model weights."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import timedelta

MODULES = ['torch', 'torchvision', 'numpy', 'scipy', 'pandas', 'hydra',
           'accelerate', 'transformers', 'safetensors', 'av',
           'simwam.models.wan22.simwam_kf', 'simwam.datasets.navsim.kf_validation',
           'navsim.planning.script.run_metric_caching', 'navsim.evaluate.pdm_score']

def probe(device):
    import torch
    from simwam.models.wan22.wan_video_dit import flash_attention, rope_apply, precompute_freqs_cis
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(42)
    q = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(16, 16, device=device, dtype=torch.bool).tril()
    loss = checkpoint(lambda x: flash_attention(x, x, x, 4, mask).float().square().mean(), q, use_reentrant=False)
    loss.backward()
    if not torch.isfinite(q.grad).all(): raise ValueError('Nonfinite SDPA/checkpoint gradients')
    # Complex rotary positional embeddings are used by the real video/action DiT.
    freqs = precompute_freqs_cis(8, 16).to(device)[:, None, :]
    rotated = rope_apply(q, freqs, 4)
    if not torch.isfinite(rotated).all(): raise ValueError('Nonfinite RoPE')
    conv = torch.nn.Conv3d(4, 4, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(conv.parameters(), lr=1e-3, foreach=False)
    x = torch.randn(1, 4, 3, 8, 8, device=device, dtype=torch.bfloat16)
    conv(x).float().square().mean().backward()
    if not all(torch.isfinite(p.grad).all() for p in conv.parameters()): raise ValueError('Nonfinite Conv3D gradient')
    opt.step()
    state = torch.cuda.get_rng_state(device)
    expected = torch.randn(8, device=device)
    torch.cuda.set_rng_state(state, device)
    torch.testing.assert_close(expected, torch.randn(8, device=device), rtol=0, atol=0)
    torch.cuda.synchronize()
    return ['bf16_masked_sdpa_backward', 'checkpoint_backward', 'complex_rope', 'conv3d_adamw', 'device_rng_restore']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--kernels', action='store_true')
    a = p.parse_args()
    result = {'status': 'failed', 'real_models_tested': False, 'real_data_tested': False,
              'python': sys.version, 'runtime': os.environ.get('SIMWAM_RUNTIME', 'cuda')}
    error = None
    try:
        result['imports'] = {}
        for name in MODULES:
            result['checking_module'] = name
            result['imports'][name] = str(importlib.import_module(name).__file__)
        result.pop('checking_module', None)
        import torch
        import torchvision
        result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                      nuplan=metadata.version('nuplan-devkit'), devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        if result['runtime'] == 'ppu':
            if sys.version_info[:2] != (3, 12) or torch.__version__.split('+')[0] != '2.6.0' or torchvision.__version__.split('+')[0] != '0.21.0':
                raise ValueError('PPU requires vendor Python 3.12 / torch 2.6.0 / torchvision 0.21.0')
            if result['nuplan'] != '1.2.0': raise ValueError('PPU profile requires pinned nuPlan 1.2.0')
            if not result['devices'] or any('PPU' not in d.upper() for d in result['devices']): raise ValueError('Expected vendor PPU device names')
        if a.kernels:
            if not torch.cuda.is_available(): raise ValueError('No CUDA-compatible accelerator')
            rank = int(os.environ.get('RANK', 0)); local = int(os.environ.get('LOCAL_RANK', 0))
            world = int(os.environ.get('WORLD_SIZE', 1))
            torch.cuda.set_device(local)
            if world > 1: torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=3))
            result['kernels'] = probe(f'cuda:{local}')
            if world > 1:
                value = torch.tensor([rank + 1.], device=f'cuda:{local}')
                torch.distributed.all_reduce(value)
                torch.testing.assert_close(value, torch.full_like(value, world * (world + 1) / 2))
                result['collective_world_size'] = world
        result['status'] = 'passed'
    except Exception as exc:
        error = exc; result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
    finally:
        out = Path(a.output)
        if int(os.environ.get('WORLD_SIZE', 1)) > 1: out = out.with_name(f'{out.stem}.rank{os.environ["RANK"]}{out.suffix}')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + '\n')
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.distributed.is_initialized(): torch_module.distributed.destroy_process_group()
    print(json.dumps(result, indent=2))
    if error: raise SystemExit(1)

if __name__ == '__main__': main()
