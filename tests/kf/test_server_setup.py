"""CPU checks of download completeness and vendor-package protection (no network)."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/kf' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('name', ['torch', 'torchvision', 'triton', 'nvidia-cublas-cu12', 'pccl-runtime', 'ppu-sdk', 'deepspeed', 'flash_attn'])
def test_installer_rejects_accelerator_replacement(name):
    installer = load('install_ppu')
    with pytest.raises(RuntimeError):
        installer.guard_plan({'install': [{'metadata': {'name': name}}]})

def test_installer_allows_python_dependencies():
    load('install_ppu').guard_plan({'install': [{'metadata': {'name': 'transformers'}}]})

def test_incomplete_dit_download_is_rejected(tmp_path):
    download = load('download_models')
    dit = tmp_path / 'Wan-AI/Wan2.2-TI2V-5B'; dit.mkdir(parents=True)
    (dit / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'diffusion_pytorch_model-00001.safetensors'}}))
    with pytest.raises(ValueError, match='Missing/invalid DiT shard'):
        download.validate_layout(tmp_path)

def test_complete_layout_and_extra_shard(tmp_path):
    download = load('download_models')
    dit = tmp_path / 'Wan-AI/Wan2.2-TI2V-5B'; dit.mkdir(parents=True)
    shard = 'diffusion_pytorch_model-00001.safetensors'
    (dit / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': shard}}))
    (dit / shard).write_bytes(b'fixture-not-a-model')
    common = tmp_path / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors'; common.mkdir(parents=True)
    for n in ['Wan2.2_VAE.safetensors', 'models_t5_umt5-xxl-enc-bf16.safetensors']: (common / n).write_bytes(b'fixture')
    tok = tmp_path / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'; tok.mkdir(parents=True)
    (tok / 'tokenizer_config.json').write_text('{}'); (tok / 'spiece.model').write_bytes(b'fixture')
    download.validate_layout(tmp_path)
    (dit / 'diffusion_pytorch_model-extra.safetensors').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='shard mismatch'): download.validate_layout(tmp_path)

def test_probe_uses_real_rotary_shape():
    import torch
    from simwam.models.wan22.wan_video_dit import precompute_freqs_cis, rope_apply
    x = torch.randn(2, 16, 32)
    freqs = precompute_freqs_cis(8, 16)[:, None, :]
    assert rope_apply(x, freqs, 4).shape == x.shape

def test_offline_verify_rejects_modified_file(tmp_path, monkeypatch):
    import sys
    download = load('download_models')
    model = tmp_path / 'model.bin'; model.write_bytes(b'original')
    lock = {'files': {'model.bin': download.sha256(model)}}
    (tmp_path / 'simwam-models.lock.json').write_text(json.dumps(lock))
    model.write_bytes(b'changed')
    monkeypatch.setattr(download, 'validate_layout', lambda root: None)
    monkeypatch.setattr(sys, 'argv', ['download_models.py', '--root', str(tmp_path), '--verify'])
    with pytest.raises(ValueError, match='Checksum mismatch'): download.main()

def test_official_common_conversion_preserves_weights(tmp_path):
    import torch
    load_file = pytest.importorskip("safetensors.torch").load_file
    download = load('download_models')
    src = tmp_path / 'Wan-AI/Wan2.2-TI2V-5B'; src.mkdir(parents=True)
    expected = {'a': torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)}
    for stem in ('Wan2.2_VAE', 'models_t5_umt5-xxl-enc-bf16'):
        torch.save(expected, src / (stem + '.pth'))
    download.convert_common(tmp_path)
    download.convert_common(tmp_path)  # existing identical files are accepted
    target = tmp_path / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'
    actual = load_file(str(target))
    assert actual['a'].dtype == expected['a'].dtype
    torch.testing.assert_close(actual['a'], expected['a'], rtol=0, atol=0)
    torch.save({'a': expected['a'] + 1}, src / 'Wan2.2_VAE.pth')
    with pytest.raises(ValueError, match='differs'): download.convert_common(tmp_path)

def test_download_sources_are_official_wan():
    specs = load('download_models').SPECS
    assert all(repo.startswith('Wan-AI/') for repo in specs)
    assert 'Wan2.2_VAE.pth' in specs['Wan-AI/Wan2.2-TI2V-5B']
