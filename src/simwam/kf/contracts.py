"""Experiment invariants and independent, candidate-addressed random streams."""
import hashlib
import json
from pathlib import Path

UPSTREAM_SHA = '68b426c162827cb7701396895dbb3572d29f3420'
SCHEMA = 1

def stable_seed(base, stream, rollout, token, occurrence=0, candidate=0, step=-1):
    data = json.dumps([base, stream, rollout, token, occurrence, candidate, step], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], 'little') % (2**63 - 1)

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def validate_config(cfg):
    kf = cfg['kf']
    validate_kf_keys(kf)
    if kf['future_frames'] not in (0, 8):
        raise ValueError('P0 supports F_train=0 or 8 only; F=4 is not implemented')
    if cfg['grpo']['sample']['group_size'] not in (4, 8):
        raise ValueError('P0 supports K=4 or 8 only')
    ev = cfg['evaluation']
    if ev['future_frames'] not in (0, 8) or ev['candidate_count'] != 1:
        raise ValueError('Official evaluation requires F_eval=0/8 and K_eval=1 (no oracle)')
    if ev['action_steps'] != 10 or not ev['deterministic_ode']:
        raise ValueError('Unified evaluation requires 10-step deterministic ODE')
    if kf['bc_future_frames'] != 0 or kf['logprob_mode'] != 'upstream_compat':
        raise ValueError('P0 requires F0 BC anchor and upstream_compat log-prob')
    gen = kf['generation']
    if gen['decode_rgb'] or gen['cache_across_rollouts'] or not gen['independent_per_candidate']:
        raise ValueError('P0 requires latent-only independent futures without cross-rollout caching')
    if gen['microbatch_size'] != 1 or gen['num_inference_steps'] < 1:
        raise ValueError('P0 serial future generation requires microbatch_size=1 and positive steps')
    if kf['adapter']['train_during_rl']:
        raise ValueError('Joint adapter RL is P1; main P0 freezes the adapter')
    if cfg['data']['train']['video_size'] != [384, 672]:
        raise ValueError('P0 resolution must be [384,672]')
    if cfg['data']['train']['camera_layout'] != 'front' or cfg['data']['train']['trajectory_mode'] != 'absolute':
        raise ValueError('P0 requires front camera and upstream absolute ego-frame trajectories')
    if cfg['model']['video_dit_config']['action_conditioned']:
        raise ValueError('P0 future generator is not action conditioned')
    if cfg['data']['train']['future_action_horizon'] != 8:
        raise ValueError('Action horizon must remain 8')
    if cfg['gradient_accumulation_steps'] != 1 or cfg['grpo']['train']['rollout_buffer_batches'] != 1:
        raise ValueError('P0 supports accumulation=1 and one rollout buffer batch')
    if cfg['max_steps'] < 1:
        raise ValueError('max_steps must be positive')

def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not r.get('scene_token') or not r.get('log_id') for r in rows):
        raise ValueError(f'Empty/invalid manifest: {path}')
    if len({r['scene_token'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate scene tokens: {path}')
    return rows

def validate_splits(splits):
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            for field in ('scene_token', 'log_id'):
                overlap = {r[field] for r in splits[name]} & {r[field] for r in splits[other]}
                if overlap:
                    raise ValueError(f'{name}/{other} {field} overlap: {sorted(overlap)[:5]}')
    return {name: {'tokens': len(rows), 'logs': len({r['log_id'] for r in rows})} for name, rows in splits.items()}

def noise_plan(tokens, k, base, rollout, steps, *, stream='action', device='cpu', dtype=None):
    import torch
    seen = {}
    plans = []
    for token in tokens:
        occurrence = seen.get(token, 0)
        seen[token] = occurrence + 1
        for candidate in range(k):
            plans.append(torch.stack([torch.randn((8, 3), generator=torch.Generator().manual_seed(
                stable_seed(base, stream, rollout, token, occurrence, candidate, step)))
                for step in range(-1, steps)]))
    return torch.stack(plans).to(device=device, dtype=dtype or torch.float32)


def validate_kf_keys(kf):
    keys = {
        'root': {'enabled','future_frames','generation','adapter','warmup','bc_future_frames','logprob_mode','strict_metric_cache','require_il_checkpoint','require_kf_checkpoint_for_rl'},
        'generation': {'num_inference_steps','microbatch_size','independent_per_candidate','decode_rgb','cache_across_rollouts'},
        'adapter': {'enabled','inner_dim','num_heads','layer_indices','gate_init','dropout','train_during_rl'},
        'warmup': {'future_frames','learning_rate','max_steps'},
    }
    for section,allowed in keys.items():
        value=kf if section=='root' else kf[section]
        if set(value)!=allowed:raise ValueError(f'Unknown/missing KF {section} keys: {set(value)^allowed}')
    if any(not kf[k] for k in ('enabled','strict_metric_cache','require_il_checkpoint','require_kf_checkpoint_for_rl')):
        raise ValueError('P0 requires strict cache and C0/C1 safety invariants')
    if kf['future_frames'] not in (0,8) or kf['warmup']['future_frames']!=8:
        raise ValueError('P0 supports only F0/F8 and F8 warmup')
