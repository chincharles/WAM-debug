import _bootstrap
import argparse
import importlib.util
import json
import os
from pathlib import Path
from simwam.kf.contracts import read_manifest,validate_splits,file_hash,UPSTREAM_SHA
from simwam.kf.execution import json_write

def preflight(stage,split=None):
    errors=[];checks={};splits={}
    if Path(os.environ.get('NAVSIM_DEVKIT_ROOT','/missing')).resolve()!=(_bootstrap.ROOT/'navsim').resolve():
        errors.append('NAVSIM_DEVKIT_ROOT must point to this checkout vendored navsim directory')
    names=['train','val'] if split is None else list(dict.fromkeys(['train','val',split]))
    keys=['SIMWAM_IL_CHECKPOINT','NAVSIM_LOG_PATH','NAVSIM_SENSOR_BLOBS_PATH','NAVSIM_METRIC_CACHE_PATH',
        'NAVSIM_VAL_METRIC_CACHE_PATH','NAVSIM_TEXT_EMBED_CACHE','NAVSIM_STATS_PATH','NUPLAN_MAPS_ROOT','DIFFSYNTH_MODEL_BASE_PATH']
    if stage=='rl':keys.append('SIMWAM_KF_CHECKPOINT')
    if split=='test':keys+=['NAVSIM_TEST_LOG_PATH','NAVSIM_TEST_SENSOR_BLOBS_PATH','NAVSIM_TEST_METRIC_CACHE_PATH']
    keys += [f'KF_{s.upper()}_MANIFEST' for s in names]
    for key in keys:
        path=os.environ.get(key)
        if not path or not Path(path).exists():errors.append(f'{key}: required path missing: {path or "<unset>"}')
        else:checks[key]=str(Path(path).resolve())
    if os.environ.get('NUPLAN_MAP_VERSION')!='nuplan-maps-v1.0':errors.append('NUPLAN_MAP_VERSION must be nuplan-maps-v1.0')
    for name in ('torch','torchvision','transformers','accelerate','hydra','navsim','nuplan','safetensors','av'):
        if importlib.util.find_spec(name) is None:errors.append(f'Missing dependency: {name}')
    for s in names:
        try:splits[s]=read_manifest(os.environ[f'KF_{s.upper()}_MANIFEST'])
        except Exception as exc:errors.append(f'{s} manifest: {exc}')
    try:checks['splits']=validate_splits(splits)
    except Exception as exc:errors.append(str(exc))
    vae=Path(os.environ.get('DIFFSYNTH_MODEL_BASE_PATH','/missing'))/'DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'
    if not vae.is_file():errors.append(f'Required VAE: {vae}; C0 supplies video/action DiT, cached text supplies T5')
    if errors:return {'status':'failed','errors':errors,'checks':checks}
    try:
        import torch
        json.loads(Path(os.environ['NAVSIM_STATS_PATH']).read_text())
        checks['normalization']='upstream fixed odo bounds; supplied statistics file is provenance only'
        for key in ['SIMWAM_IL_CHECKPOINT']+(['SIMWAM_KF_CHECKPOINT'] if stage=='rl' else []):
            payload=torch.load(os.environ[key],map_location='cpu',weights_only=False,mmap=True)
            if 'mot' not in payload or 'proprio_encoder' not in payload:raise ValueError(f'{key}: missing mot/proprio weights')
            if key=='SIMWAM_KF_CHECKPOINT':
                m=payload['kf_metadata']
                if m['upstream_sha']!=UPSTREAM_SHA:raise ValueError('C1 upstream mismatch')
                if m['il_hash']!=file_hash(os.environ['SIMWAM_IL_CHECKPOINT']):raise ValueError('C1 did not originate from configured C0')
                if m['stats_hash']!=file_hash(os.environ['NAVSIM_STATS_PATH']):raise ValueError('C1 statistics provenance mismatch')
                for name in ('train','val'):
                    if m['split_hashes'][name]!=file_hash(os.environ[f'KF_{name.upper()}_MANIFEST']):raise ValueError(f'C1 {name} manifest mismatch')
            checks[key+'_sha256']=file_hash(os.environ[key]);del payload
        checks['vae_sha256']=file_hash(vae)
        from simwam.kf.config import compose_config
        from simwam.kf.execution import make_dataset,make_reward
        cfg=compose_config('warmup' if stage=='warmup' else 'rl')
        from omegaconf import OmegaConf
        source_root=Path(os.environ['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/config/common/train_test_split/scene_filter'
        official={s:set(OmegaConf.load(source_root/f'{s}.yaml').tokens) for s in ('navtrain','navtest')}
        for s,rows in splits.items():
            allowed=official['navtest' if s=='test' else 'navtrain']
            extra={r['scene_token'] for r in rows}-allowed
            if extra:raise ValueError(f'{s} tokens outside official source: {sorted(extra)[:5]}')
            ds=make_dataset(cfg,s);reward=make_reward(cfg,s,ds)
            for i,token in enumerate(ds.tokens):
                frames=ds.scene_loader.scene_frames_dicts[token]
                current=frames[ds.scene_filter.num_history_frames-1]
                if current['log_name']!=ds.allowed[token]:raise ValueError(f'{token}: manifest log_id mismatch')
                # Actually decode the current image and read text tensor; never load future sensor images.
                sample=ds[i]
                if sample['video'].shape!=(3,1,384,672):raise ValueError(f'{token}: image shape mismatch')
                if sample['context'].shape!=(256,4096):raise ValueError(f'{token}: expected text embedding [256,4096]')
                if not torch.isfinite(sample['context']).all():raise ValueError(f'{token}: nonfinite text cache')
                reward.metric_cache_loader.get_from_token(token)  # detect corruption before RL
            checks[s+'_cache_coverage']=1.0
    except Exception as exc:errors.append(f'Compatibility check failed: {type(exc).__name__}: {exc}')
    return {'status':'failed' if errors else 'succeeded','errors':errors,'checks':checks}

def main():
    p=argparse.ArgumentParser(description='Read-only resource and compatibility audit; no auto-downloads')
    p.add_argument('--stage',choices=['warmup','rl'],required=True);p.add_argument('--split',choices=['val','test'])
    p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    result={'status':'dry_run','stage':a.stage,'split':a.split} if a.dry_run else preflight(a.stage,a.split)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);json_write(a.output,result);print(json.dumps(result,indent=2))
    if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
