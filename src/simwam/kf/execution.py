"""Shared real-model execution and provenance (no auto-download or fake scores)."""
import json
import os
import platform
import subprocess
from pathlib import Path
from .contracts import file_hash,read_manifest,validate_splits,UPSTREAM_SHA

def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')

def prepare_output(path,resume=None):
    path=Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f'Refusing nonempty output: {path}; use explicit --resume for full training state')
    path.mkdir(parents=True,exist_ok=True)
    return path

def dataset_config(cfg,split):
    from omegaconf import OmegaConf
    config=OmegaConf.to_container(cfg.data.train,resolve=True)
    config['manifest_path']=os.environ[f'KF_{split.upper()}_MANIFEST']
    config['is_training_set']=split=='train'
    if split=='test':
        config['navsim_log_path']=os.environ['NAVSIM_TEST_LOG_PATH']
        config['sensor_blobs_path']=os.environ['NAVSIM_TEST_SENSOR_BLOBS_PATH']
        config['scene_filter']=str(Path(os.environ['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml')
    return config

def make_dataset(cfg,split):
    from hydra.utils import instantiate
    return instantiate(dataset_config(cfg,split))

def make_reward(cfg,split,dataset,official=False):
    from omegaconf import OmegaConf
    from simwam.datasets.navsim.kf_validation import StrictPDMReward
    args=OmegaConf.to_container(cfg.grpo.reward,resolve=True)
    args['metric_cache_path']=os.environ[{'train':'NAVSIM_METRIC_CACHE_PATH','val':'NAVSIM_VAL_METRIC_CACHE_PATH','test':'NAVSIM_TEST_METRIC_CACHE_PATH'}[split]]
    return StrictPDMReward(allowed_tokens=dataset.tokens,official=official,**args)

def load_model(cfg,checkpoint,require_kf,device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    if not Path(checkpoint).is_file(): raise FileNotFoundError(f'Required C0/C1 checkpoint: {checkpoint}')
    # Inspect metadata before allocating the 5B model.
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if require_kf and 'kf_metadata' not in payload: raise ValueError('RL/evaluation requires C1/KF export with adapter metadata')
    if 'mot' not in payload: raise ValueError('Checkpoint is not an upstream SimWAM IL/KF export')
    if require_kf:
        metadata=payload['kf_metadata']
        for split in ('train','val'):
            if metadata['split_hashes'][split]!=file_hash(os.environ[f'KF_{split.upper()}_MANIFEST']):
                raise ValueError(f'Checkpoint {split} manifest mismatch; cannot verify evaluation independence')
        if metadata['normalization']!='upstream_fixed_odo_absolute': raise ValueError('Normalization mismatch')
        if metadata['stats_hash']!=file_hash(os.environ['NAVSIM_STATS_PATH']): raise ValueError('Checkpoint statistics provenance mismatch')
        vae_file=Path(os.environ['DIFFSYNTH_MODEL_BASE_PATH'])/'DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'
        if metadata['vae_hash']!=file_hash(vae_file):raise ValueError('Checkpoint VAE mismatch')
    del payload
    args=OmegaConf.to_container(cfg.model,resolve=True);args.pop('checkpoint_path',None)
    model=instantiate(args,device=device,model_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    model.load_checkpoint(checkpoint)
    if not require_kf:
        model.kf_metadata['il_hash']=file_hash(checkpoint)
        model.kf_metadata['il_source']=str(Path(checkpoint).resolve())
    return model

def provenance(cfg,output,model):
    import torch
    import importlib.metadata
    from omegaconf import OmegaConf
    splits={s:read_manifest(os.environ[f'KF_{s.upper()}_MANIFEST']) for s in ('train','val')}
    meta=validate_splits(splits)
    hashes={s:file_hash(os.environ[f'KF_{s.upper()}_MANIFEST']) for s in splits}
    vae_path=Path(os.environ['DIFFSYNTH_MODEL_BASE_PATH'])/'DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'
    model.kf_metadata.update(vae_hash=file_hash(vae_path),split_hashes=hashes,c1_hash=file_hash(cfg.model.checkpoint_path),
        stats_hash=file_hash(os.environ['NAVSIM_STATS_PATH']),seed=int(cfg.seed),train_K=int(cfg.grpo.sample.group_size),
        train_F=int(cfg.kf.future_frames),resolved_config=OmegaConf.to_container(cfg,resolve=True))
    json_write(output/'data_manifest_meta.json',{'splits':meta,'hashes':hashes})
    root=Path(__file__).resolve().parents[3]
    versions={}
    for name in ('torch','torchvision','transformers','accelerate','numpy','hydra-core','deepspeed'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    json_write(output/'environment.json',{'upstream_sha':UPSTREAM_SHA,
        'working_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'navsim_tree':subprocess.check_output(['git','rev-parse','HEAD:navsim'],cwd=root,text=True).strip(),
        'source_hashes':{str(p.relative_to(root)):file_hash(p) for folder in ('src/simwam','scripts/kf','configs') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml','.json')},
        'dirty_diff_sha256':__import__('hashlib').sha256(subprocess.check_output(['git','diff'],cwd=root)).hexdigest(),
        'python':platform.python_version(),'versions':versions,'cuda':torch.version.cuda,
        'world_size':int(os.environ.get('WORLD_SIZE',1)),
        'gpus':[{'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        'weights':{'IL':model.kf_metadata['il_hash'],'initial':model.kf_metadata['c1_hash']}})
    OmegaConf.save(cfg,output/'resolved_config.yaml',resolve=True)
