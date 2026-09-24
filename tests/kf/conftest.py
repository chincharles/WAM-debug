import os
import sys
from pathlib import Path
from types import SimpleNamespace,MethodType
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/kf'))

@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch,request):
    if request.node.get_closest_marker("gpu"):
        return
    torch.set_num_threads(1);torch.manual_seed(7)
    for key in ['NAVSIM_LOG_PATH','NAVSIM_SENSOR_BLOBS_PATH','NAVSIM_METRIC_CACHE_PATH','NAVSIM_DEVKIT_ROOT']:
        monkeypatch.setenv(key,'/unavailable')

@pytest.fixture
def tiny():
    from simwam.models.wan22.wan_video_dit import WanVideoDiT
    from simwam.models.wan22.action_dit import ActionDiT
    from simwam.models.wan22.mot import MoT
    from simwam.models.wan22.simwam_kf import SimWAMKF
    from simwam.kf.config import compose_config
    from omegaconf import OmegaConf
    common=dict(hidden_dim=32,ffn_dim=64,text_dim=16,freq_dim=16,eps=1e-6,num_heads=2,attn_head_dim=12,num_layers=2)
    video=WanVideoDiT(**common,in_dim=4,out_dim=4,patch_size=(1,2,2),has_image_input=False,
                     seperated_timestep=True,video_attention_mask_mode='first_frame_causal')
    action=ActionDiT(**common,action_dim=3)
    mot=MoT(mixtures={'video':video,'action':action},mot_checkpoint_mixed_attn=False)
    vae=SimpleNamespace(model=SimpleNamespace(z_dim=4),temporal_downsample_factor=4,upsampling_factor=4,to=lambda *a,**kw:None)
    model=SimWAMKF(video,action,mot,vae,text_dim=16,proprio_dim=8)
    cfg=compose_config();kf=OmegaConf.to_container(cfg.kf,resolve=True)
    kf['adapter'].update(inner_dim=16,num_heads=2,layer_indices=[0,1]);kf['generation']['num_inference_steps']=2
    model.setup_kf(kf)
    def encode(self,input_image,tiled=False):
        image=torch.nn.functional.avg_pool2d(input_image,4)
        return torch.cat([image,image[:,:1]],dim=1)[:,:,None]
    model._encode_input_image_latents_tensor=MethodType(encode,model)
    model.eval()
    return model,cfg

@pytest.fixture
def batch():
    return {'video':torch.randn(2,3,3,16,16),'context':torch.randn(2,3,16),
        'context_mask':torch.ones(2,3,dtype=torch.bool),'proprio':torch.randn(2,8,8),
        'action':torch.randn(2,8,3),'token':['s0','s1']}
