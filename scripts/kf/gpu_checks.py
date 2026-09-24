import _bootstrap
import argparse
import json
import os
from pathlib import Path
from simwam.kf.execution import load_model,make_dataset,json_write

def main():
    p=argparse.ArgumentParser(description='Real-weight F0 and video-only/joint latent regression')
    p.add_argument('--output',required=True);p.add_argument('--atol',type=float,default=0.02)
    p.add_argument('--rtol',type=float,default=0.02);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:print(json.dumps(vars(a)));return
    import torch
    from simwam.kf.config import compose_config
    from simwam.models.wan22.simwam import SimWAM
    from simwam.models.wan22.future_generator import sample_video_latents
    if not torch.cuda.is_available():raise RuntimeError('GPU integration requires CUDA')
    cfg=compose_config('warmup');m=load_model(cfg,os.environ['SIMWAM_IL_CHECKPOINT'],False,'cuda:0')
    s=make_dataset(cfg,'train')[0]
    kwargs=dict(prompt=None,input_image=s['video'][:,0],action_horizon=8,proprio=s['proprio'][0],
                context=s['context'],context_mask=s['context_mask'],num_inference_steps=10,seed=2026)
    with torch.no_grad():
        reference=SimWAM.infer_action(m,**kwargs)['action'];actual=m.infer_action(**kwargs)['action']
        torch.testing.assert_close(actual,reference,atol=a.atol,rtol=a.rtol)
        c=m.build_action_condition(kwargs['input_image'][None],8,s['context'][None],s['context_mask'][None],s['proprio'][0][None])
        first=c['first_frame_latents'];shape=(1,first.shape[1],8//m.vae.temporal_downsample_factor+1,*first.shape[-2:])
        z=torch.randn(shape,generator=torch.Generator().manual_seed(2026)).to(m.device,m.torch_dtype);z[:,:,0:1]=first
        joint=z.clone();ts,ds=m.infer_video_scheduler.build_inference_schedule(20,m.device,m.torch_dtype)
        for t,d in zip(ts,ds):
            velocity,_=m._predict_joint_noise(joint,torch.zeros(1,8,3,device=m.device,dtype=m.torch_dtype),t[None],t[None],c['context'],c['context_mask'],True)
            joint=m.infer_video_scheduler.step(velocity,d,joint);joint[:,:,0:1]=first
        video=sample_video_latents(m,first,c['context'],c['context_mask'],2026,20,initial_latents=z)
        torch.testing.assert_close(video,joint,atol=a.atol,rtol=a.rtol)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    json_write(a.output,{'status':'succeeded','dtype':str(m.torch_dtype),'atol':a.atol,'rtol':a.rtol,
        'tolerance_note':'Provisional explicit bf16 threshold; inspect measured differences before accepting experiment',
        'f0_max_abs':(actual-reference).abs().max().item(),'video_max_abs':(video-joint).abs().max().item()})
if __name__=='__main__':main()
