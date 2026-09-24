"""Supervised adapter-only velocity warmup using upstream action scheduler."""
import torch
from .kf.contracts import noise_plan, stable_seed

def warmup_loss(model,batch,seed,rollout):
    tokens=list(batch['token'])
    cond=model.build_action_condition(batch['video'][:,:,0],8,batch['context'],batch['context_mask'],
        batch['proprio'][:,0] if model.proprio_encoder is not None else None)
    cond=model.with_future(cond,tokens,1,seed,rollout,8)
    target=batch['action'].to(device=model.device,dtype=model.torch_dtype)
    noise=noise_plan(tokens,1,seed,rollout,0,stream='warmup',device=model.device,dtype=model.torch_dtype)[:,0]
    scheduler=model.train_action_scheduler
    # Isolated RNG while retaining upstream training time sampling exactly.
    with torch.random.fork_rng(devices=[model.device.index or 0] if model.device.type=='cuda' else []):
        torch.manual_seed(stable_seed(seed,'warmup_time',rollout,'batch'))
        t=scheduler.sample_training_t(len(tokens),model.device,model.torch_dtype)
    x=scheduler.add_noise(target,noise,t)
    prediction=model.action_velocity(x,t,cond)
    velocity=scheduler.training_target(target,noise,t)
    return ((prediction.float()-velocity.float()).square().mean((1,2))*scheduler.training_weight(t)).mean()
