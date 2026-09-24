"""P0 loop: upstream PPO/BC objective, explicit rank reduction, rollout-boundary resume.

One observation per rank by default. Every rank has a complete model; no claim of
ZeRO sharding. Future activations use microbatch=1. Loss remains candidate-mean.
"""
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .kf.contracts import noise_plan, stable_seed


def set_trainable(model,stage):
    model.eval(); model.requires_grad_(False)
    for name,p in model.action_expert.named_parameters():
        p.requires_grad_('future_adapters.' in name if stage == 'warmup' else 'lora_' in name)
    return [n for n,p in model.action_expert.named_parameters() if p.requires_grad]

def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def capture_rng():
    return {'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    torch.set_rng_state(state['torch']); np.random.set_state(state['numpy']); random.setstate(state['python'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])

def save_training_state(path,model,optimizer,scheduler,progress,config_signature):
    torch.save({'mot':model.mot.state_dict(),'proprio_encoder':model.proprio_encoder.state_dict() if model.proprio_encoder is not None else None,'reference':model.ref_action_expert.state_dict() if hasattr(model,'ref_action_expert') else None,
        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng(),
        'progress':progress,'resume_boundary':'rollout','config_signature':config_signature},path)

def load_training_state(path,model,optimizer,scheduler,config_signature):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('resume_boundary') != 'rollout' or state['config_signature'] != config_signature:
        raise ValueError('Resume requires same config/data/world size at a rollout boundary')
    model.mot.load_state_dict(state['mot'],strict=True)
    if model.proprio_encoder is not None:model.proprio_encoder.load_state_dict(state['proprio_encoder'],strict=True)
    if state['reference'] is not None: model.ref_action_expert.load_state_dict(state['reference'],strict=True)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    return state['progress']

class KFTrainer:
    def __init__(self,model,dataset,reward,cfg,stage='rl'):
        self.model,self.train_dataset,self.reward,self.cfg,self.stage=model,dataset,reward,cfg,stage
        self.rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.output=Path(cfg.output_dir); self.seed=int(cfg.seed)
        self.group_size=int(cfg.grpo.sample.group_size)
        self.adv_eps=float(cfg.grpo.train.adv_eps)
        self.ppo_clip_range=float(cfg.grpo.train.ppo_clip_range)
        self.adv_clip_max=float(cfg.grpo.train.adv_clip_max)
        self.inner=int(cfg.grpo.train.num_inner_epochs) if stage=='rl' else 1
        if int(cfg.max_steps)%self.inner:
            raise ValueError('max_steps must end at a complete rollout boundary (multiple of inner epochs)')
        names=set_trainable(model,stage)
        self.params=[p for p in model.action_expert.parameters() if p.requires_grad]
        if not self.params: raise ValueError('No trainable parameters')
        if self.rank==0:
            (self.output/'trainable_parameters.json').write_text(json.dumps({'names':names,'count':sum(p.numel() for p in self.params)},indent=2))
        self.optimizer=torch.optim.AdamW(self.params,lr=float(cfg.learning_rate),weight_decay=float(cfg.weight_decay),betas=(.9,.95))
        # Same capped linear warmup + constant schedule as upstream default.
        n=min(int(cfg.max_steps*.05),int(cfg.lr_warmup_steps))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step: min(1.,(step+1)/max(n,1)))
        self.progress={'step':0,'rollout':0,'epoch':0,'batch':0,'observed_scenes':0,'sampled_trajectories':0}
        self.signature={k:model.kf_metadata[k] for k in ('il_hash','split_hashes','c1_hash')}
        self.signature.update(world=self.world,stage=stage,K=self.group_size,F=int(cfg.kf.future_frames),seed=self.seed,
            max_steps=int(cfg.max_steps),inner=self.inner,batch_size=int(cfg.batch_size),lr=float(cfg.learning_rate))
        from omegaconf import OmegaConf
        from hashlib import sha256
        protocol=OmegaConf.to_container(cfg,resolve=True);protocol.pop('output_dir',None);protocol.pop('resume',None)
        self.signature['protocol_hash']=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if cfg.resume:
            path=Path(cfg.resume)/f'rank{self.rank}.pt'
            self.progress=load_training_state(path,model,self.optimizer,self.scheduler,self.signature)

    def batch(self):
        from torch.utils.data._utils.collate import default_collate
        n=len(self.train_dataset); global_batch=int(self.cfg.batch_size)*self.world
        if n<global_batch: raise ValueError('Manifest smaller than global observation batch')
        batches=n//global_batch  # fixed drop-last; documented and identical across cells
        if self.progress['batch']>=batches:
            self.progress['epoch']+=1;self.progress['batch']=0
        order=torch.randperm(n,generator=torch.Generator().manual_seed(stable_seed(self.seed,'data',self.progress['epoch'],'all'))).tolist()
        start=self.progress['batch']*global_batch+self.rank*int(self.cfg.batch_size)
        self.progress['batch']+=1
        return default_collate([self.train_dataset[i] for i in order[start:start+int(self.cfg.batch_size)]])

    def condition(self,batch):
        # Expert actions are not consulted for horizon or any policy condition.
        return self.model.build_action_condition(batch['video'][:,:,0],8,batch['context'],batch['context_mask'],
            batch['proprio'][:,0] if self.model.proprio_encoder is not None else None)

    @torch.no_grad()
    def rollout(self,batch):
        model=self.model; tokens=list(batch['token']); rid=self.progress['rollout']
        synchronize();start=time.perf_counter()
        cond=self.condition(batch)
        synchronize();condition_end=time.perf_counter()
        future=model.with_future(cond,tokens,self.group_size,self.seed,rid,int(self.cfg.kf.future_frames))
        synchronize();future_end=time.perf_counter()
        plan=noise_plan(tokens,self.group_size,self.seed,rid,model.grpo_num_inference_steps,device=model.device,dtype=model.torch_dtype)
        chain,ts,ds=model.sample_action_chain(future,8,init_actions=plan[:,0],step_noises=plan[:,1:])
        synchronize();action_end=time.perf_counter()
        absolute=self.train_dataset.denormalize_action(chain[:,-1])
        expanded=[t for t in tokens for _ in range(self.group_size)]
        rewards=self.reward.score_batch(absolute,expanded,device=model.device)
        synchronize();reward_end=time.perf_counter()
        mat=rewards.reshape(len(tokens),self.group_size)
        std=mat.std(1,unbiased=False,keepdim=True)
        advantages=((mat-mat.mean(1,keepdim=True))/(std+self.adv_eps)).flatten()
        qlo=float(self.cfg.grpo.train.adv_clip_lower_quantile);qhi=float(self.cfg.grpo.train.adv_clip_upper_quantile)
        if qlo>0 or qhi<1:
            advantages=advantages.clamp(torch.quantile(advantages,qlo),torch.quantile(advantages,qhi))
        old=model.action_chain_logprobs(future,chain,ts,ds).detach()
        # Separate fixed reference random stream; never consumes action/future streams.
        ref=noise_plan(tokens,1,self.seed,rid,model.grpo_num_inference_steps,stream='reference',device=model.device,dtype=model.torch_dtype)
        ref_chain,ref_ts,ref_ds=model.sample_action_chain(cond,8,velocity_use_ref=True,init_actions=ref[:,0],step_noises=ref[:,1:])
        poses=absolute.reshape(len(tokens),self.group_size,8,3).float()
        xy=(poses[:,:,None,:,:2]-poses[:,None,:,:,:2]).norm(dim=-1).mean()
        yaw=(poses[:,:,None,:,2]-poses[:,None,:,:,2]);yaw=torch.atan2(yaw.sin(),yaw.cos()).abs().mean()
        return dict(cond_b=cond,cond_bg=future,chain=chain,timesteps=ts,deltas=ds,rewards=rewards,
            advantages=advantages,old_logp=old,reference_chain=(ref_chain,ref_ts,ref_ds),
            diagnostics={'current_condition_s':condition_end-start,'future_generation_s':future_end-condition_end,'action_sampling_s':action_end-future_end,
                'reward_s':reward_end-action_end,'group_reward_std':std.mean().item(),
                'zero_advantage_fraction':(std==0).float().mean().item(),'invalid_trajectory_fraction':0.,
                'xy_diversity_m':xy.item(),'yaw_diversity_rad':yaw.item(),
                'reward_components':{k:float(np.mean([r[k] for r in self.reward.last_results])) for k in self.reward.last_results[0]}})

    def policy_loss(self,rollout):
        # Reuse exact upstream PPO transition log-prob, discount, clip and reduction.
        # The only extension to its objective is precomputed independent F0 reference chain.
        from .trainer_grpo import SimWAMGRPOTrainer
        return SimWAMGRPOTrainer._policy_loss(self,self.model,rollout)

    def train(self):
        start=time.perf_counter()
        while self.progress['step']<int(self.cfg.max_steps):
            batch=self.batch()
            rollout=self.rollout(batch) if self.stage=='rl' else None
            for _ in range(self.inner):
                synchronize();update_start=time.perf_counter()
                if self.stage=='warmup':
                    from .trainer_kf_warmup import warmup_loss
                    loss=warmup_loss(self.model,batch,self.seed,self.progress['rollout'])
                    metrics={'warmup_loss':loss.detach()}
                else:
                    loss,metrics=self.policy_loss(rollout)
                    if _ == 0 and abs(float(metrics['ratio_mean'])-1.) > 0.02:
                        raise FloatingPointError('Unchanged-policy PPO ratio differs from one; inspect condition/noise/dropout')
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                if self.world>1:
                    for p in self.params:
                        if p.grad is None: p.grad=torch.zeros_like(p)
                        torch.distributed.all_reduce(p.grad);p.grad.div_(self.world)
                norm=torch.nn.utils.clip_grad_norm_(self.params,float(self.cfg.max_grad_norm),error_if_nonfinite=True)
                self.optimizer.step();self.scheduler.step();self.progress['step']+=1
                synchronize();elapsed=time.perf_counter()-start
                row={'seed':self.seed,'K':self.group_size,'F_train':int(self.cfg.kf.future_frames),'optimizer_step':self.progress['step'],
                    'rollout_id':self.progress['rollout'],'resume_from':self.cfg.resume,'rank':self.rank,'world_size':self.world,'loss':loss.item(),
                    'valid_scenes':len(batch['token'])*self.world,
                    'observed_scenes':self.progress['observed_scenes']+len(batch['token'])*self.world,
                    'sampled_trajectories':self.progress['sampled_trajectories']+len(batch['token'])*self.world*(self.group_size if self.stage=='rl' else 1),
                    'grad_norm':float(norm),'learning_rate':self.optimizer.param_groups[0]['lr'],
                    'update_s':time.perf_counter()-update_start,'wall_s':elapsed,'gpu_hours':elapsed*self.world/3600 if torch.cuda.is_available() else None,
                    'peak_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    'peak_reserved':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    'adapter_gates':{k:float(a.alpha.detach()) for k,a in self.model.action_expert.future_adapters.items()},
                    **{k:float(v) for k,v in metrics.items()}}
                if rollout is not None:
                    row.update(rollout['diagnostics']);row.update(reward_mean=rollout['rewards'].mean().item(),reward_std=rollout['rewards'].std(unbiased=False).item())
                with (self.output/f'train_metrics.rank{self.rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if self.rank==0:
                    with (self.output/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps({k:row[k] for k in ('optimizer_step','rollout_id','loss','grad_norm')}),flush=True)
            self.progress['rollout']+=1
            self.progress['observed_scenes']+=len(batch['token'])*self.world
            self.progress['sampled_trajectories']+=len(batch['token'])*self.world*(self.group_size if self.stage=='rl' else 1)
            # Save only after every inner epoch is consumed. No dropped on-policy buffer.
            if self.progress['step']%int(self.cfg.save_every)==0 or self.progress['step']==int(self.cfg.max_steps):
                state=self.output/'checkpoints'/f"step_{self.progress['step']:06d}";state.mkdir(parents=True,exist_ok=True)
                save_training_state(state/f'rank{self.rank}.pt',self.model,self.optimizer,self.scheduler,self.progress,self.signature)
                if self.world>1:torch.distributed.barrier()
        if self.rank==0:
            export=self.output/'export';export.mkdir(exist_ok=True)
            self.model.save_checkpoint(export/'kf_policy.pt',step=self.progress['step'])
        if self.world>1:torch.distributed.barrier()
        return self.progress
