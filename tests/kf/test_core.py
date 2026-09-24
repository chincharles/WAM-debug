import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from simwam.models.wan22.simwam import SimWAM
from simwam.models.wan22.simwam_grpo import SimWAMGRPO
from simwam.models.wan22.future_generator import generate_future_latents,sample_video_latents
from simwam.kf.contracts import noise_plan,stable_seed
from simwam.trainer_kf_grpo import set_trainable,save_training_state,load_training_state

def condition(m,b):
    return m.build_action_condition(b['video'][:,:,0],8,b['context'],b['context_mask'],b['proprio'][:,0])

def test_f0_upstream_inference_and_no_future(tiny,batch,monkeypatch):
    m,_=tiny
    def forbidden(*a,**kw):raise AssertionError('F0 invoked future generation')
    monkeypatch.setattr('simwam.models.wan22.simwam_kf.generate_future_latents',forbidden)
    monkeypatch.setattr('simwam.models.wan22.future_adapter.apply_future_adapter',forbidden)
    args=dict(prompt=None,input_image=batch['video'][0,:,0],action_horizon=8,context=batch['context'][0],
              context_mask=batch['context_mask'][0],proprio=batch['proprio'][0,0],seed=91,num_inference_steps=3)
    expected=SimWAM.infer_action(m,**args)['action'];actual=m.infer_action(**args)['action']
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert m.with_future(condition(m,batch),batch['token'],4,42,0,0)['future_packet'] is None
    assert generate_future_latents(None,None,num_future_frames=0,scene_tokens=[],candidate_ids=[],seeds=[]) is None

@pytest.mark.parametrize('f',[0,8])
def test_ratio_noise_alignment_condition_persistence(tiny,batch,f):
    m,cfg=tiny;m.configure_grpo(cfg.grpo)
    for a in m.action_expert.future_adapters.values():a.alpha.data.fill_(.5)
    c=condition(m,batch);c=m.with_future(c,batch['token'],4,42,0,f)
    plan=noise_plan(batch['token'],4,42,0,10)
    chain,ts,ds=m.sample_action_chain(c,8,init_actions=plan[:,0],step_noises=plan[:,1:])
    with torch.no_grad():old=m.action_chain_logprobs(c,chain,ts,ds)
    new=m.action_chain_logprobs(c,chain,ts,ds)
    torch.testing.assert_close((new-old).exp(),torch.ones_like(new),rtol=0,atol=0)
    assert new.requires_grad
    if f:
        packet=c['future_packet'];assert packet.scene_tokens==['s0']*4+['s1']*4
        assert packet.candidate_ids==[0,1,2,3]*2
        assert not torch.equal(packet.latents[0],packet.latents[1])
        changed=dict(c,future_packet=replace(packet,latents=packet.latents+5))
        assert not torch.allclose(m.action_chain_logprobs(changed,chain,ts,ds),old)
        assert c['future_packet'] is packet
    p8=noise_plan(batch['token'],8,42,0,10).reshape(2,8,11,8,3)
    torch.testing.assert_close(plan.reshape(2,4,11,8,3),p8[:,:4],rtol=0,atol=0)
    assert torch.all(ds<0)

def test_gate_dependency_gradients_reference(tiny,batch):
    m,cfg=tiny;c=condition(m,batch);future=m.with_future(c,batch['token'],1,42,0,8)
    x=torch.randn(2,8,3);t=torch.ones(2)*500
    torch.testing.assert_close(m.action_velocity(x,t,c),m.action_velocity(x,t,future),rtol=0,atol=0)
    set_trainable(m,'warmup');m.action_velocity(x,t,future).square().mean().backward()
    assert any(a.alpha.grad.abs()>0 for a in m.action_expert.future_adapters.values())
    m.zero_grad()
    for a in m.action_expert.future_adapters.values():a.alpha.data.fill_(.5)
    m.action_velocity(x,t,future).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in m.action_expert.future_adapters.parameters())
    m.configure_grpo(cfg.grpo);set_trainable(m,'rl')
    assert all(('lora_' in n)==p.requires_grad for n,p in m.action_expert.named_parameters())
    before=m.action_velocity(x,t,future,use_ref=True).detach()
    m.action_expert.future_adapters['0'].alpha.data.add_(2)
    torch.testing.assert_close(before,m.action_velocity(x,t,future,use_ref=True),rtol=0,atol=0)
    assert not any('lora_' in n for n in m.action_expert.future_adapters.state_dict())

def test_video_only_matches_joint_latent(tiny,batch):
    m,_=tiny;c=condition(m,batch);first=c['first_frame_latents'][:1]
    z=torch.randn(1,4,3,4,4);z[:,:,0:1]=first
    expected=z.clone();ts,ds=m.infer_video_scheduler.build_inference_schedule(2,'cpu',torch.float32)
    for t,d in zip(ts,ds):
        v,_=m._predict_joint_noise(expected,torch.zeros(1,8,3),t[None],t[None],c['context'][:1],c['context_mask'][:1],True)
        expected=m.infer_video_scheduler.step(v,d,expected);expected[:,:,0:1]=first
    actual=sample_video_latents(m,first,c['context'][:1],c['context_mask'][:1],9,2,initial_latents=z)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

def test_export_strict_reload_and_resume(tiny,batch,tmp_path):
    m,cfg=tiny;plain=copy.deepcopy(m);m.configure_grpo(cfg.grpo);set_trainable(m,'rl')
    for n,p in m.action_expert.named_parameters():
        if 'lora_B' in n:p.data.normal_(std=.01)
    for a in m.action_expert.future_adapters.values():a.alpha.data.fill_(.3)
    export=tmp_path/'kf.pt';m.save_checkpoint(export);plain.load_checkpoint(export)
    c=condition(m,batch);c=m.with_future(c,batch['token'],1,42,0,8)
    for cond in (c,dict(c,future_packet=None)):
        x=torch.randn(2,8,3);t=torch.ones(2)*300
        torch.testing.assert_close(m.action_velocity(x,t,cond),plain.action_velocity(x,t,cond),rtol=1e-5,atol=1e-6)
    payload=torch.load(export,weights_only=False);key=next(k for k in payload['mot'] if 'future_adapters' in k)
    del payload['mot'][key];torch.save(payload,export)
    with pytest.raises(ValueError,match='missing'):plain.load_checkpoint(export)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    m.action_velocity(torch.randn(2,8,3),torch.ones(2)*300,c).square().mean().backward();opt.step();sch.step()
    state=tmp_path/'state.pt';save_training_state(state,m,opt,sch,{'step':4,'rollout':1},{'test':1})
    random_expected=torch.rand(4);reference=copy.deepcopy(m.ref_action_expert.state_dict())
    for p in m.parameters():p.data.add_(1)
    progress=load_training_state(state,m,opt,sch,{'test':1})
    assert progress['step']==4 and opt.state and sch.last_epoch==1
    torch.testing.assert_close(torch.rand(4),random_expected,rtol=0,atol=0)
    for n,p in m.ref_action_expert.state_dict().items():torch.testing.assert_close(p,reference[n],rtol=0,atol=0)

def test_warmup_only_adapter_and_no_future_leak(tiny,batch):
    from simwam.trainer_kf_warmup import warmup_loss
    m,_=tiny;set_trainable(m,'warmup')
    before=copy.deepcopy(m.mot.state_dict());opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    loss=warmup_loss(m,batch,42,0);loss.backward();opt.step()
    assert torch.isfinite(loss)
    for n,p in m.mot.state_dict().items():
        if 'future_adapters' not in n:torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    c=condition(m,batch)
    altered=dict(batch,video=batch['video'].clone(),action=batch['action']+100)
    altered['video'][:,:,1:]=999
    c2=condition(m,altered)
    for k in ('context','context_mask','first_frame_latents'):torch.testing.assert_close(c[k],c2[k],rtol=0,atol=0)

def test_upstream_policy_loss_group_mean(tiny,batch):
    from simwam.trainer_grpo import SimWAMGRPOTrainer
    m,cfg=tiny;m.configure_grpo(cfg.grpo);c=condition(m,batch)
    chain,ts,ds=m.sample_action_chain(c,8)
    lp=m.action_chain_logprobs(c,chain,ts,ds).detach()
    runner=SimpleNamespace(adv_clip_max=5.,ppo_clip_range=.02)
    r=dict(cond_b=c,cond_bg=c,chain=chain,timesteps=ts,deltas=ds,advantages=torch.zeros(2),old_logp=lp,reference_chain=(chain,ts,ds))
    loss,metrics=SimWAMGRPOTrainer._policy_loss(runner,m,r)
    assert metrics['policy_loss']==0
    doubled=dict(r,cond_bg=m.expand_condition(c,2),chain=chain.repeat_interleave(2,0),advantages=torch.zeros(4),old_logp=lp.repeat_interleave(2,0))
    loss2,_=SimWAMGRPOTrainer._policy_loss(runner,m,doubled)
    torch.testing.assert_close(loss,loss2)

@pytest.mark.parametrize('f',[0,8])
def test_full_cpu_training_loop(tiny,batch,tmp_path,f):
    from simwam.trainer_kf_grpo import KFTrainer
    m,cfg=tiny;cfg.output_dir=str(tmp_path);cfg.max_steps=4;cfg.save_every=4;cfg.batch_size=2;cfg.kf.future_frames=f
    m.kf_metadata.update(il_hash='tiny-C0',c1_hash='tiny-C1',split_hashes={'train':'tiny'})
    m.configure_grpo(cfg.grpo)
    class Dataset:
        def __len__(self):return 2
        def __getitem__(self,i):return {k:v[i] for k,v in batch.items()}
        def denormalize_action(self,x):return x.detach().cpu()
    class TestReward:
        # Unit-test fixture only, never imported by production reward/evaluator.
        def score_batch(self,poses,tokens,device):
            self.last_results=[{'score':.5} for _ in tokens]
            return torch.linspace(0,1,len(tokens),device=device)
    original=copy.deepcopy(m.mot.state_dict());ref=copy.deepcopy(m.ref_action_expert.state_dict())
    trainer=KFTrainer(m,Dataset(),TestReward(),cfg);progress=trainer.train()
    assert progress['step']==4 and progress['rollout']==1
    assert (tmp_path/'export/kf_policy.pt').exists()
    assert (tmp_path/'checkpoints/step_000004/rank0.pt').exists()
    assert any(not torch.equal(v,original[n]) for n,v in m.mot.state_dict().items() if 'lora_' in n)
    for n,v in m.mot.state_dict().items():
        if 'lora_' not in n:torch.testing.assert_close(v,original[n],atol=0,rtol=0)
    for n,v in m.ref_action_expert.state_dict().items():torch.testing.assert_close(v,ref[n],atol=0,rtol=0)
