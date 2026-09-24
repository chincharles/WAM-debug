"""Generated future interventions; shuffled futures are explicitly OOD diagnostics."""
import torch
from simwam.kf.contracts import stable_seed

def run_diagnostics(model,dataset,reward,seed,limit=8):
    from dataclasses import replace
    samples=[dataset[i] for i in range(min(limit,len(dataset)))]
    packets=[]
    for s in samples:
        c=model.build_action_condition(s['video'][:,0][None],8,s['context'][None],s['context_mask'][None],s['proprio'][0][None])
        packets.append(model.with_future(c,[s['token']],1,seed,0,8)['future_packet'])
    rows=[]
    for i,s in enumerate(samples):
        outputs={}
        for mode in ('normal','shuffled','disabled'):
            packet=packets[i] if mode=='normal' else replace(packets[i],latents=packets[(i+1)%len(packets)].latents)
            args=dict(prompt=None,input_image=s['video'][:,0],action_horizon=8,context=s['context'],context_mask=s['context_mask'],
                proprio=s['proprio'][0],seed=stable_seed(seed,'diagnostic_action',0,s['token']),future_frames=0 if mode=='disabled' else 8)
            if mode!='disabled':args['future_packet']=packet
            outputs[mode]=dataset.denormalize_action(model.infer_action(**args)['action'][None])
        row={'scene_token':s['token'],'shuffled_is_ood':True}
        for mode,poses in outputs.items():
            row[mode+'_PDMS']=100*reward.score_batch(poses,[s['token']]).item()
            row[mode+'_xy_delta_m']=(poses[...,:2]-outputs['normal'][...,:2]).norm(dim=-1).mean().item()
        rows.append(row)
    return {'rows':rows,'gates':{k:float(a.alpha.detach()) for k,a in model.action_expert.future_adapters.items()}}
