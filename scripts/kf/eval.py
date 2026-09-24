import _bootstrap
import argparse
import json
import time
from pathlib import Path
from simwam.kf.contracts import file_hash,stable_seed
from simwam.kf.scoring import reward_to_report
from simwam.kf.execution import prepare_output,json_write,load_model,make_dataset,make_reward

def main():
    p=argparse.ArgumentParser(description='Official NAVSIM v1 single-trajectory policy evaluation')
    p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--future-frames',type=int,choices=[0,8],default=0);p.add_argument('--candidate-count',type=int,choices=[1],default=1)
    p.add_argument('--action-steps',type=int,choices=[10],default=10);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output-dir',required=True);p.add_argument('--allow-c0',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--diagnostics',action='store_true')
    a=p.parse_args()
    from simwam.kf.config import compose_config,dry_run_environment
    if a.dry_run:dry_run_environment()
    from omegaconf import OmegaConf
    cfg=compose_config('rl',[f'evaluation.future_frames={a.future_frames}'])
    output=prepare_output(a.output_dir);OmegaConf.save(cfg,output/'resolved_config.yaml',resolve=True)
    if a.dry_run:json_write(output/'evaluation_plan.json',vars(a));return
    import torch
    import numpy as np
    from simwam.trainer_kf_grpo import synchronize
    if not torch.cuda.is_available():raise RuntimeError('Real evaluator needs CUDA and C0/C1; no CPU PDMS substitution')
    if a.allow_c0 and a.future_frames:raise ValueError('C0 has no warmed adapter; baseline evaluation requires F0')
    json_write(output/'status.json',{'status':'running'})
    try:
        from preflight import preflight
        report=preflight('warmup',a.split)
        if report['errors']:raise RuntimeError(json.dumps(report['errors']))
        model=load_model(cfg,a.checkpoint,not a.allow_c0,'cuda:0').eval()
        dataset=make_dataset(cfg,a.split);reward=make_reward(cfg,a.split,dataset,official=True)
        checkpoint_hash=file_hash(a.checkpoint)
        rows=[];latencies=[]
        for i in range(len(dataset)):
            sample=dataset[i];token=sample['token'];seed=stable_seed(a.seed,'evaluation_action',0,token)
            kwargs=dict(prompt=None,input_image=sample['video'][:,0],action_horizon=8,
                context=sample['context'],context_mask=sample['context_mask'],proprio=sample['proprio'][0],
                seed=seed,scene_token=token,num_inference_steps=a.action_steps,future_frames=a.future_frames)
            if i==0:model.infer_action(**kwargs)  # excluded kernel warmup, not scored/selected
            synchronize();start=time.perf_counter();pred=model.infer_action(**kwargs)['action']
            synchronize();latency=time.perf_counter()-start;latencies.append(latency)
            poses=dataset.denormalize_action(pred[None]);reward.score_batch(poses,[token])
            result=reward.last_results[0]
            row={'scene_token':token,'log_id':sample['log_id'],'seed':a.seed,
                'train_seed':model.kf_metadata.get('seed'),'eval_seed':a.seed,
                'K_train':model.kf_metadata.get('train_K'),'F_train':model.kf_metadata.get('train_F'),
                'F_eval':a.future_frames,'action_steps':a.action_steps,'checkpoint_hash':checkpoint_hash,
                'PDMS':reward_to_report(result['score']),'NC':result['no_at_fault_collisions'],'DAC':result['drivable_area_compliance'],
                'EP':result['ego_progress'],'TTC':result['time_to_collision_within_bound'],'comfort':result['comfort'],
                'trajectory':poses[0].tolist(),'latency_s':latency}
            rows.append(row)
            with (output/'per_scene.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        summary={'PDMS':float(np.mean([r['PDMS'] for r in rows])),'scene_count':len(rows),
            'split':a.split,'F_eval':a.future_frames,'K_eval':1,'action_steps':10,'video_steps':int(cfg.kf.generation.num_inference_steps),
            'seed':a.seed,'checkpoint_hash':checkpoint_hash,'K_train':model.kf_metadata.get('train_K'),
            'F_train':model.kf_metadata.get('train_F'),'latency_s':float(np.mean(latencies)),
            'latency_scope':'after one warmup; model only, includes VAE encode/current KV/future; excludes dataset I/O, T5 and RGB decode',
            'official_scorer':'vendored NAVSIM v1 PDMScorerConfig defaults',
            'manifest_hash':file_hash(dataset_config_path(a.split))}
        json_write(output/'summary.json',summary)
        if a.diagnostics:
            from diagnostics import run_diagnostics
            json_write(output/'diagnostics.json',run_diagnostics(model,dataset,reward,a.seed))
        json_write(output/'status.json',{'status':'succeeded'})
    except Exception as exc:
        json_write(output/'status.json',{'status':'failed','reason':str(exc)});raise

def dataset_config_path(split):
    import os
    return os.environ[f'KF_{split.upper()}_MANIFEST']
if __name__=='__main__':main()
