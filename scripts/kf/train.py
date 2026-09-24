import _bootstrap
import argparse
import os
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='KF adapter warmup or fixed K/F FlowGRPO')
    p.add_argument('--stage',choices=['warmup','rl'],default='rl')
    p.add_argument('--k',type=int,choices=[4,8],default=4)
    p.add_argument('--future-frames',type=int,choices=[0,8],default=0)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-steps',type=int,default=200)
    p.add_argument('--run-dir',required=True);p.add_argument('--resume');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--inner-epochs',type=int,default=4)
    a=p.parse_args()
    from simwam.kf.config import compose_config,dry_run_environment
    if a.dry_run:dry_run_environment()
    from simwam.kf.execution import prepare_output,json_write,load_model,make_dataset,make_reward,provenance
    from omegaconf import OmegaConf
    cfg=compose_config(a.stage,[f'grpo.sample.group_size={a.k}',f'kf.future_frames={8 if a.stage=="warmup" else a.future_frames}',
        f'seed={a.seed}',f'max_steps={a.max_steps}',f'output_dir={a.run_dir}',f'grpo.train.num_inner_epochs={a.inner_epochs}'])
    cfg.resume=a.resume
    if a.dry_run:
        output=prepare_output(a.run_dir,a.resume);OmegaConf.save(cfg,output/'resolved_config.yaml',resolve=True)
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    import torch
    from simwam.trainer_kf_grpo import KFTrainer
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if not torch.cuda.is_available():raise RuntimeError('Real-model execution requires a CUDA-compatible accelerator (NVIDIA or validated vendor PPU); CPU tests: pytest tests/kf')
    torch.cuda.set_device(local)
    if world>1:torch.distributed.init_process_group('nccl')
    torch.manual_seed(a.seed)
    output=Path(a.run_dir)
    if rank==0:prepare_output(output,a.resume)
    if world>1:torch.distributed.barrier()
    try:
        if rank==0:json_write(output/'status.json',{'status':'running'})
        from preflight import preflight
        checks=preflight(a.stage)
        if checks['errors']:raise RuntimeError(json.dumps(checks['errors'],ensure_ascii=False))
        model=load_model(cfg,str(cfg.model.checkpoint_path),a.stage=='rl',f'cuda:{local}')
        # Metadata required on every rank for exact resume signature.
        if rank==0:provenance(cfg,output,model)
        if world>1:
            objects=[model.kf_metadata];torch.distributed.broadcast_object_list(objects,src=0);model.kf_metadata=objects[0]
        if a.stage=='rl':model.configure_grpo(cfg.grpo)
        dataset=make_dataset(cfg,'train');reward=make_reward(cfg,'train',dataset) if a.stage=='rl' else None
        trainer=KFTrainer(model,dataset,reward,cfg,a.stage);progress=trainer.train()
        if rank==0:json_write(output/'status.json',{'status':'succeeded',**progress})
    except Exception as exc:
        if rank==0:json_write(output/'status.json',{'status':'failed','reason':str(exc),'last_step':trainer.progress['step'] if 'trainer' in locals() else 0})
        raise
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()
