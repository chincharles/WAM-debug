import _bootstrap
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Train common C1 and evaluate C0/F0, C1/F0, C1/F8')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-steps',type=int,default=1000)
    p.add_argument('--run-dir',required=True);p.add_argument('--resume');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    cmd=['bash','scripts/kf/train.sh','--stage','warmup','--seed',str(a.seed),'--max-steps',str(a.max_steps),'--run-dir',a.run_dir]
    if a.resume:cmd+=['--resume',a.resume]
    if a.dry_run:cmd+=['--dry-run']
    subprocess.run(cmd,check=True)
    root=Path(a.run_dir)
    for label,checkpoint,f in [('c0',os.environ.get('SIMWAM_IL_CHECKPOINT','/missing/C0'),0),
                               ('c1',str(root/'export/kf_policy.pt'),0),('c1',str(root/'export/kf_policy.pt'),8)]:
        cmd=[sys.executable,'scripts/kf/eval.py','--checkpoint',checkpoint,'--split','val','--future-frames',str(f),
             '--output-dir',str(root/f'eval_{label}_val_f{f}')]
        if label=='c0':cmd+=['--allow-c0']
        if f==8:cmd+=['--diagnostics']
        if a.dry_run:cmd+=['--dry-run']
        subprocess.run(cmd,check=True)
    if not a.dry_run:
        import json
        import numpy as np
        r0=[json.loads(s) for s in (root/'eval_c0_val_f0/per_scene.jsonl').read_text().splitlines()]
        r1=[json.loads(s) for s in (root/'eval_c1_val_f0/per_scene.jsonl').read_text().splitlines()]
        if [r['scene_token'] for r in r0]!=[r['scene_token'] for r in r1]:raise ValueError('C0/C1 regression scene mismatch')
        delta=max(float(np.max(np.abs(np.asarray(a['trajectory'])-np.asarray(b['trajectory'])))) for a,b in zip(r0,r1))
        if delta!=0:raise ValueError(f'Adapter-only warmup changed F0 trajectory: max_abs={delta}')
        (root/'c0_c1_f0_regression.json').write_text(json.dumps({'status':'succeeded','trajectory_max_abs':delta}))
if __name__=='__main__':main()
