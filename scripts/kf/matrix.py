import _bootstrap
import argparse
import os
import subprocess
import sys
from pathlib import Path
from simwam.kf.contracts import file_hash
from simwam.kf.execution import prepare_output,json_write

def main():
    p=argparse.ArgumentParser(description='Fail-fast four-cell matrix; one immutable C1 for every run')
    p.add_argument('--stage',choices=['pilot','confirm','smoke'],default='pilot');p.add_argument('--seeds',default='42')
    p.add_argument('--max-steps',type=int,default=200);p.add_argument('--output-root',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();root=prepare_output(a.output_root)
    c1=os.environ.get('SIMWAM_KF_CHECKPOINT')
    if a.dry_run and not c1:
        c1='/UNSET/SIMWAM_KF_CHECKPOINT';os.environ['SIMWAM_KF_CHECKPOINT']=c1
    if not c1:raise ValueError('Set SIMWAM_KF_CHECKPOINT to the common C1')
    digest=None if a.dry_run else file_hash(c1)
    json_write(root/'matrix.json',{'stage':a.stage,'seeds':a.seeds,'common_C1':c1,'common_C1_hash':digest,'max_steps':a.max_steps})
    for seed in [int(s) for s in a.seeds.split(',')]:
        for k,f in ((4,0),(8,0),(4,8),(8,8)):
            run=root/f'k{k}_f{f}_seed{seed}'
            if not a.dry_run and file_hash(c1)!=digest:raise RuntimeError('Common C1 changed during matrix')
            cmd=['bash','scripts/kf/train.sh','--k',str(k),'--future-frames',str(f),'--seed',str(seed),
                '--max-steps',str(a.max_steps),'--run-dir',str(run)]
            if a.stage=='smoke':cmd+=['--inner-epochs','1']
            if a.dry_run:cmd+=['--dry-run']
            subprocess.run(cmd,check=True)
            for ef in (0,8):
                cmd=[sys.executable,'scripts/kf/eval.py','--checkpoint',str(run/'export/kf_policy.pt'),'--split','val',
                    '--future-frames',str(ef),'--candidate-count','1','--action-steps','10','--seed','2026',
                    '--output-dir',str(run/f'eval_val_f{ef}')]
                if a.dry_run:cmd+=['--dry-run']
                subprocess.run(cmd,check=True)
if __name__=='__main__':main()
