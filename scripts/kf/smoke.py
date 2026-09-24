import _bootstrap
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from simwam.kf.contracts import read_manifest,validate_splits
from simwam.kf.execution import prepare_output,json_write

def main():
    p=argparse.ArgumentParser(description='Real data smoke: regression, 2 warmup steps, four 2-update RL cells, F0/F8 eval')
    p.add_argument('--output-root',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    root=prepare_output(a.output_root);env=os.environ.copy()
    if not a.dry_run:
        subsets={s:read_manifest(env[f'KF_{s.upper()}_MANIFEST'])[:8] for s in ('train','val')}
        validate_splits(subsets)
        for s,rows in subsets.items():
            path=root/f'{s}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            env[f'KF_{s.upper()}_MANIFEST']=str(path.resolve())
    commands=[[sys.executable,'scripts/kf/gpu_checks.py','--output',str(root/'gpu_regression.json')],
        ['bash','scripts/kf/warmup.sh','--seed','42','--max-steps','2','--run-dir',str(root/'warmup')]]
    for cmd in commands:
        subprocess.run(cmd+(['--dry-run'] if a.dry_run else []),env=env,check=True)
    env['SIMWAM_KF_CHECKPOINT']=str((root/'warmup/export/kf_policy.pt').resolve())
    cmd=['bash','scripts/kf/run_matrix.sh','--stage','smoke','--seeds','42','--max-steps','2','--output-root',str(root/'matrix')]
    subprocess.run(cmd+(['--dry-run'] if a.dry_run else []),env=env,check=True)
    json_write(root/'status.json',{'status':'dry_run' if a.dry_run else 'succeeded','inner_epochs':1,'research_result':False})
if __name__=='__main__':main()
