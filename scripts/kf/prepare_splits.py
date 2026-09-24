import _bootstrap
import argparse
import json
import os
import random
from pathlib import Path
from simwam.kf.contracts import validate_splits,file_hash
from simwam.kf.execution import prepare_output,json_write

def source_scenes(name):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import filter_scenes
    base=Path(os.environ['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/config'
    config=base/f'common/train_test_split/scene_filter/{name}.yaml'
    filt=instantiate(OmegaConf.load(config))
    path=os.environ['NAVSIM_TEST_LOG_PATH' if name=='navtest' else 'NAVSIM_LOG_PATH']
    scenes=filter_scenes(Path(path),filt)
    return [{ 'scene_token':token,'log_id':frames[filt.num_history_frames-1]['log_name'],
              'source_split':name} for token,frames in sorted(scenes.items())]

def main():
    p=argparse.ArgumentParser(description='Official NAVSIM v1 token manifests with log-disjoint development split')
    p.add_argument('--source-split',choices=['navtrain'],default='navtrain')
    p.add_argument('--val-log-fraction',type=float,default=.1);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output-dir',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not 0<a.val_log_fraction<1:p.error('val-log-fraction must be in (0,1)')
    if a.dry_run:print(json.dumps(vars(a)));return
    from omegaconf import OmegaConf
    rows=source_scenes(a.source_split)
    split_path=Path(os.environ['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/config/training/default_train_val_test_log_split.yaml'
    original=OmegaConf.load(split_path)
    val_logs=set(original.get('val_logs',[]));train_logs=set(original.get('train_logs',[]))
    vals=[r for r in rows if r['log_id'] in val_logs];trains=[r for r in rows if r['log_id'] in train_logs]
    method='upstream_train_val_log_split'
    if not vals:
        method='seeded_log_holdout_from_official_train_logs'
        logs=sorted({r['log_id'] for r in trains});random.Random(a.seed).shuffle(logs)
        if len(logs)<2:raise ValueError('Need at least two training logs')
        val_logs=set(logs[:max(1,min(len(logs)-1,round(len(logs)*a.val_log_fraction)))])
        vals=[r for r in trains if r['log_id'] in val_logs];trains=[r for r in trains if r['log_id'] not in val_logs]
    splits={'train':trains,'val':vals,'test':source_scenes('navtest')}
    if any(not v for v in splits.values()):raise ValueError('A required official split is empty')
    meta=validate_splits(splits);output=prepare_output(a.output_dir)
    for name,subset in splits.items():
        for row in subset:row.update(split=name,split_seed=a.seed,split_method=method if name!='test' else 'official_navtest')
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset))
    json_write(output/'metadata.json',{'counts':meta,'method':method,'seed':a.seed,'source_hash':file_hash(split_path),
        'hashes':{s:file_hash(output/f'{s}.jsonl') for s in splits}})
if __name__=='__main__':main()
