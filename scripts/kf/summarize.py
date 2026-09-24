import _bootstrap
import argparse
import json
import csv
import re
from pathlib import Path
import numpy as np
from simwam.kf.execution import prepare_output,json_write

def paired_bootstrap(rows,seed=2026):
    # Cluster bootstrap by log. Preserve all adjacent scenes within sampled logs.
    logs=sorted({r[0] for r in rows});rng=np.random.default_rng(seed)
    if len(logs)<2:return {'mean':float(np.mean([r[1] for r in rows])),'ci95':None,'unit':'log','logs':len(logs)}
    groups={l:[d for log,d in rows if log==l] for l in logs}
    estimates=[np.mean([v for l in rng.choice(logs,len(logs),replace=True) for v in groups[l]]) for _ in range(2000)]
    return {'mean':float(np.mean([r[1] for r in rows])),'ci95':np.quantile(estimates,[.025,.975]).tolist(),'unit':'log','logs':len(logs)}

def main():
    p=argparse.ArgumentParser(description='Summarize actual completed evaluations, never impute missing cells')
    p.add_argument('--runs-root',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();output=prepare_output(a.output_dir);root=Path(a.runs_root)
    if a.dry_run:json_write(output/'plan.json',vars(a));return
    cells={};missing=[];table=[]
    for run in sorted(root.glob('k*_f*_seed*')):
        match=re.fullmatch(r'k(4|8)_f(0|8)_seed(\d+)',run.name)
        if not match:continue
        k,f,seed=map(int,match.groups())
        for ef in (0,8):
            d=run/f'eval_val_f{ef}'
            if not (d/'summary.json').exists() or not (d/'status.json').exists() or json.loads((d/'status.json').read_text())['status']!='succeeded':
                missing.append(str(d));continue
            summary=json.loads((d/'summary.json').read_text())
            metrics_path=run/'train_metrics.jsonl'
            logs=[json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
            summary['gpu_hours']=logs[-1].get('gpu_hours') if logs else None
            rows=[json.loads(x) for x in (d/'per_scene.jsonl').read_text().splitlines()]
            if summary['K_eval']!=1 or summary['action_steps']!=10:raise ValueError('Incompatible evaluation protocol')
            cells[k,f,ef,seed]=(summary,{r['scene_token']:r for r in rows})
    for k in (4,8):
        for f in (0,8):
            for ef in (0,8):
                matches=[v[0] for key,v in cells.items() if key[:3]==(k,f,ef)]
                if matches and len({(s['scene_count'],s.get('manifest_hash')) for s in matches})!=1:
                    raise ValueError('Seeds must use the same evaluation manifest')
                table.append({'train_K':k,'train_F':f,'eval_F':ef,'seed_count':len(matches),
                    'PDMS':float(np.mean([s['PDMS'] for s in matches])) if matches else None,
                    'PDMS_std':float(np.std([s['PDMS'] for s in matches],ddof=1)) if len(matches)>1 else None,
                    'scene_count':matches[0]['scene_count'] if matches else None,
                    'GPU_hours':float(sum(s['gpu_hours'] for s in matches)) if matches and all(s['gpu_hours'] is not None for s in matches) else None,
                    'latency_s':float(np.mean([s['latency_s'] for s in matches])) if matches else None})
    comparisons={}
    for seed in sorted({key[3] for key in cells}):
        definitions={'delta_K_F0':[(8,0,0,1),(4,0,0,-1)],'delta_K_F8':[(8,8,0,1),(4,8,0,-1)],
            'delta_F_K4':[(4,8,0,1),(4,0,0,-1)],'delta_F_K8':[(8,8,0,1),(8,0,0,-1)],
            'interaction':[(8,8,0,1),(4,8,0,-1),(8,0,0,-1),(4,0,0,1)]}
        for k in (4,8):
            for f in (0,8):definitions[f'inference_K{k}_F{f}']=[(k,f,8,1),(k,f,0,-1)]
        for label,terms in definitions.items():
            keys=[(k,f,ef,seed) for k,f,ef,_ in terms]
            if any(key not in cells for key in keys):comparisons[f'{label}_seed{seed}']={'status':'missing'};continue
            sets=[set(cells[key][1]) for key in keys]
            if any(s!=sets[0] for s in sets):raise ValueError('Paired statistics require identical scene sets')
            values=[(cells[keys[0]][1][t]['log_id'],sum(sign*cells[key][1][t]['PDMS'] for key,(*_,sign) in zip(keys,terms))) for t in sorted(sets[0])]
            comparisons[f'{label}_seed{seed}']=paired_bootstrap(values)
    with (output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    across_seed={}
    for label in {key.rsplit('_seed',1)[0] for key in comparisons}:
        vals=[v['mean'] for key,v in comparisons.items() if key.rsplit('_seed',1)[0]==label and 'mean' in v]
        across_seed[label]={'seed_count':len(vals),'mean':float(np.mean(vals)) if vals else None,
            'std':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
    json_write(output/'summary.json',{'status':'incomplete' if missing or any(r['seed_count']==0 for r in table) else 'complete','across_seed':across_seed,'table':table,'paired_comparisons':comparisons,'missing':missing,
        'interpretation':'No experiment values are filled for missing runs; one seed has no between-seed SD.'})
if __name__=='__main__':main()
