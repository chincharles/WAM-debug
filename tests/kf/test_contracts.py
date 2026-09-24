import json
import pytest
from omegaconf import OmegaConf
from simwam.kf.config import compose_config
from simwam.kf.contracts import read_manifest,validate_splits,validate_config

@pytest.mark.parametrize('k,f',[(4,0),(4,8),(8,0),(8,8)])
def test_config(k,f):
    cfg=compose_config(overrides=[f'grpo.sample.group_size={k}',f'kf.future_frames={f}'])
    assert cfg.model._target_=='simwam.runtime_kf.create_simwam_kf_grpo'
    assert cfg.model.kf.future_frames==f
    assert cfg.grpo.train.num_inner_epochs==4
    assert cfg.model.skip_dit_load_from_pretrain

@pytest.mark.parametrize('override',['kf.future_frames=4','evaluation.candidate_count=8','evaluation.action_steps=20'])
def test_invalid_config(override):
    with pytest.raises(ValueError):compose_config(overrides=[override])

def test_split(tmp_path):
    a=[{'scene_token':'a','log_id':'l1'}];b=[{'scene_token':'b','log_id':'l1'}]
    with pytest.raises(ValueError,match='overlap'):validate_splits({'train':a,'val':b})
    p=tmp_path/'a.jsonl';p.write_text(json.dumps(a[0])+'\n'+json.dumps(a[0]))
    with pytest.raises(ValueError,match='Duplicate'):read_manifest(p)

def test_preflight_missing(tmp_path):
    from preflight import preflight
    result=preflight('warmup')
    assert result['status']=='failed'
    assert any('SIMWAM_IL_CHECKPOINT' in e for e in result['errors'])
    assert not any('SIMWAM_KF_CHECKPOINT' in e for e in result['errors'])

def test_no_overwrite(tmp_path):
    from simwam.kf.execution import prepare_output
    (tmp_path/'existing').write_text('keep')
    with pytest.raises(FileExistsError):prepare_output(tmp_path)

def test_strict_score_contracts():
    import numpy as np
    from simwam.kf.scoring import validate_score_inputs,reward_to_report
    poses=np.zeros((1,8,3))
    with pytest.raises(FileNotFoundError):validate_score_inputs(poses,['s'],{'s'}, {})
    with pytest.raises(ValueError,match='outside'):validate_score_inputs(poses,['s'],{'other'}, {'s':1})
    poses[0,0,0]=np.nan
    with pytest.raises(ValueError,match='numerical'):validate_score_inputs(poses,['s'],{'s'}, {'s':1})
    assert reward_to_report(.75)==75
    assert reward_to_report(0)==0
    with pytest.raises(ValueError):reward_to_report(75)

def test_factory_explicit_kf(monkeypatch):
    from types import SimpleNamespace
    import simwam.runtime_grpo as runtime
    from simwam.runtime_kf import create_simwam_kf_grpo
    captured={}
    model=SimpleNamespace(setup_kf=lambda k:captured.update(kf=k))
    def factory(**kwargs):captured.update(kwargs);return model
    monkeypatch.setattr(runtime,'create_simwam_grpo',factory)
    cfg=compose_config()
    assert create_simwam_kf_grpo(kf=cfg.kf,model_id='tiny') is model
    assert captured['kf']['future_frames']==0 and captured['model_id']=='tiny'
    bad=OmegaConf.to_container(cfg.kf);bad['unknown']=1
    with pytest.raises(ValueError,match='keys'):create_simwam_kf_grpo(kf=bad)

def test_cli_help_and_matrix_dry_run(tmp_path,monkeypatch):
    import os,sys,subprocess
    from pathlib import Path
    monkeypatch.setenv('PATH',str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('SIMWAM_KF_CHECKPOINT','/unavailable/common-C1.pt')
    scripts=['train.py','warmup.py','smoke.py','matrix.py','eval.py','preflight.py','prepare_splits.py','summarize.py','gpu_checks.py','cache_text.py']
    for name in scripts:
        subprocess.run([sys.executable,'scripts/kf/'+name,'--help'],check=True,capture_output=True,text=True)
    out=tmp_path/'matrix'
    subprocess.run(['bash','scripts/kf/run_matrix.sh','--stage','pilot','--seeds','42','--max-steps','4',
                    '--output-root',str(out),'--dry-run'],check=True,capture_output=True,text=True)
    for k in (4,8):
        for f in (0,8):
            run=out/f'k{k}_f{f}_seed42';cfg=OmegaConf.load(run/'resolved_config.yaml')
            assert cfg.grpo.sample.group_size==k and cfg.kf.future_frames==f
            for ef in (0,8):assert (run/f'eval_val_f{ef}/evaluation_plan.json').exists()
