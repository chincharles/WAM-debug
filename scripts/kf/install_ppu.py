#!/usr/bin/env python3
"""Adapted from the user's VLA-RFT-navsim PPU installer.
Isolate Python deps while retaining the image's original PPU framework files.

Run with the PPU image's Python, never with a generic replacement torch wheel.
Vendor packages are linked into a clean venv, not copied or installed from PyPI.
"""
import argparse
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
CORE={'torch','torchvision','torchaudio','triton'}
FORBIDDEN=CORE|{'flash-attn','xformers','vllm','deepspeed','acext','deep-ep'}
# The supplied image contains an optional DALI package whose metadata is
# inconsistent with the isolated Python dependency set (missing astunparse,
# dm-tree and gast, plus incompatible packaging/six bounds). NAVSIM and
# nuPlan do not import DALI, so do not bridge this unrelated image package
# into the clean venv. Accelerator packages used by torch remain protected.
IGNORED_VENDOR_PREFIXES=('nvidia-dali',)
NUPLAN_PIN='ce3c323af01c0d7ec5672f7832ef53f9c679aab0'

def canonical(name):
    return re.sub(r'[-_.]+','-',name).lower()

def protected(name):
    name=canonical(name)
    return name in FORBIDDEN or name.startswith(('nvidia-','accl','pccl','ppu-'))

def guard_plan(report):
    bad=[r['metadata']['name'] for r in report.get('install',[]) if protected(r['metadata']['name'])]
    if bad:raise RuntimeError(f'Refusing to replace/install vendor accelerator packages: {bad}')

def inventory():
    records={}
    for dist in metadata.distributions():
        name=canonical(dist.metadata['Name'])
        if name.startswith(IGNORED_VENDOR_PREFIXES):
            continue
        if name not in CORE and not name.startswith(('nvidia-', 'accl', 'pccl', 'ppu-')):continue
        # Only site-packages entries are bridged. No bin scripts or arbitrary .pth.
        roots=set()
        for file in dist.files or []:
            part=Path(str(file)).parts[0]
            if part in ('.','..') or part.endswith('.pth'):continue
            path=Path(dist.locate_file(part)).absolute()
            if path.exists():roots.add(str(path))
        if not roots:raise RuntimeError(f'No installed file inventory for {name}')
        records[name]=dict(version=dist.version,paths=sorted(roots))
    if not {'torch','torchvision'}.issubset(records):raise RuntimeError('Use the PPU image Python containing torch and torchvision')
    return records

def link_packages(records,site):
    site=Path(site);site.mkdir(parents=True,exist_ok=True)
    linked={}
    for record in records.values():
        for source in record['paths']:
            src=Path(source);dst=site/src.name
            if dst.name in linked and linked[dst.name]!=source:raise RuntimeError(f'Conflicting vendor package: {dst.name}')
            linked[dst.name]=source
            if dst.exists() or dst.is_symlink():
                if not dst.is_symlink() or dst.resolve()!=src.resolve():raise RuntimeError(f'Vendor link replaced: {dst}')
            else:dst.symlink_to(src,target_is_directory=src.is_dir())

def validate_base(expected_devices):
    if sys.platform!='linux' or sys.version_info[:2]!=(3,12):raise RuntimeError('PPU profile requires the supplied Linux Python 3.12 image')
    import torch
    import torchvision
    if torch.__version__.split('+')[0]!='2.6.0' or torchvision.__version__.split('+')[0]!='0.21.0':
        raise RuntimeError('This PPU profile targets the supplied torch 2.6.0 / torchvision 0.21.0 image')
    if not torch.cuda.is_available() or torch.cuda.device_count()<expected_devices:
        raise RuntimeError('PPU devices are not visible through the vendor torch.cuda interface')
    names=[torch.cuda.get_device_name(i) for i in range(expected_devices)]
    if any('PPU' not in name.upper() for name in names):raise RuntimeError(f'Expected PPU devices, found {names}')
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
        torchvision=torchvision.__version__,torch_file=torch.__file__,torchvision_file=torchvision.__file__,
        cuda_build=torch.version.cuda,devices=names,nccl_available=torch.distributed.is_nccl_available())

def nuplan_wheel(py,env):
    deps=Path(env['KF_SERVER_WORK'])/'deps';deps.mkdir(parents=True,exist_ok=True)
    supplied=os.environ.get('KF_NUPLAN_WHEEL')
    if supplied:
        wheel=Path(supplied).expanduser().resolve()
        if not wheel.is_file() or wheel.name != 'nuplan_devkit-1.2.0-py3-none-any.whl':
            raise RuntimeError(f'KF_NUPLAN_WHEEL is not a nuPlan v1.2 wheel: {wheel}')
        import zipfile
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None: raise RuntimeError('Incomplete supplied nuPlan wheel')
        return wheel
    with (deps/'nuplan-build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        src=deps/'nuplan-v1.2';wheel_dir=deps/'nuplan-wheels'
        if not src.exists():
            subprocess.run(['git','init','-q',str(src)],check=True)
            subprocess.run(['git','-C',str(src),'remote','add','origin','https://github.com/motional/nuplan-devkit.git'],check=True)
        head=subprocess.run(['git','-C',str(src),'rev-parse','HEAD'],capture_output=True,text=True)
        if head.returncode:
            subprocess.run(['git','-C',str(src),'fetch','--depth','1','origin',NUPLAN_PIN],check=True)
            subprocess.run(['git','-C',str(src),'checkout','--detach','FETCH_HEAD'],check=True)
        actual=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(src),'status','--porcelain','--untracked-files=no'],text=True).strip()
        if actual!=NUPLAN_PIN or dirty:raise RuntimeError('nuPlan checkout differs from the required untouched v1.2 source')
        wheels=list(wheel_dir.glob('nuplan_devkit-1.2.0-*.whl'))
        if not wheels:
            subprocess.run([str(py),'setup.py','build','--build-base',str(deps/'nuplan-build'),
                'bdist_wheel','--dist-dir',str(wheel_dir)],cwd=src,env=env,check=True)
            wheels=list(wheel_dir.glob('nuplan_devkit-1.2.0-*.whl'))
        if len(wheels)!=1:raise RuntimeError('Expected one pinned nuPlan wheel')
        import zipfile
        with zipfile.ZipFile(wheels[0]) as archive:
            if archive.testzip() is not None:raise RuntimeError('Incomplete nuPlan wheel; preserve/remove it before retrying')
        return wheels[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--venv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--devices',type=int,default=2);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base=validate_base(a.devices);records=inventory()
    os.environ.setdefault('KF_SERVER_WORK', str(out.resolve()))
    root=Path(a.venv).absolute();py=root/'bin/python';marker=root/'ppu-vendor-links.json'
    if marker.exists():
        if json.loads(marker.read_text())!=records:raise RuntimeError('PPU image package versions/paths changed since this run')
    elif root.exists():raise FileExistsError(f'Unmarked partial environment {root}; preserve/rename it and retry')
    else:
        subprocess.run([sys.executable,'-m','venv',str(root)],check=True)
        site=Path(subprocess.check_output([str(py),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())
        link_packages(records,site)
        marker.write_text(json.dumps(records,indent=2))
    constraints=out/'vendor-constraints.txt'
    constraints.write_text(''.join(f'{name}=={r["version"]}\n' for name,r in sorted(records.items())))
    (out/'ppu-base.json').write_text(json.dumps(base,indent=2))
    # Keep the current interpreter's SDK/linker environment, but isolate Python packages.
    env=dict(os.environ);env.pop('PYTHONHOME',None)
    env.update(VIRTUAL_ENV=str(root),PATH=str(root/'bin')+os.pathsep+env['PATH'],
        SIMWAM_RUNTIME='ppu',PYTHONNOUSERSITE='1',PYTHONPATH=os.pathsep.join(map(str,[
            REPO/'configs/server/compat',REPO/'src',REPO/'navsim'])))
    env['PIP_CONSTRAINT']=str(constraints)
    report=out/'pip-plan.json'
    command=[str(py),'-m','pip','install','-r',str(REPO/'configs/server/requirements-ppu.txt'),'-c',str(constraints)]
    subprocess.run([*command,'--dry-run','--report',str(report)],env=env,check=True)
    plan=json.loads(report.read_text());guard_plan(plan)
    # Install exactly the artifacts that were inspected, without a second resolution.
    urls=[r['download_info']['url'] for r in plan.get('install',[])]
    for i,r in enumerate(plan.get('install',[])):
        info=r['download_info']
        if 'vcs_info' in info:
            vcs=info['vcs_info'];urls[i]=f'{vcs["vcs"]}+{info["url"]}@{vcs["commit_id"]}'
    if urls:subprocess.run([str(py),'-m','pip','install','--no-deps',*urls],env=env,check=True)
    wheel=nuplan_wheel(py,env)
    subprocess.run([str(py),'-m','pip','install','--no-deps',str(wheel)],env=env,check=True)
    subprocess.run([str(py),'-m','pip','check'],env=env,check=True)
    check='import torch,torchvision,json; print(json.dumps([torch.__file__,torchvision.__file__,torch.__version__,torchvision.__version__]))'
    loaded=json.loads(subprocess.check_output([str(py),'-c',check],env=env,text=True).strip().splitlines()[-1])
    expected=[base['torch_file'],base['torchvision_file'],base['torch'],base['torchvision']]
    if [str(Path(x).resolve()) for x in loaded[:2]]!=[str(Path(x).resolve()) for x in expected[:2]] or loaded[2:]!=expected[2:]:
        raise RuntimeError('Vendor torch/torchvision changed or was shadowed')
    subprocess.run([str(py),str(REPO/'scripts/kf/check_server.py'),'--output',str(out/'ppu-imports.json')],env=env,check=True)
    print(f'PPU isolated environment ready: {root}',flush=True)

if __name__=='__main__':main()
