"""Export a self-contained inference bundle, without optimizer/training data."""
from __future__ import annotations
import shutil
from pathlib import Path
from .io import ROOT,config,read_json,write_json,digest


def export_bundle(output,adapter=None,profile=None):
    output=Path(output)
    if output.exists():raise FileExistsError('bundle output must be new')
    if digest(ROOT/'models/base/model.safetensors')!=config()['model_sha256']:
        raise ValueError('base weight checksum mismatch')
    if adapter and not (Path(adapter)/'adapter_model.safetensors').is_file():raise FileNotFoundError('adapter incomplete')
    output.mkdir(parents=True)
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for p in (ROOT/'models/base').iterdir():
        if p.is_file() and p.suffix in ('.safetensors','.json','.txt'):copy(p,output/'models/base'/p.name)
    if adapter:
        for name in ('adapter_model.safetensors','adapter_config.json'):
            copy(Path(adapter)/name,output/'adapter'/name)
    shutil.copytree(ROOT/'src/physicalai_vla',output/'src/physicalai_vla',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'configs',output/'configs')
    copy(ROOT/'uv.lock',output/'uv.lock');copy(ROOT/'pyproject.toml',output/'pyproject.toml')
    copy(ROOT/'data/tasks.resolved.json',output/'data/tasks.resolved.json')
    if profile:write_json(output/'configs/inference.json',profile)
    files=[{'path':str(p.relative_to(output)),'sha256':digest(p),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()]
    write_json(output/'bundle.json',{'schema_version':1,'base_repo':config()['model_repo'],'base_revision':config()['model_revision'],
                                  'adapter':bool(adapter),'inference_profile':read_json(output/'configs/inference.json'),'files':files},exclusive=True)
    return output


def verify_bundle(path):
    path=Path(path);m=read_json(path/'bundle.json')
    for f in m['files']:
        p=(path/f['path']).resolve()
        if not p.is_relative_to(path.resolve()) or digest(p)!=f['sha256']:raise ValueError(f'bad bundle file: {f["path"]}')
    return m
