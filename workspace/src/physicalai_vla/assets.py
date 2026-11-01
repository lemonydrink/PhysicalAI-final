"""Pinned model/dataset metadata acquisition; never scans credential files."""
from pathlib import Path
from .io import ROOT, config, digest, write_json


def download_model():
    from huggingface_hub import snapshot_download
    c = config()
    target = ROOT / 'models/base'
    snapshot_download(c['model_repo'], revision=c['model_revision'], local_dir=target,
                      allow_patterns=['*.json', '*.txt', '*.safetensors'])
    weight = target / 'model.safetensors'
    if digest(weight) != c['model_sha256']:
        raise ValueError('base weight digest mismatch')
    write_json(target / 'provenance.json', {'repo': c['model_repo'], 'revision': c['model_revision'], 'sha256': digest(weight)})
    return target


def download_metadata():
    from huggingface_hub import snapshot_download
    c = config()
    return Path(snapshot_download(c['dataset_repo'], repo_type='dataset', revision=c['dataset_revision'],
               local_dir=ROOT / 'data/raw', allow_patterns=['meta/**']))
