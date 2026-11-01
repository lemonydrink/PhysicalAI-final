"""Artifact IO: deterministic hashes, exclusive output and strict JSON."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def config():
    return read_json(ROOT / 'configs/project.json')
