"""Download and verify the exact pretrained checkpoints used by tabular models."""
import argparse
import hashlib
from pathlib import Path
import requests

MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / 'model'
CHECKPOINTS = {
    'TabPFN': {
        'filename': 'tabpfn-v3-classifier-v3_default.ckpt',
        'url': 'https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-classifier-v3_default.ckpt',
        'sha256': 'd0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988',
    },
    'TabICLv2': {
        'filename': 'tabicl-classifier-v2-20260212.ckpt',
        'url': 'https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt',
        'sha256': 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0',
    },
}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def default_checkpoint(name):
    return MODEL_DIRECTORY / CHECKPOINTS[name]['filename']


def ensure_checkpoint(name, requested=None):
    spec = CHECKPOINTS[name]
    path = default_checkpoint(name) if requested in (None, 'auto') else Path(requested).expanduser().resolve()
    if path.exists():
        if file_digest(path) != spec['sha256']:
            raise ValueError(f'{name} checkpoint SHA256 mismatch: {path}. Expected the requested {spec["filename"]}.')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    print(f'Downloading {name} pretrained checkpoint to {path}', flush=True)
    try:
        digest = hashlib.sha256()
        with requests.get(spec['url'], stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
                        digest.update(block)
        if digest.hexdigest() != spec['sha256']:
            raise ValueError(f'Downloaded {name} checkpoint failed SHA256 verification')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args()
    for name, spec in CHECKPOINTS.items():
        print(ensure_checkpoint(name, args.model_dir / spec['filename']), flush=True)
