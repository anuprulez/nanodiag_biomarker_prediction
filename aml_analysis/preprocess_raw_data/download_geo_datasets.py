#!/usr/bin/env python3
"""Download published beta-value matrices, validate gzip, and record SHA256."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
from geo_common import BASE, SERIES, matrix_path, read_header


def validate(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    with gzip.open(path, 'rt') as handle:
        read_header(handle)
        for _ in handle:  # Checks gzip CRC and rejects truncated transfers.
            pass
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=BASE / 'raw/raw')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for series in SERIES:
        url = (f'https://ftp.ncbi.nlm.nih.gov/geo/series/{series[:-3]}nnn/'
               f'{series}/matrix/{series}_series_matrix.txt.gz')
        target = matrix_path(args.data_dir, series)
        if not target.exists() or args.force:
            part = target.with_suffix(target.suffix + '.part')
            for attempt in range(3):
                try:
                    print(f'Downloading {series}: {url}', flush=True)
                    request = urllib.request.Request(url, headers={'User-Agent': 'nanodiag-aml/1.0'})
                    with urllib.request.urlopen(request, timeout=180) as response, part.open('wb') as out:
                        shutil.copyfileobj(response, out, length=1024 * 1024)
                    checksum = validate(part)
                    part.replace(target)
                    break
                except Exception as error:
                    print(f'Attempt {attempt + 1} failed: {error}', flush=True)
                    part.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        else:
            checksum = validate(target)
            print(f'Using validated cached file: {target}', flush=True)
        manifest.append(dict(series=series, url=url, file=target.name,
                             bytes=target.stat().st_size, sha256=checksum))
        temporary = args.data_dir / 'download_manifest.json.part'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(args.data_dir / 'download_manifest.json')


if __name__ == '__main__':
    main()
