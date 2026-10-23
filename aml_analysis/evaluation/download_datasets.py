#!/usr/bin/env python3
"""Download and validate the evaluation matrices using the existing GEO workflow."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'preprocess_raw_data'))
import download_geo_datasets

if __name__ == '__main__':
    download_geo_datasets.SERIES = ('GSE40279', 'GSE62298')
    if '--data-dir' not in sys.argv:
        sys.argv.extend(['--data-dir', str(BASE / 'raw')])
    download_geo_datasets.main()
