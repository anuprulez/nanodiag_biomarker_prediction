#!/usr/bin/env python3
"""Download published external breast cancer/control betas and platform annotation."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
from geo_common import BASE, SOURCES, PLATFORM_URL, digest, annotations


def validate_gzip(path):
    with gzip.open(path, 'rb') as source:
        while source.read(4*1024*1024):
            pass


def download(url, path, force=False):
    if path.exists() and not force:
        validate_gzip(path)
        print(f'Validated cached file: {path}', flush=True)
        return
    part = path.with_suffix(path.suffix + '.part')
    if force:
        part.unlink(missing_ok=True)
    subprocess.run(['curl','-fL','--retry','3','--continue-at','-',url,'-o',str(part)], check=True)
    validate_gzip(part)
    part.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=BASE / 'raw')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url in SOURCES.items():
        path = args.raw_dir / name
        download(url,path,args.force)
        manifest.append({'url':url,'file':name,'bytes':path.stat().st_size,'sha256':digest(path)})
        (args.raw_dir/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cache = args.raw_dir/'GPL13534_probe_annotation.tsv'
    if not cache.exists() or args.force:
        existing = BASE.parents[1]/'aml_analysis/raw/GPL13534_probe_annotation.tsv'
        if existing.exists() and not args.force:
            shutil.copy2(existing,cache)
        else:
            soft = args.raw_dir/'GPL13534_family.soft.gz'
            download(PLATFORM_URL,soft,args.force)
            temp = cache.with_suffix('.tsv.part')
            with gzip.open(soft,'rt') as source,temp.open('w') as out:
                for line in source:
                    if line.strip()=='!platform_table_begin':break
                else:raise ValueError('Platform table absent')
                reader=csv.DictReader(source,delimiter='\t')
                fields=['ID','UCSC_RefGene_Name','CHR','MAPINFO']
                if not set(fields)<=set(reader.fieldnames or []):raise ValueError('Platform schema mismatch')
                writer=csv.DictWriter(out,fieldnames=fields,delimiter='\t',lineterminator='\n')
                writer.writeheader()
                for row in reader:
                    if row['ID']=='!platform_table_end':break
                    writer.writerow({key:row[key] for key in fields})
                else:raise ValueError('Truncated platform table')
            temp.replace(cache)
    annotations(args.raw_dir)
    manifest.append({'url':PLATFORM_URL,'file':cache.name,'bytes':cache.stat().st_size,'sha256':digest(cache),
                     'annotation':'GPL13534 UCSC_RefGene_Name, legacy hg19'})
    (args.raw_dir/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
