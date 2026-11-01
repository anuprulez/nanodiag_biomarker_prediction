"""Atomic downloads, offline caches and portable R discovery for cross-cohort analysis."""
from pathlib import Path
import gzip,hashlib,json,os,shutil,subprocess,time
from urllib.request import Request,urlopen

HGNC_URL='https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt'

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def download(url,path,offline=False):
    """Never expose truncated downloads as completed cache files."""
    path=Path(path)
    if path.is_file() and path.stat().st_size:
        if path.suffix=='.json':json.loads(path.read_text())
        return path
    if offline:raise FileNotFoundError(f'Offline resource missing: {path}\nSource: {url}')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            print(f'Downloading {url} -> {path}',flush=True)
            with urlopen(Request(url,headers={'User-Agent':'nanodiag-cross-cohort/1.0','Accept':'application/json' if path.suffix=='.json' else '*/*'}),timeout=90) as response,temporary.open('wb') as target:
                expected=response.headers.get('Content-Length');shutil.copyfileobj(response,target)
            if not temporary.stat().st_size:raise ValueError(f'Empty resource: {url}')
            if expected is not None and temporary.stat().st_size!=int(expected):raise ValueError(f'Incomplete download: {url}')
            if path.suffix=='.gz':
                with gzip.open(temporary,'rb') as source:
                    while source.read(1024*1024):pass
            if path.suffix=='.json':json.loads(temporary.read_text())
            temporary.replace(path);return path
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(1+attempt)


def platform_table(platform,path,offline=False):
    """Read only the full platform table, not the huge following GEO sample family."""
    path=Path(path)
    if path.is_file() and path.stat().st_size:return path
    prefix=platform[:-3]+'nnn' if len(platform)>6 else 'GPLnnn'
    url=f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/{platform}/soft/{platform}_family.soft.gz'
    if offline:raise FileNotFoundError(f'Offline platform table missing: {path}; source {url}')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.part')
    try:
        print(f'Downloading {platform} platform table -> {path}',flush=True)
        with urlopen(url,timeout=90) as response,gzip.GzipFile(fileobj=response) as zipped,temporary.open('w') as target:
            import io
            stream=io.TextIOWrapper(zipped);active=False;complete=False;rows=0
            for line in stream:
                if line.startswith('!platform_table_end'):complete=True;break
                if active:target.write(line);rows+=1
                if line.startswith('!platform_table_begin'):active=True
        if not complete or rows<1000:raise ValueError(f'Incomplete {platform} annotation')
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    return path


def rscript_path(explicit=None):
    selected=explicit or os.environ.get('RSCRIPT') or shutil.which('Rscript')
    if not selected:
        for parent in Path(__file__).resolve().parents:
            candidate=parent/'.analysis_envs/bc-limma/bin/Rscript'
            if candidate.is_file():selected=str(candidate);break
    if not selected:raise FileNotFoundError('Rscript not found. Set --rscript or RSCRIPT; install post_evaluation_support/environment-cross-cohort.yml.')
    selected=shutil.which(str(selected)) or str(Path(selected).expanduser().resolve())
    result=subprocess.run([selected,'-e','suppressPackageStartupMessages(library(limma)); suppressPackageStartupMessages(library(statmod)); cat(as.character(packageVersion("limma")))'],check=True,capture_output=True,text=True)
    print(f'Rscript: {selected}; limma {result.stdout.strip()}',flush=True)
    return selected


def cached_query(host,query,path,offline=False):
    import requests
    path=Path(path);query_path=path.with_name(path.name+'.query.txt')
    legacy=path.with_suffix('.query.txt')
    recorded=query_path if query_path.exists() else legacy
    # Requests are keyed by complete query, including actual sample and gene IDs.
    if path.is_file() and recorded.is_file() and recorded.read_text().strip()==query.strip():
        return json.loads(path.read_text())
    if offline:raise FileNotFoundError(f'Offline Xena query cache missing or stale: {path}')
    for attempt in range(3):
        try:
            response=requests.post(host+'/data/',data=query,timeout=90);response.raise_for_status();values=response.json();break
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            time.sleep(attempt+1)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.part');temporary.write_text(json.dumps(values));temporary.replace(path)
    query_path.write_text(query)
    return values
