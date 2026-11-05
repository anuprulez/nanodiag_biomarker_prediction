"""Fetch genome-wide TCGA-BRCA expression for limma; validate exact request caches."""
from pathlib import Path
import argparse,json
import pandas as pd
from .expression_resources import cached_query


def main(base,offline=False):
    base=Path(base);out=base/'direction_concordance'
    for f in ['source','data','plots']:(out/f).mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(base/'data/sample_metadata.tsv',sep='\t',index_col=0)
    samples=meta.index.tolist()
    fields=json.loads((base/'source/expression_fields.json').read_text());fields=sorted(set(fields)-{'sampleID'})
    frames=[]
    for start in range(0,len(fields),1000):
     genes=fields[start:start+1000];path=out/'source'/f'expression_genome_{start:05d}.json'
     q='((fn [dataset samples probes] (fetch [{:table dataset :columns probes :samples samples}])) "TCGA.BRCA.sampleMap/HiSeqV2" '+json.dumps(samples)+' '+json.dumps(genes)+')'
     values=cached_query('https://tcga.xenahubs.net',q,path,offline)
     assert len(values)==len(genes) and all(len(v)==len(samples) for v in values)
     frames.append(pd.DataFrame(values,index=genes,columns=samples).apply(pd.to_numeric,errors='coerce'))
     print(f'Fetched {min(start+1000,len(fields))}/{len(fields)} genes',flush=True)
    matrix=pd.concat(frames);matrix.index.name='gene'
    assert matrix.index.is_unique
    matrix.to_csv(out/'data/expression_genome.tsv.gz',sep='\t')
    print(matrix.shape,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();main(args.base,args.offline)
