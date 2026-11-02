"""Download-to-figure orchestration shared by both post-evaluation entry points."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
import pandas as pd
from .expression_resources import download,platform_table,rscript_path,digest,HGNC_URL

HERE=Path(__file__).resolve().parent


def add_arguments(parser):
    parser.add_argument('--post-evaluation-analysis',choices=['all','cross-cohort'],default='all',
                        help='Default: existing post-evaluation plus cross-cohort plots; cross-cohort: run only download-to-plot analyses')
    parser.add_argument('--rscript',help='Rscript executable with limma and statmod; defaults to RSCRIPT/PATH or local analysis environment')
    parser.add_argument('--refresh-expression',action='store_true',help='Rebuild expression matrices and refit limma from cached/downloaded sources')


def prepare_fixed_inputs(root,config,out,offline=False):
    from .analysis import load_panel,read_original,original_metadata
    disease='BC' if root.name=='breast_cancer_analysis' else 'AML'
    panel,_,internal,_=load_panel(Path(config.p_base),disease,33 if disease=='BC' else 17)
    (out/'data').mkdir(parents=True,exist_ok=True)
    fixed=out/'data/fixed_candidate_panel.tsv'
    if fixed.exists():
        old=pd.read_csv(fixed,sep='\t')
        if set(old.CpG_ID)!=set(panel.CpG_ID):raise ValueError('Saved fixed-panel identities differ from selected frozen CpGs')
        np.testing.assert_allclose(old.set_index('CpG_ID').loc[panel.CpG_ID,internal],panel[internal],atol=1e-12)
    else:panel.to_csv(fixed,sep='\t',index=False)
    annotation_path=out/'data/candidate_genomic_annotation.tsv'
    if not annotation_path.exists():
        sources=[root/'raw/GPL13534_probe_annotation.tsv',root/'evaluation/raw/GPL13534_probe_annotation.tsv']
        source=next((p for p in sources if p.exists()),None)
        if source is None:source=platform_table('GPL13534',out/'methylation_expression/source/GPL13534_probe_annotation.tsv',offline)
        ann=pd.read_csv(source,sep='\t',dtype=str).rename(columns={'ID':'CpG_ID'}).drop_duplicates('CpG_ID').set_index('CpG_ID')
        selected=ann.loc[panel.CpG_ID].reset_index()
        if 'UCSC_RefGene_Group' not in selected:
            context=pd.read_csv(HERE/'resources/GPL13534_context.tsv.gz',sep='\t',dtype=str).rename(columns={'ID':'CpG_ID'})
            selected=selected.merge(context,on='CpG_ID',validate='one_to_one')
        selected.to_csv(annotation_path,sep='\t',index=False)
    annotation=pd.read_csv(annotation_path,sep='\t')
    if set(annotation.CpG_ID)!=set(panel.CpG_ID):raise ValueError('Genomic annotation does not match fixed panel')
    path=out/'data/original_candidate_beta.tsv.gz'
    if path.exists():original=pd.read_csv(path,sep='\t',index_col=0)
    else:
        original=read_original(config.p_arrays,set(panel.CpG_ID))
        original.to_csv(path,sep='\t')
    meta=original_metadata(config,original)
    meta.to_csv(out/'data/original_patient_annotations.tsv',sep='\t')
    delta=original.loc[meta.label.eq(1),panel.CpG_ID].mean()-original.loc[meta.label.eq(0),panel.CpG_ID].mean()
    np.testing.assert_allclose(panel[internal],panel.CpG_ID.map(delta),atol=2e-9,rtol=0)
    return disease


def cached_step(name,inputs,outputs,stamp,action,refresh=False):
    """Reuse only when both content-addressed inputs AND outputs still match."""
    token={str(p):digest(p) for p in inputs}
    old=json.loads(stamp.read_text()) if stamp.exists() else {}
    complete=all(p.is_file() for p in outputs)
    actual={str(p):digest(p) for p in outputs} if complete else {}
    if not refresh and old.get('inputs')==token and complete and old.get('outputs')==actual:
        print(f'{name}: verified cache',flush=True);return 'verified cache'
    print(f'{name}: running',flush=True);action()
    missing=[str(p) for p in outputs if not p.is_file()]
    if missing:raise FileNotFoundError(f'{name}: missing outputs {missing}')
    stamp.parent.mkdir(parents=True,exist_ok=True)
    stamp.write_text(json.dumps(dict(inputs=token,outputs={str(p):digest(p) for p in outputs}),indent=2))
    return 'executed'


def _run(root,config,rscript=None,offline=False,refresh=False):
    root=Path(root).resolve();out=Path(config.p_base).resolve()/'accumulated_results/post_evaluation'
    executable=rscript_path(rscript)
    disease=prepare_fixed_inputs(root,config,out,offline)
    base=out/'methylation_expression';steps={};(base/'source').mkdir(parents=True,exist_ok=True)
    print(f'{disease}: cross-cohort download-to-plot analysis -> {base}',flush=True)
    if disease=='BC':
        from .fetch_bc_expression import main as fetch
        from .fetch_bc_expression_genome import main as genome
        from .bc_direction_concordance import main as concordance
        from .bc_cross_cohort_comparison import main as plot
        fetch(base,offline)
        source=base/'direction_concordance/source';data=base/'direction_concordance/data'
        source.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
        fields=sorted(set(json.loads((base/'source/expression_fields.json').read_text()))-{'sampleID'})
        # Cached JSON chunks are also inputs: editing/removing them invalidates the matrix.
        query_inputs=[source/f'expression_genome_{i:05d}.json' for i in range(0,len(fields),1000)]
        matrix=data/'expression_genome.tsv.gz'
        inputs=[base/'data/sample_metadata.tsv',base/'source/expression_fields.json',HERE/'fetch_bc_expression_genome.py',HERE/'expression_resources.py']
        if all(p.exists() for p in query_inputs):inputs+=query_inputs
        steps['genome']=cached_step('BC genome-wide expression',inputs,[matrix],source/'genome_cache.json',lambda:genome(base,offline),refresh)
        de=[data/f'all_genes_{model}_limma.tsv' for model in ['paired','blocked']]
        steps['limma']=cached_step('BC paired and patient-blocked limma',[matrix,base/'data/sample_metadata.tsv',HERE/'bc_differential_expression.R'],
                                  [*de,data/'limma_methods.txt'],source/'limma_cache.json',
                                  lambda:subprocess.run([executable,str(HERE/'bc_differential_expression.R'),str(base)],check=True),refresh)
        concordance(base);plot(base)
        from .validate_bc_cross_cohort import main as validate
        validate(out)
        figure=base/'cross_cohort_comparison/plots/GSE66695_delta_beta_vs_TCGA_expression_log2FC'
    else:
        from .aml_cross_cohort_comparison import main as plot,prepare_expression,symbol_mapper
        from .validate_aml_cross_cohort import main as validate
        target=base/'cross_cohort_comparison';source=target/'source';data=target/'data'
        for folder in [source,data,target/'plots']:folder.mkdir(parents=True,exist_ok=True)
        download('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE15nnn/GSE15061/matrix/GSE15061_series_matrix.txt.gz',source/'GSE15061_series_matrix.txt.gz',offline)
        platform_table('GPL570',source/'GPL570_probe_annotation.tsv',offline)
        hgnc=download(HGNC_URL,source/'HGNC_complete_set.tsv',offline)
        inputs=[source/'GSE15061_series_matrix.txt.gz',source/'GPL570_probe_annotation.tsv',hgnc,HERE/'aml_cross_cohort_comparison.py']
        outputs=[data/'GSE15061_DS_probe_expression.tsv.gz',data/'GPL570_unambiguous_probe_gene_mapping.tsv',data/'GSE15061_sample_metadata.tsv',data/'GSE15061_all_870_records_audit.tsv',data/'group_by_center_counts.tsv']
        steps['expression']=cached_step('AML unique DS samples and probe mapping',inputs,outputs,source/'expression_cache.json',lambda:prepare_expression(target,symbol_mapper(hgnc)),refresh)
        de=[data/f'all_genes_{model}_limma.tsv' for model in ['center_adjusted','unadjusted']]
        steps['limma']=cached_step('AML center-adjusted and unadjusted limma',[*outputs[:3],HERE/'aml_expression_limma.R'],
                                  [*de,data/'GSE15061_normalized_gene_expression.tsv.gz',data/'design_center_adjusted.tsv',data/'design_unadjusted.tsv'],source/'limma_cache.json',
                                  lambda:subprocess.run([executable,str(HERE/'aml_expression_limma.R'),str(target)],check=True),refresh)
        plot(out,hgnc_path=hgnc);validate(out)
        figure=target/'plots/AML_17_CpGs_methylation_vs_GSE15061_expression'
    from .expression_publication import run as publication_figures
    extra_figures=publication_figures(out,disease)
    artifacts=[Path(str(figure)+'.'+ext) for ext in ['pdf','svg','png']]+extra_figures
    for path in artifacts:
        if not path.is_file() or not path.stat().st_size:raise FileNotFoundError(path)
    manifest=dict(status='complete',disease=disease,stage='post-evaluation',steps=steps,rscript=executable,
                  offline=offline,original_methylation=True,outputs={str(p):digest(p) for p in artifacts},
                  input_hashes={str(out/'data'/name):digest(out/'data'/name) for name in ['fixed_candidate_panel.tsv','candidate_genomic_annotation.tsv','original_candidate_beta.tsv.gz']})
    (base/'pipeline_manifest.json').write_text(json.dumps(manifest,indent=2))
    report=out/'REPORT.md'
    marker='\n\n## Cross-cohort methylation/expression'
    previous=report.read_text().split(marker)[0] if report.exists() else ''
    report.write_text(previous+marker+'\n\n[Analysis report](methylation_expression/cross_cohort_comparison/REPORT.md). '
                      'The complete download-to-plot pipeline and validation status are recorded in methylation_expression/pipeline_manifest.json.\n')
    print(f'{disease}: cross-cohort analysis and validation complete -> {figure}.pdf',flush=True)
    return manifest


def run(root,config,rscript=None,offline=False,refresh=False):
    status=Path(config.p_base).resolve()/'accumulated_results/post_evaluation/methylation_expression/pipeline_manifest.json'
    status.parent.mkdir(parents=True,exist_ok=True)
    status.write_text(json.dumps({'status':'running','stage':'post-evaluation'}))
    try:
        return _run(root,config,rscript,offline,refresh)
    except Exception as error:
        status.write_text(json.dumps({'status':'failed','stage':'post-evaluation','error':str(error)},indent=2))
        raise


if __name__=='__main__':
    from omegaconf import OmegaConf
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--offline',action='store_true');add_arguments(parser);args=parser.parse_args()
    run(args.root,OmegaConf.load(args.config),args.rscript,args.offline,args.refresh_expression)
