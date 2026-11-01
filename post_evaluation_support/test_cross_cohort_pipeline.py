"""Failure handling, cache invalidation, offline behavior and real stage routing."""
from pathlib import Path
import gzip,io,json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
from omegaconf import OmegaConf
from . import expression_resources as resources
from .cross_cohort_pipeline import cached_step


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_truncated_download_never_becomes_cache(self):
        class Response(io.BytesIO):headers={'Content-Length':'20'}
        target=self.root/'resource.txt'
        with patch.object(resources,'urlopen',side_effect=lambda *a,**kw:Response(b'partial')),patch.object(resources.time,'sleep'):
            with self.assertRaisesRegex(ValueError,'Incomplete'):resources.download('https://example.invalid/data',target)
        self.assertFalse(target.exists());self.assertFalse(target.with_suffix('.txt.part').exists())

    def test_offline_uses_cache_and_missing_resource_fails(self):
        p=self.root/'ok.json';p.write_text('{"ok":true}')
        with patch.object(resources,'urlopen',side_effect=AssertionError('Network forbidden')):
            self.assertEqual(resources.download('unused',p,offline=True),p)
            with self.assertRaises(FileNotFoundError):resources.download('unused',self.root/'missing',offline=True)

    def test_platform_stops_at_table_end_instead_of_entire_sample_family(self):
        # Trailing gzip stream is intentionally incomplete: platform extraction
        # must succeed before attempting to parse following sample records.
        content='!platform_table_begin\nID\tGene Symbol\n'+''.join(f'p{i}\tG{i}\n' for i in range(1001))+'!platform_table_end\n'
        response=io.BytesIO(gzip.compress(content.encode())+b'invalid unrelated sample family')
        with patch.object(resources,'urlopen',return_value=response):
            path=resources.platform_table('GPL570',self.root/'platform.tsv')
        self.assertEqual(len(path.read_text().splitlines()),1002)

    def test_query_cache_must_match_sample_ids_even_at_same_dimensions(self):
        p=self.root/'values.json';p.write_text('[[1,2]]');p.with_name(p.name+'.query.txt').write_text('old sample IDs')
        with self.assertRaisesRegex(FileNotFoundError,'stale'):
            resources.cached_query('https://unused','new sample IDs',p,offline=True)

    def test_editing_inputs_or_outputs_invalidates_success_cache(self):
        inp=self.root/'in';out=self.root/'out';stamp=self.root/'stamp';inp.write_text('one');calls=[]
        def action():calls.append(True);out.write_text(inp.read_text())
        self.assertEqual(cached_step('step',[inp],[out],stamp,action),'executed')
        self.assertEqual(cached_step('step',[inp],[out],stamp,action),'verified cache')
        inp.write_text('two');cached_step('step',[inp],[out],stamp,action)
        out.write_text('corrupted');cached_step('step',[inp],[out],stamp,action)
        self.assertEqual(len(calls),3)

    def test_failed_step_is_not_marked_complete(self):
        inp=self.root/'in';inp.write_text('input');stamp=self.root/'stamp'
        with self.assertRaises(RuntimeError):
            cached_step('step',[inp],[self.root/'out'],stamp,lambda:(_ for _ in ()).throw(RuntimeError('failure')))
        self.assertFalse(stamp.exists())

    def test_failed_pipeline_replaces_stale_success_manifest(self):
        from . import cross_cohort_pipeline
        config=SimpleNamespace(p_base=str(self.root))
        status=self.root/'accumulated_results/post_evaluation/methylation_expression/pipeline_manifest.json'
        status.parent.mkdir(parents=True);status.write_text('{"status":"complete"}')
        with patch.object(cross_cohort_pipeline,'_run',side_effect=RuntimeError('download failed')):
            with self.assertRaises(RuntimeError):cross_cohort_pipeline.run(self.root,config)
        self.assertEqual(json.loads(status.read_text())['status'],'failed')

    def test_cross_cohort_stage_routes_without_running_classifiers(self):
        from . import analysis,cross_cohort_pipeline
        config=self.root/'config.yaml';OmegaConf.save(OmegaConf.create({'p_base':str(self.root)}),config)
        for disease in ['breast_cancer_analysis','aml_analysis']:
            argv=['post_evaluation.py','--config',str(config),'--post-evaluation-analysis','cross-cohort','--offline','--rscript','custom-R','--refresh-expression']
            with patch('sys.argv',argv),patch.object(cross_cohort_pipeline,'run') as run,patch.object(analysis,'panel_comparisons',side_effect=AssertionError('Unrelated classifier analysis')):
                analysis.main(self.root/disease)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0],self.root/disease)
            self.assertEqual(run.call_args.args[2:],('custom-R',True,True))

if __name__=='__main__':unittest.main()
