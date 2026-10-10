import sys
from pathlib import Path
import unittest
import contextlib
import io
import json
import os
import tempfile
import types
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments3'))
from analysis_core import fixed_vertical_diagonal_kernel
from sweep_core import convert, summarize, SIZES


class SweepTests(unittest.TestCase):
    def test_reference_sampling_matches_explicit_antidiagonal_oracle(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch unavailable; run this numerical test in fyc_qwen')
        import numpy as np
        from run_sweep import reference_sampled_scores
        rng=np.random.default_rng(17)
        q=rng.normal(size=(1,2,141,4)).astype(np.float32)
        k=rng.normal(size=q.shape).astype(np.float32)
        module=types.SimpleNamespace(choose_conv_prefill_chunk_size=lambda n:1024)
        with mock.patch.dict(sys.modules,{'conv_sparse_ops':module}):
            got,meta=reference_sampled_scores(torch.tensor(q),torch.tensor(k),torch,stride=8)
        qp=np.pad(q,((0,0),(0,0),(0,1024-141),(0,0)))
        kp=np.pad(k,((0,0),(0,0),(0,1024-141),(0,0)))
        expected=np.zeros((1,2,8,8),dtype=np.float32)
        for h in range(2):
            for qr in range(18):
                logits=np.array([sum(float(qp[0,h,qr*8+7-i]@kp[0,h,kr*8+i]) for i in range(8)) /16
                                 for kr in range(qr+1)])
                p=np.exp(logits-logits.max());p/=p.sum()
                for kr,value in enumerate(p):expected[0,h,qr//16,kr//16]+=value
        np.testing.assert_allclose(got.numpy(),expected,rtol=2e-6,atol=2e-6)
        self.assertEqual(meta['q_real_blocks'],2)

    def test_cpu_selector_matches_causal_reference_and_budget(self):
        import numpy as np
        from run_sweep import cpu_ratio_mask
        scores=np.random.default_rng(42).normal(size=(2,3,29,31)).astype(np.float32)
        scores[...,0,:]=0  # Explicit ties, including causally invalid keys.
        mask=cpu_ratio_mask(scores,0.65,offset=2)
        for b in range(2):
            for h in range(3):
                for q in range(29):
                    visible=min(31,q+3)
                    count=int(np.ceil(np.float32(visible)*np.float32(0.65)))
                    expected=sorted(range(visible),key=lambda k:(-float(scores[b,h,q,k]),k))[:count]
                    self.assertEqual(np.flatnonzero(mask[b,h,q]).tolist(),sorted(expected))
                    self.assertEqual(int(mask[b,h,q].sum()),count)
                    self.assertFalse(mask[b,h,q,visible:].any())
        self.assertEqual(np.flatnonzero(mask[0,0,0]).tolist(),[0,1])
        self.assertFalse(cpu_ratio_mask(scores,0.65,offset=-30).any())
        self.assertTrue(cpu_ratio_mask(scores,1.0,causal=False).all())

    def test_model_load_ignores_scheduler_rank_env_and_restores_it(self):
        from run_sweep import single_process_load_environment
        env={'WORLD_SIZE':'1','RANK':'0','MASTER_ADDR':'fixture','MASTER_PORT':'12345',
             'CUDA_VISIBLE_DEVICES':'3,7','OTHER_SETTING':'keep'}
        original=env.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            with single_process_load_environment(env):
                # This is the HF 4.51 auto-TP trigger reported in the failure.
                self.assertFalse(int(env.get('WORLD_SIZE','0')))
                self.assertNotIn('RANK',env)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3,7')
            self.assertEqual(env,original)
            with self.assertRaises(RuntimeError):
                with single_process_load_environment(env):raise RuntimeError('test load failure')
            self.assertEqual(env,original)
        real_torchrun={'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0'}
        with self.assertRaises(ValueError):
            with single_process_load_environment(real_torchrun):pass
        self.assertEqual(real_torchrun,{'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0'})

    def test_model_load_manifest_repair_rejects_saved_losses(self):
        from run_sweep import ensure_run_manifest, MODEL_LOAD_BUG_HASHES, ROOT
        import run_sweep
        key=str(Path(run_sweep.__file__).relative_to(ROOT))
        old={'arguments':{'ratio':0.65},'source_sha256':{key:sorted(MODEL_LOAD_BUG_HASHES)[0]}}
        current={'arguments':{'ratio':0.65},'source_sha256':{key:'fixed source'}}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);manifest=out/'run_manifest.json';manifest.write_text(json.dumps(old))
            with contextlib.redirect_stdout(io.StringIO()):ensure_run_manifest(out,current,True)
            self.assertTrue((out/'run_manifest.before_model_load_fix.json').is_file())
            self.assertEqual(json.loads(manifest.read_text()),current)
            manifest.write_text(json.dumps(old))
            case=out/'qa_2/s0';case.mkdir(parents=True);(case/'fixed_1x1.json').write_text('{}')
            with self.assertRaises(ValueError):ensure_run_manifest(out,current,True)

    def test_path_fix_manifest_migration_is_narrow(self):
        from prepare_data import ensure_manifest, PATH_BUG_SOURCE_HASHES
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);key='prepare_data.py'
            previous={'seed':42,'generator_sources':{key:sorted(PATH_BUG_SOURCE_HASHES)[0],'qa.py':'same'}}
            current={'seed':42,'generator_sources':{key:'repaired','qa.py':'same'}}
            (output/'preparation_manifest.json').write_text(json.dumps(previous))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(ensure_manifest(output,current,key))
            self.assertEqual(json.loads((output/'preparation_manifest.before_path_fix.json').read_text()),previous)
            self.assertEqual(json.loads((output/'preparation_manifest.json').read_text()),current)
            with self.assertRaises(ValueError):ensure_manifest(output,{**current,'seed':43},key)

    def test_recover_preserves_source_and_rejects_conflicting_destination(self):
        from prepare_data import recover_misplaced_raw
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'misplaced.jsonl';dest=Path(tmp)/'correct/validation.jsonl'
            source.write_text('source fixture')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(recover_misplaced_raw(source,dest))
            self.assertEqual(source.read_text(),dest.read_text())
            dest.write_text('different fixture')
            with self.assertRaises(ValueError):recover_misplaced_raw(source,dest)

    def test_generator_in_another_cwd_uses_absolute_paths(self):
        import prepare_data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scripts=root/'eval/RULER/scripts';data=scripts/'data';synthetic=data/'synthetic'
            synthetic.mkdir(parents=True)
            fake_script=root/'experiments5/kernel_task_sweep/prepare_data.py'
            fake_script.parent.mkdir(parents=True);fake_script.write_text('test source')
            fake_script.with_name('sweep_core.py').write_text('test converter')
            for path in [scripts/'synthetic.yaml',data/'template.py',synthetic/'constants.py',
                         synthetic/'niah.py',synthetic/'freq_words_extraction.py']:
                path.write_text('fixture')
            # A real child process has the generator cwd and checks its path arguments.
            (synthetic/'qa.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--save_dir');p.add_argument('--save_name');p.add_argument('--tokenizer_path')
a,_=p.parse_known_args()
assert Path(a.save_dir).is_absolute()
assert Path(a.tokenizer_path).is_absolute()
assert Path.cwd()==Path(__file__).resolve().parents[1]
out=Path(a.save_dir)/a.save_name/'validation.jsonl'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(''.join(json.dumps({'input':'fixture prompt','outputs':['fixture answer'],'index':i})+'\\n' for i in range(2)))
''')
            model=root/'model';model.mkdir();(model/'config.json').write_text(json.dumps({'model_type':'llama','max_position_embeddings':131072}))
            tokenizer=types.SimpleNamespace(encode=lambda *args,**kwargs:[1])
            transformers=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args,**kwargs:tokenizer))
            yaml=types.SimpleNamespace(safe_load=lambda text:{'qa_2':{'task':'qa','args':{'dataset':'hotpotqa'}}})
            def fake_run_path(path):
                return {'Templates':{'meta-llama3':'{task_template}'}} if Path(path).name=='template.py' else {
                    'TASKS':{'qa':{'template':'test template','tokens_to_generate':32}}}
            prior=Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(sys,'argv',
                    ['prepare_data.py','--model','model','--output','results','--tasks','qa_2']),\
                    mock.patch.object(prepare_data,'ROOT',root),mock.patch.object(prepare_data,'__file__',str(fake_script)),\
                    mock.patch.object(prepare_data.runpy,'run_path',side_effect=fake_run_path),\
                    mock.patch.dict(sys.modules,{'transformers':transformers,'yaml':yaml}):
                    prepare_data.main()
            finally:os.chdir(prior)
            self.assertTrue((root/'results/raw/qa_2/validation.jsonl').is_file())
            self.assertTrue((root/'results/data_ready.json').is_file())
            self.assertFalse((data/'results').exists())

    def test_kernel_support_and_overlap(self):
        for size in SIZES:
            k=fixed_vertical_diagonal_kernel(size)
            self.assertEqual(k.shape,(size,size))
            self.assertEqual(k.sum(),2*size-1)
            self.assertEqual(k[size//2,size//2],1)

    def test_reference_semantics(self):
        row={'input':'prompt','outputs':['one','two'],'index':3}
        self.assertEqual(convert(row,'qa_2')['label'],'one')
        for t in ('fwe','niah_single_1','niah_multikey_1'):
            self.assertEqual(convert(row,t)['label'],'one, two')
        with self.assertRaises(ValueError):convert({'input':'p','outputs':[]},'qa_2')

    def test_task_balancing_and_paired_guard(self):
        rows=[]
        # Unequal task sample counts: macro must not equal pooled-input mean.
        for task,losses in [('qa_2',[1.0]),('fwe',[3.0,5.0])]:
            for i,loss in enumerate(losses):
                rows.extend([dict(task=task,sample_index=i,policy='fixed_1x1',label_loss=loss),
                             dict(task=task,sample_index=i,policy='fixed_3x3',label_loss=loss-0.2)])
        summary=summarize(rows,['qa_2','fwe'],['fixed_1x1','fixed_3x3'])
        self.assertEqual(summary['macro_mean_nll']['fixed_1x1'],2.5)
        self.assertAlmostEqual(summary['macro_nll_decrease_from_1x1']['fixed_3x3'],0.2)
        with self.assertRaises(ValueError):summarize(rows[:-1],['qa_2','fwe'],['fixed_1x1','fixed_3x3'])

    def test_interrupted_sweep_reuses_committed_policies(self):
        import run_sweep
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data.jsonl';out=root/'out'
            data.write_text(json.dumps(dict(task='qa_2',sample_index=0,prompt='fixture',label='fixture'))+'\n')
            command=['run_sweep.py','--fixed-only','--data',str(data),'--output',str(out),
                     '--model',str(root/'model'),'--tasks','qa_2','--samples-per-task','1','--resume']
            calls=[]
            def simulated(a,env,item,policy):
                calls.append(policy)
                if len(calls)==3:raise KeyboardInterrupt('test interruption')
                return dict(policy=policy,label_loss=1.,token_losses=[1.],repeat_drift=0.,prompt_tokens=1,label_tokens=1)
            with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(sys,'argv',command),\
                 mock.patch.object(run_sweep,'setup',return_value='CPU fixture'),\
                 mock.patch.object(run_sweep,'evaluate_policy',side_effect=simulated):
                with self.assertRaises(KeyboardInterrupt):run_sweep.main()
            self.assertTrue((out/'qa_2/s0/fixed_1x1.json').is_file())
            self.assertTrue((out/'qa_2/s0/fixed_3x3.json').is_file())
            self.assertFalse((out/'qa_2/s0/fixed_5x5.json').exists())
            resumed=[]
            def complete(a,env,item,policy):
                resumed.append(policy)
                return dict(policy=policy,label_loss=1.,token_losses=[1.],repeat_drift=0.,prompt_tokens=1,label_tokens=1)
            with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(sys,'argv',command),\
                 mock.patch.object(run_sweep,'setup',return_value='CPU fixture'),\
                 mock.patch.object(run_sweep,'evaluate_policy',side_effect=complete):run_sweep.main()
            self.assertEqual(resumed,['fixed_5x5','fixed_7x7','fixed_9x9','fixed_11x11'])
            self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
