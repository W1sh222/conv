import sys
from pathlib import Path
import unittest
import contextlib
import io
import json
import tempfile
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments3'))
from analysis_core import fixed_vertical_diagonal_kernel
from sweep_core import convert, summarize, SIZES


class SweepTests(unittest.TestCase):
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
