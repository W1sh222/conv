"""Run with python -m unittest discover -s ft_scripts/sparse_ruler -p test_replay_continuation.py."""
import csv
import ast
import json
import random
import string
import tempfile
import unittest
from pathlib import Path
from select_replay_checkpoint import choose, longbench_mean, ruler_mean, LB_TASKS, RU_TASKS
try:
    import torch
except ImportError:
    torch=None


class CheckpointGateTests(unittest.TestCase):
    def test_rejects_ruler_gain_with_longbench_regression(self):
        candidates=[dict(weight='bad',longbench=37.6,ruler10=85),
                    dict(weight='good',longbench=37.8,ruler10=78)]
        self.assertEqual(choose(37.7,77.45,candidates)['weight'],'good')
        self.assertIsNone(choose(37.7,77.45,candidates[:1]))

    def test_exclusions_do_not_drop_qa2_or_null_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'summary.csv'
            scores=[100]*13
            for t in ['niah_multikey_2','niah_multikey_3','qa_1','qa_2']:
                scores[RU_TASKS.index(t)]=0
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Tasks']+RU_TASKS);w.writerow(['Score']+scores)
            self.assertEqual(ruler_mean(path),90)

    def test_longbench_requires_complete_task_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            path.write_text(json.dumps({t:40 for t in LB_TASKS[:-1]}))
            with self.assertRaises(ValueError): longbench_mean(path)


class SyntheticAggregationTests(unittest.TestCase):
    def test_ten_word_cwe_and_three_word_fwe(self):
        # Execute the actual pure generator functions without importing the
        # model/tokenizer dependencies required only for dataset construction.
        source=ast.parse(Path(__file__).with_name('build_ruler_mix_sft.py').read_text(encoding='utf-8'))
        names={'rand_lower_word','rand_code_word','make_marker','wrap_marked',
               'choose_from_mix','choose_position','choose_position_mixed','build_word_aggregation'}
        nodes=ast.parse('from __future__ import annotations').body
        nodes += [n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
        namespace={'random':random,'string':string}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<generator-functions>','exec'),namespace)
        build=namespace['build_word_aggregation']
        for task,count in [('cwe',10),('fwe',3)]:
            spec=build(random.Random(100),1,task,{'uniform':1.0},cwe_num_words=10)
            words=spec['target_values']
            self.assertEqual(len(words),count)
            self.assertEqual(len(set(words)),count)
            self.assertEqual(set(spec['answer'].split(', ')),set(words))
            self.assertTrue(spec['aggregation_spans'])
        with self.assertRaises(ValueError):
            build(random.Random(100),1,'cwe',{'uniform':1.0},cwe_num_words=0)


@unittest.skipIf(torch is None,'PyTorch unavailable; run these tests in fyc_qwen on the training server')
class ContinuationLossTests(unittest.TestCase):
    def test_future_scores_cannot_change_distribution(self):
        from continuation_losses import row_distribution
        x=torch.randn(1,2,4,4)
        y=x.clone();y[:,:,0,1:]=1e6;y[:,:,1,2:]=1e6
        for positive in [False,True]:
            self.assertTrue(torch.allclose(row_distribution(x,[0,1],positive_mass=positive),
                                           row_distribution(y,[0,1],positive_mass=positive)))

    def test_zero_mass_is_finite(self):
        from continuation_losses import row_distribution
        p=row_distribution(-torch.ones(1,2,4,4),[0,3],positive_mass=True)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(1,2,2)))

    def test_inference_teacher_allows_backward(self):
        from continuation_losses import distribution_kl, row_distribution, anchor_replay_loss
        x=torch.tensor([[[[1.,2.],[3.,1.]]]],requires_grad=True)
        with torch.inference_mode():
            teacher=torch.tensor([[[[.5,.5]]]])
        loss=distribution_kl(row_distribution(x,[1],positive_mass=True),teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertAlmostEqual(anchor_replay_loss(x,x.detach(),[1],.015).item(),0,places=6)


if __name__=='__main__': unittest.main()
