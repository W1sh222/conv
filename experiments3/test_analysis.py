"""CPU tests of analysis/interventions; no experimental results are manufactured."""
import unittest
import numpy as np
from analysis_core import ranks,spearman,row_swap,compare_sets,ranking_summary,input_cluster_summary,fixed_vertical_diagonal_kernel,validate_checkpoint_layout


class AnalysisTests(unittest.TestCase):
    def test_qwen_llama_checkpoint_layout(self):
        validate_checkpoint_layout('qwen3',36,32,(36,32,7,7))
        validate_checkpoint_layout('llama',32,32,(32,32,7,7))
        with self.assertRaises(ValueError): validate_checkpoint_layout('llama',32,32,(36,32,7,7))
        with self.assertRaises(ValueError): validate_checkpoint_layout('other',32,32,(32,32,7,7))
    def test_fixed_kernel_union_has_thirteen_positions(self):
        k=fixed_vertical_diagonal_kernel()
        self.assertEqual(k.sum(),13)
        self.assertEqual(k[3,3],1)
        np.testing.assert_equal(k[:,3],np.ones(7))
        np.testing.assert_equal(k.diagonal(),np.ones(7))
        with self.assertRaises(ValueError): fixed_vertical_diagonal_kernel(6)
    def test_average_ranks_and_constant(self):
        np.testing.assert_equal(ranks([4,1,1,2]),[4,1.5,1.5,3])
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1)
        self.assertIsNone(spearman([1,1,1],[1,2,3]))
    def test_budget_and_no_mutation(self):
        base=np.array([True,False,True,False,False])
        changed=row_swap(base,3,2,1)
        np.testing.assert_equal(base,[True,False,True,False,False])
        self.assertEqual(base.sum(),changed.sum())
        with self.assertRaises(ValueError): row_swap(base,3,2,4)
        with self.assertRaises(ValueError): row_swap(base,3,2,0)
    def test_pairing_independent_of_labels(self):
        a=[True,True,False,False];b=[True,False,True,False]
        d=compare_sets(a,b,3,[4,3,2,1],[4,1,3,2])
        self.assertEqual(d,{'common':[0],'removed':[1],'added':[2],'pairs':[(1,2)]})
        self.assertEqual(compare_sets(a,a,3,[4,3,2,1],[4,3,2,1])['pairs'],[])
        with self.assertRaises(ValueError): compare_sets(a,[True,False,False,False],3,[4]*4,[4]*4)
    def test_ranking_known_utility(self):
        trials=[dict(key_block=i,initial_score=-i,learned_score=i,utility=i-.5) for i in range(3)]
        d=ranking_summary(trials,[1,10],1e-5)
        self.assertAlmostEqual(d['initial_spearman'],-1)
        self.assertAlmostEqual(d['learned_spearman'],1)
        self.assertEqual(d['top_candidates'][0]['initial_improvement_fraction'],0)
        self.assertEqual(d['top_candidates'][0]['learned_improvement_fraction'],1)
        self.assertEqual(d['top_candidates'][1]['requested_pool_top_n'],10)
        self.assertEqual(d['top_candidates'][1]['actual_pool_top_n'],3)
    def test_inputs_not_heads_are_replicates(self):
        records=[dict(input_id='a',value=0),dict(input_id='a',value=2),dict(input_id='b',value=5)]
        d=input_cluster_summary(records,'value',seed=42)
        self.assertEqual(d['n_inputs'],2);self.assertEqual(d['mean'],3)
        single=input_cluster_summary(records[:2],'value')
        self.assertIsNone(single['ci95']);self.assertEqual(single['n_inputs'],1)


if __name__=='__main__': unittest.main()
