"""Render synthetic unit-test fixtures only inside an automatically deleted temp dir."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from plot_analysis import plot


class PlotTests(unittest.TestCase):
    def test_all_plot_paths_and_formats(self):
        with tempfile.TemporaryDirectory(prefix='conv_analysis_test_') as tmp:
            root=Path(tmp);case=root/'test_fixture';case.mkdir()
            trials=[dict(key_block=i,initial_score=.1*i,learned_score=.2*i,
                         utility=(i-1)*.01,trial_type='single_swap') for i in range(3)]
            (case/'trials.jsonl').write_text('\n'.join(json.dumps(r) for r in trials))
            np.savez(case/'block_maps.npz',initial_mask=[True,False,True,False],
                     learned_mask=[True,True,False,False],query_block=2)
            metrics={'initial_spearman':-.2,'learned_spearman':.4,
                     'top_candidates':[dict(actual_pool_top_n=1,initial_mean_utility=-.01,learned_mean_utility=.01)],
                     'whole_row_utility':.005,
                     'losses':dict(initial=1.1,positive_1x1=1.1,fixed_vertical_diagonal=1.12,learned_5000=1.08)}
            (case/'experiment.json').write_text(json.dumps(dict(status='complete',metrics=metrics)))
            for kind in (3,4,5): plot(root,kind)
            metrics['losses']['learned_llama_replay']=metrics['losses'].pop('learned_5000')
            (case/'experiment.json').write_text(json.dumps(dict(status='complete',metrics=metrics,learned_policy='learned_llama_replay')))
            plot(root,5)
            for stem in ('rank_correlation','top_candidates_utility','whole_row_utility','neighborhood_ablation_nll'):
                for ext in ('png','pdf','svg'):
                    self.assertGreater((root/'plots'/f'{stem}.{ext}').stat().st_size,100)


if __name__=='__main__': unittest.main()
