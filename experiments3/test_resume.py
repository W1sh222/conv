"""CPU-only recovery and provenance tests; never runs model inference."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from resume_state import TrialCheckpoint, prepare_run


class ResumeTests(unittest.TestCase):
    def test_recovery_and_baseline_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            c=TrialCheckpoint(path)
            c.check_baseline({'label_loss':1.0},1e-5)
            record={'key_block':4,'removed':3,'initial_score':0.1,'label_loss':0.9}
            c.save(record)
            # An interrupted temporary write must not replace the last committed trial.
            path.with_name('checkpoint.json.tmp').write_text('{')
            resumed=TrialCheckpoint(path)
            self.assertEqual(resumed.get({'key_block':4,'removed':3,'initial_score':0.1}),record)
            self.assertIsNone(resumed.get({'key_block':5,'removed':3}))
            with self.assertRaises(ValueError): resumed.get({'key_block':4,'removed':3,'initial_score':0.2})
            with self.assertRaises(ValueError): resumed.check_baseline({'label_loss':1.1},1e-5)

    def test_policy_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=TrialCheckpoint(Path(tmp)/'checkpoint.json')
            c.save({'policy':'learned_9750','label_loss':0.5,'repeat_drift':0.0})
            self.assertEqual(TrialCheckpoint(c.path).get({'policy':'learned_9750'})['label_loss'],0.5)

    def test_provenance_and_legacy_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';weight=root/'weight'
            data.write_text('input');weight.write_text('weights')
            a=argparse.Namespace(data=str(data),conv_weights=str(weight),model=str(root/'model'),
                                 resume=False,output=str(root/'out'),no_plots=True,dry_run=False,ratio=0.65)
            prepare_run(a.output,3,a)
            with self.assertRaises(FileExistsError): prepare_run(a.output,3,a)
            a.resume=True;prepare_run(a.output,3,a)
            a.no_plots=False;prepare_run(a.output,3,a)
            a.ratio=0.7
            with self.assertRaises(ValueError): prepare_run(a.output,3,a)
            a.ratio=0.65;weight.write_text('other checkpoint')
            with self.assertRaises(ValueError): prepare_run(a.output,3,a)
            legacy=root/'legacy';legacy.mkdir();(legacy/'summary.json').write_text('{}')
            with self.assertRaises(ValueError): prepare_run(legacy,3,a)


if __name__=='__main__': unittest.main()
