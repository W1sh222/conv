"""CPU-only contract tests; these do not substitute for a CUDA model run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_ruler_observation as pipeline


class FakeTokenizer:
    def encode(self, text, add_special_tokens):
        return list(range(len(text.split()) + int(add_special_tokens)))


class RulerObservationTests(unittest.TestCase):
    def setUp(self):
        self.args = pipeline.parser().parse_args(["--model", "/models/Qwen3-8B"])
        self.row = {"index": 0, "input": "A = 42; B = A; C = B; Answer: ",
                    "outputs": ["A", "B", "C"], "length": 32}
        self.task = {"name": "vt", "args": {"num_hops": 2}}

    def test_all_required_answers_and_prompt_preserved(self):
        item = pipeline.convert_record(self.row, "vt", self.task)
        self.assertEqual(item["label"], "A, B, C")
        self.assertEqual(item["prompt"], self.row["input"])
        self.assertEqual(item["ruler_outputs"], self.row["outputs"])

    def test_reject_missing_or_duplicate_answers(self):
        for outputs in (["A"], ["A", "B", "B"], ["A", "B", "ABSENT"], "A, B, C"):
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                pipeline.convert_record({**self.row, "outputs": outputs}, "vt", self.task)

    def test_no_double_chat_template_and_all_options_forwarded(self):
        self.args.rope_factor = 4
        self.args.max_position_embeddings = 131072
        self.args.conv_weights = "/weights/path with spaces.pt"
        cmd = pipeline.observation_command(self.args, Path("/results/with spaces"))
        self.assertNotIn("--chat-template", cmd)
        self.assertEqual(cmd[cmd.index("--conv-weights") + 1], self.args.conv_weights)
        self.assertEqual(cmd[cmd.index("--rope-factor") + 1], "4")
        self.assertEqual(cmd[cmd.index("--sample-index") + 1], "0")
        self.assertEqual(cmd[cmd.index("--selector") + 1], "initial")

    def test_generation_reuses_templates_and_requested_seed(self):
        self.args.seed = 123
        task = {"name": "vt", "template": "Track {context}. Find {query}", "answer_prefix": " Answer: ",
                "tokens_to_generate": 30, "args": {"num_chains": 1, "num_hops": 4}}
        cmd = pipeline.generation_command(self.args, task, "qwen3", Path("out"))
        self.assertEqual(cmd[cmd.index("--random_seed") + 1], "123")
        self.assertEqual(cmd[cmd.index("--num_hops") + 1], "4")
        template = cmd[cmd.index("--template") + 1]
        self.assertIn("/no_think", template)
        self.assertIn("{context}", template)
        self.assertTrue(template.endswith(" Answer: "))

    def test_actual_lengths_checked_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, prepared = Path(directory) / "raw.jsonl", Path(directory) / "prepared.jsonl"
            raw.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            task = {"name": "vt", "args": {"num_hops": 2}}
            result = pipeline.prepare_data(raw, prepared, self.args, task, FakeTokenizer())
            self.assertEqual(result[0]["label"], "A, B, C")
            self.assertNotIn("prompt", result[0])
            self.args.seq_length = 2
            with self.assertRaisesRegex(ValueError, "no truncation"):
                pipeline.prepare_data(raw, Path(directory) / "too_long.jsonl", self.args, task, FakeTokenizer())
            self.assertFalse((Path(directory) / "too_long.jsonl").exists())

    def test_missing_generated_samples_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text(json.dumps(self.row), encoding="utf-8")
            self.args.num_samples = 2
            with self.assertRaisesRegex(ValueError, "Expected 2 samples"):
                pipeline.prepare_data(raw, Path(directory) / "out.jsonl", self.args,
                                      {"name": "vt", "args": {"num_hops": 2}}, FakeTokenizer())

    def test_child_failure_is_propagated_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.run_logged([sys.executable, "-c", "print('failed child'); raise SystemExit(7)"], log)
            self.assertIn("failed child", log.read_text(encoding="utf-8"))

    def test_invalid_sparse_configuration_rejected(self):
        for options in (["--ratio", "1"], ["--stride", "3"], ["--sample-index", "1"],
                        ["--selector", "conv"], ["--query-block", "-1"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pipeline.validate_args(pipeline.parser().parse_args(options))

    def test_fwe_generation_and_conversion(self):
        args = pipeline.parser().parse_args(["--model", "/models/Qwen3-8B", "--task", "fwe"])
        task = {"name": "fwe", "template": "Track {context}", "answer_prefix": " Answer: ",
                "tokens_to_generate": 50, "args": {"alpha": 2.0}}
        cmd = pipeline.generation_command(args, task, "qwen3", Path("out"))
        generator = next(item for item in cmd if item.endswith("freq_words_extraction.py"))
        self.assertTrue(generator.endswith("freq_words_extraction.py"))
        self.assertEqual(cmd[cmd.index("--alpha") + 1], "2.0")
        self.assertEqual(cmd[cmd.index("--length-increment") + 1], "4")
        row = {"input": "foo bar baz", "outputs": ["foo", "bar", "baz"]}
        item = pipeline.convert_record(row, "fwe", task)
        self.assertEqual(item["label"], "foo, bar, baz")


if __name__ == "__main__":
    unittest.main()
