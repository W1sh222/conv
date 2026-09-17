#!/usr/bin/env python3
"""Generate repository-standard RULER VT data, then run the label-loss swap test."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RULER = ROOT / "eval/RULER/scripts"
RUNNER = ROOT / "experiments/block_label_swap/run_experiment.py"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Local model path; default: Qwen3-8B from existing eval config")
    p.add_argument("--model-template", choices=["qwen3", "meta-llama3"], default=None,
                   help="Normally inferred from local model config.json")
    p.add_argument("--seq-length", type=int, default=32768)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--head", type=int, default=8, help="Zero-based query head, not KV head")
    p.add_argument("--query-block", default="last")
    p.add_argument("--ratio", type=float, default=0.65)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--selector", choices=["initial", "conv"], default="initial")
    p.add_argument("--conv-weights")
    p.add_argument("--background", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--rope-factor", type=float)
    p.add_argument("--rope-original-length", type=int, default=32768)
    p.add_argument("--max-position-embeddings", type=int)
    p.add_argument("--loss-tolerance", type=float, default=1e-5)
    p.add_argument("--output", type=Path, help="New/empty output directory")
    p.add_argument("--prepare-only", action="store_true", help="Generate and validate data without loading model weights")
    p.add_argument("--dry-run", action="store_true", help="Print plan and environment issues; do not write or execute")
    p.add_argument("--no-plots", action="store_true")
    return p


def validate_args(a):
    if a.seq_length < 4096:
        raise ValueError("Use seq-length >= 4096 for this long-context observation")
    if not 0 <= a.sample_index < a.num_samples:
        raise ValueError("sample-index must be in [0, num-samples)")
    if not 0 < a.ratio < 1 or a.stride <= 0 or 128 % a.stride:
        raise ValueError("ratio must be in (0,1); stride must divide block size 128")
    if a.layer < 0 or a.head < 0 or a.loss_tolerance < 0:
        raise ValueError("layer, head and loss-tolerance must be nonnegative")
    if a.query_block != "last" and int(a.query_block) < 0:
        raise ValueError("query-block must be last or a nonnegative integer")
    if a.rope_factor is not None and a.rope_factor <= 0:
        raise ValueError("rope-factor must be positive")
    if a.rope_original_length <= 0 or (a.max_position_embeddings is not None and a.max_position_embeddings <= 0):
        raise ValueError("Context lengths must be positive")
    if a.selector == "conv" and not a.conv_weights:
        raise ValueError("--selector conv requires --conv-weights")


def default_model():
    config = ROOT / "eval/LongBench/config/model2path.json"
    return json.loads(config.read_text(encoding="utf-8"))["Qwen3-8B"]


def environment_issues(a):
    issues = []
    model_config = Path(a.model) / "config.json"
    if not model_config.is_file():
        issues.append(f"Local model config not found: {model_config}; supply --model on the GPU host")
    required = ["yaml", "numpy", "tqdm", "tenacity", "transformers"]
    if not a.prepare_only:
        required += ["torch", "accelerate", "triton", "block_sparse_attn"]
        if not a.no_plots:
            required.append("matplotlib")
    for name in required:
        if importlib.util.find_spec(name) is None:
            issues.append(f"Missing Python module: {name}" + (" (PyYAML)" if name == "yaml" else ""))
    if not a.prepare_only and importlib.util.find_spec("torch") is not None:
        try:
            import torch
            if not torch.cuda.is_available():
                issues.append("CUDA is not available to this Python interpreter")
        except Exception as exc:
            issues.append(f"Cannot import torch: {exc}")
    if not a.prepare_only and importlib.util.find_spec("transformers") is not None:
        from importlib.metadata import version
        if version("transformers") != "4.51.0":
            issues.append("Sparse adapter requires transformers==4.51.0")
    if a.selector == "conv" and not Path(a.conv_weights).is_file():
        issues.append(f"Conv weights not found: {a.conv_weights}")
    return issues


def load_task():
    import yaml
    custom = yaml.safe_load((RULER / "synthetic.yaml").read_text(encoding="utf-8"))["vt"]
    if custom["task"] != "variable_tracking":
        raise ValueError("Expected repository vt task to use variable_tracking")
    base = runpy.run_path(str(RULER / "data/synthetic/constants.py"))["TASKS"][custom["task"]]
    return {**base, **custom}


def resolve_template(a):
    cfg = json.loads((Path(a.model) / "config.json").read_text(encoding="utf-8"))
    expected = {"qwen3": "qwen3", "llama": "meta-llama3"}.get(cfg.get("model_type"))
    if expected is None:
        raise ValueError("Only Qwen3 and Llama models are supported")
    if a.model_template and a.model_template != expected:
        raise ValueError("Requested template does not match model_type")
    if a.layer >= cfg["num_hidden_layers"] or a.head >= cfg["num_attention_heads"]:
        raise ValueError("Requested layer/head is outside model layout")
    context_limit = cfg["max_position_embeddings"]
    if a.rope_factor is not None:
        import math
        context_limit = math.ceil(a.rope_original_length * a.rope_factor)
    if a.max_position_embeddings is not None:
        context_limit = a.max_position_embeddings
    if a.seq_length > context_limit:
        raise ValueError(f"seq-length exceeds configured context {context_limit}; pass the evaluation RoPE settings explicitly")
    return expected


def generation_command(a, task, template_name, out):
    templates = runpy.run_path(str(RULER / "data/template.py"))["Templates"]
    # Exactly the composition in RULER prepare.py; keep answer prefix and few-shot behavior.
    template = templates[template_name].format(task_template=task["template"]) + task["answer_prefix"]
    cmd = [sys.executable, "-u", str(RULER / "data/synthetic/variable_tracking.py"),
           "--save_dir", str(out / "raw"), "--save_name", "vt", "--subset", "validation",
           "--tokenizer_path", a.model, "--tokenizer_type", "hf",
           "--max_seq_length", str(a.seq_length), "--tokens_to_generate", str(task["tokens_to_generate"]),
           "--num_samples", str(a.num_samples), "--random_seed", str(a.seed), "--template", template]
    for key in ("num_chains", "num_hops"):
        cmd += ["--" + key, str(task["args"][key])]
    return cmd


def convert_record(row, num_hops):
    prompt, outputs = row.get("input"), row.get("outputs")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("RULER input must be a nonempty string")
    if (not isinstance(outputs, list) or len(outputs) != num_hops + 1
            or any(not isinstance(x, str) or not x.strip() for x in outputs)):
        raise ValueError("VT requires all num_hops+1 variable names in outputs")
    if len(set(outputs)) != len(outputs) or any(x not in prompt for x in outputs):
        raise ValueError("VT variables must be distinct and present in the prompt")
    return {"prompt": prompt, "label": ", ".join(outputs), "ruler_outputs": outputs,
            "ruler_index": row.get("index"), "ruler_reported_length": row.get("length"),
            "task": "vt", "label_format": "All variables in generator order, comma-space separated; no EOS"}


def prepare_data(raw, destination, a, task, tokenizer):
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != a.num_samples:
        raise ValueError(f"Expected {a.num_samples} samples, found {len(rows)}")
    converted = []
    for row in rows:
        item = convert_record(row, task["args"]["num_hops"])
        # Same tokenization boundary and special-token policy as run_experiment.py.
        n_prompt = len(tokenizer.encode(item["prompt"], add_special_tokens=True))
        n_label = len(tokenizer.encode(item["label"], add_special_tokens=False))
        if not n_prompt or not n_label or n_prompt + n_label > a.seq_length:
            raise ValueError(f"Actual prompt+label length {n_prompt}+{n_label} exceeds requested budget or is empty; no truncation")
        item.update(prompt_tokens=n_prompt, label_tokens=n_label,
                    prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest())
        converted.append(item)
    destination.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in converted), encoding="utf-8")
    return [{k: v for k, v in x.items() if k != "prompt"} for x in converted]


def observation_command(a, out):
    cmd = [sys.executable, "-u", str(RUNNER), "--model", a.model,
           "--data", str(out / "observation.jsonl"), "--output", str(out / "swap")]
    for name in ("sample_index", "layer", "head", "query_block", "ratio", "stride", "selector",
                 "background", "device_map", "dtype", "seed", "loss_tolerance", "rope_factor",
                 "rope_original_length", "max_position_embeddings", "conv_weights"):
        value = getattr(a, name)
        if value is not None:
            cmd += ["--" + name.replace("_", "-"), str(value)]
    if a.no_plots:
        cmd.append("--no-plots")
    # Prompt already includes the repository model template: never --chat-template.
    return cmd


def run_logged(cmd, log):
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()


def main():
    a = parser().parse_args()
    validate_args(a)
    a.model = a.model or default_model()
    out = (a.output or ROOT / "output/ruler_observation" /
           f"vt_{a.seq_length}_seed{a.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    issues = environment_issues(a)
    manifest = {"status": "planned", "task": "vt", "arguments": vars(a), "output": str(out),
                "python": sys.executable, "environment_issues": issues,
                "observation_command": observation_command(a, out),
                "metric": "Mean teacher-forced label NLL, not RULER free-generation accuracy"}
    if a.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        return 0
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    def save():
        (out / "pipeline.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    try:
        if issues:
            manifest["status"] = "environment_blocked"
            print("Cannot execute this experiment:\n- " + "\n- ".join(issues), file=sys.stderr)
            return 2
        task = load_task()
        template = resolve_template(a)
        cmd = generation_command(a, task, template, out)
        manifest.update(status="generating", task_config=task, model_template=template,
                        generation_command=cmd)
        manifest["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [RULER / "synthetic.yaml", RULER / "data/template.py",
                                              RULER / "data/synthetic/constants.py",
                                              RULER / "data/synthetic/variable_tracking.py", RUNNER]}
        save()
        run_logged(cmd, out / "generation.log")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
        manifest["samples"] = prepare_data(out / "raw/vt/validation.jsonl", out / "observation.jsonl", a, task, tokenizer)
        del tokenizer
        manifest["status"] = "data_ready"
        save()
        if not a.prepare_only:
            manifest["status"] = "running_observation"
            save()
            run_logged(manifest["observation_command"], out / "observation.log")
            result = json.loads((out / "swap/experiment.json").read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                raise RuntimeError("Observation runner did not report complete")
            manifest.update(status="complete", observation_summary=result["summary"])
        print(f"{manifest['status']}: {out}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(manifest["error"], file=sys.stderr)
        return 1
    finally:
        save()


if __name__ == "__main__":
    sys.exit(main())
