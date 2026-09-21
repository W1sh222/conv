#!/usr/bin/env python3
"""
Build RULER-Mix synthetic SFT data for guarded conv-kernel training.

This script does NOT use RULER test data. It generates RULER-style tasks from
NoLiMa haystack text plus synthetic records.

Output JSONL schema:
{
  "messages": [{"role":"user",...}, {"role":"assistant",...}],
  "meta": {
    "task_type": "niah_multikey_2" | ...,
    "loss_type": "ranking" | "aggregation" | "dense",
    "target_record_spans": [
      {"start_marker":"...", "end_marker":"...", "kind":"target"}
    ],
    "aggregation_spans": [ ... ],
    "target_keys": [...],
    "target_values": [...]
  }
}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from transformers import AutoTokenizer


# ----------------------------- basic utils -----------------------------


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_space(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def collect_haystack_texts(nolima_root: str, haystack_subdirs: List[str]) -> List[str]:
    root = Path(nolima_root)
    texts: List[str] = []
    for sub in haystack_subdirs:
        p = root / "haystack" / sub
        if not p.exists():
            print(f"[WARN] haystack subdir not found: {p}")
            continue
        for txt_path in sorted(p.rglob("*.txt")):
            text = normalize_space(read_text(txt_path))
            if len(text) >= 1000:
                texts.append(text)
                print(f"[haystack] loaded {txt_path} chars={len(text)}")
    if not texts:
        raise RuntimeError(f"No haystack txt files found under {root / 'haystack'}")
    return texts


def tokenize_plain(tokenizer, text: str) -> List[int]:
    return tokenizer(text, truncation=False, padding=False, add_special_tokens=False)["input_ids"]


def decode_plain(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def build_haystack_token_pool(tokenizer, haystack_texts: List[str]) -> List[List[int]]:
    pool: List[List[int]] = []
    for i, text in enumerate(haystack_texts):
        ids = tokenize_plain(tokenizer, text)
        if len(ids) >= 512:
            pool.append(ids)
            print(f"[tokenize haystack] idx={i} tokens={len(ids)}")
    if not pool:
        raise RuntimeError("No haystack text has >=512 tokens.")
    return pool


def sample_base_text(rng: random.Random, pool: List[List[int]], tokenizer, num_tokens: int) -> str:
    sep_ids = tokenize_plain(tokenizer, "\n\n")
    out: List[int] = []
    while len(out) < num_tokens:
        src = rng.choice(pool)
        take = min(num_tokens - len(out), len(src))
        start_max = max(0, len(src) - take - 1)
        start = rng.randint(0, start_max) if start_max > 0 else 0
        out.extend(src[start:start + take])
        out.extend(sep_ids)
    return decode_plain(tokenizer, out[:num_tokens])


def parse_mix(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":")
        out[name.strip()] = float(value)
    if not out:
        raise ValueError("empty mix")
    return out


def choose_from_mix(rng: random.Random, mix: Dict[str, float]) -> str:
    names = list(mix.keys())
    weights = [mix[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def rand_lower_word(rng: random.Random, min_len: int = 5, max_len: int = 9) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def rand_word_phrase(rng: random.Random, n: int = 3) -> str:
    return " ".join(rand_lower_word(rng) for _ in range(n))


def rand_number(rng: random.Random, digits: int = 7) -> str:
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    return str(rng.randint(lo, hi))


def rand_code_word(rng: random.Random) -> str:
    return rand_lower_word(rng, 4, 8)


def rand_uuid_str(rng: random.Random) -> str:
    # deterministic from rng by drawing 128 bits
    return str(uuid.UUID(int=rng.getrandbits(128)))


def make_marker(sample_id: int, idx: int, kind: str, side: str) -> str:
    return f"[{kind.upper()}_{side.upper()}_{sample_id:06d}_{idx:03d}]"


def wrap_marked(sample_id: int, idx: int, kind: str, text: str) -> Tuple[str, Dict[str, Any]]:
    start = make_marker(sample_id, idx, kind, "START")
    end = make_marker(sample_id, idx, kind, "END")
    wrapped = f"{start} {text} {end}"
    meta = {"start_marker": start, "end_marker": end, "kind": kind, "text": text}
    return wrapped, meta


def choose_position(rng: random.Random, mode: str, sample_id: int) -> float:
    if mode == "uniform":
        return rng.uniform(0.03, 0.97)
    if mode == "edge":
        return rng.choice([rng.uniform(0.03, 0.18), rng.uniform(0.82, 0.97)])
    if mode == "bimodal":
        return rng.choice([rng.uniform(0.03, 0.30), rng.uniform(0.70, 0.97)])
    if mode == "front":
        return rng.uniform(0.03, 0.25)
    if mode == "middle":
        return rng.uniform(0.35, 0.65)
    if mode == "back":
        return rng.uniform(0.75, 0.97)
    if mode == "grid":
        grid = [0.03, 0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97]
        return grid[sample_id % len(grid)]
    raise ValueError(f"Unsupported position mode: {mode}")


def choose_position_mixed(rng: random.Random, position_mix: Dict[str, float], sample_id: int) -> float:
    mode = choose_from_mix(rng, position_mix)
    return choose_position(rng, mode, sample_id)


def insert_records_into_base(base_text: str, records: List[Dict[str, Any]]) -> str:
    records = sorted(records, key=lambda x: x["pos"])
    n = len(base_text)
    chunks: List[str] = []
    cursor = 0
    for r in records:
        pos = int(max(0.0, min(1.0, float(r["pos"]))) * n)
        pos = max(cursor, min(n, pos))
        chunks.append(base_text[cursor:pos])
        chunks.append("\n\n" + r["line"] + "\n\n")
        cursor = pos
    chunks.append(base_text[cursor:])
    return "".join(chunks)


def make_messages(context: str, question: str, answer: str) -> List[Dict[str, str]]:
    user = (
        "You are given a long context. Carefully retrieve or aggregate the requested information.\n"
        "Answer with the requested value only. Do not explain.\n\n"
        "Long context:\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]


def prompt_only_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Build exactly the prompt seen by sparse-prefill inference.

    The reference assistant answer must not be included in Q/K training.  The
    previous builder counted a right-truncated prompt+answer, which could accept
    over-length examples whose question had already been truncated away.
    """
    prompt_messages = list(messages)
    if prompt_messages and str(prompt_messages[-1].get("role", "")).lower() == "assistant":
        prompt_messages = prompt_messages[:-1]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def chat_token_len(
    tokenizer,
    messages: List[Dict[str, str]],
    max_seq_length: int,
    marker_spans: List[Dict[str, Any]] | None = None,
) -> int:
    del max_seq_length
    text = prompt_only_text(tokenizer, messages)
    # Match training/evaluation text exactly: marker strings are metadata
    # anchors only and must not become an easy retrieval feature.
    for span in marker_spans or []:
        for marker_key in ("start_marker", "end_marker"):
            marker = span.get(marker_key)
            if marker:
                marker = str(marker)
                text = text.replace(marker, " " * len(marker))
    return len(
        tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
    )


# ----------------------------- task generators -----------------------------


def niah_record(key: str, value: str, value_name: str = "value") -> str:
    return f"The {value_name} associated with the key phrase {key} is {value}."


def build_niah_single(
    rng: random.Random,
    sample_id: int,
    task_type: str,
    num_distractors: int,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    # three single variants differ mainly by position style / distractor intensity.
    key = rand_word_phrase(rng)
    value = rand_number(rng)
    target_line, span = wrap_marked(sample_id, 0, "target", niah_record(key, value))
    pos = choose_position_mixed(rng, pos_mix, sample_id)
    records = [{"line": target_line, "pos": pos, "is_target": True}]

    d = max(0, num_distractors // 4)
    if task_type == "niah_single_2":
        d = max(2, num_distractors // 2)
    elif task_type == "niah_single_3":
        d = max(4, num_distractors)
    for j in range(d):
        dk, dv = rand_word_phrase(rng), rand_number(rng)
        records.append({"line": niah_record(dk, dv), "pos": rng.uniform(0.03, 0.97), "is_target": False})

    question = f"What is the value associated with the key phrase {key}?"
    return {
        "records": records,
        "question": question,
        "answer": value,
        "target_keys": [key],
        "target_values": [value],
        "target_record_spans": [span],
        "aggregation_spans": [],
        "loss_type": "ranking",
    }


def build_niah_multikey(
    rng: random.Random,
    sample_id: int,
    task_type: str,
    num_distractors: int,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    if task_type == "niah_multikey_3":
        key, value = rand_uuid_str(rng), rand_uuid_str(rng)
    else:
        key, value = rand_word_phrase(rng), rand_number(rng)
    target_line, span = wrap_marked(sample_id, 0, "target", niah_record(key, value))
    records = [{"line": target_line, "pos": choose_position_mixed(rng, pos_mix, sample_id), "is_target": True}]

    for j in range(num_distractors):
        if task_type == "niah_multikey_3":
            dk, dv = rand_uuid_str(rng), rand_uuid_str(rng)
        else:
            dk, dv = rand_word_phrase(rng), rand_number(rng)
        records.append({"line": niah_record(dk, dv), "pos": rng.uniform(0.03, 0.97), "is_target": False})

    question = f"What is the value associated with the key phrase {key}?"
    return {
        "records": records,
        "question": question,
        "answer": value,
        "target_keys": [key],
        "target_values": [value],
        "target_record_spans": [span],
        "aggregation_spans": [],
        "loss_type": "ranking",
    }


def build_niah_multivalue(
    rng: random.Random,
    sample_id: int,
    num_distractors: int,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    key = rand_word_phrase(rng)
    values = [rand_number(rng) for _ in range(4)]
    records: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []
    for j, value in enumerate(values):
        line, span = wrap_marked(sample_id, j, "target", niah_record(key, value))
        records.append({"line": line, "pos": choose_position_mixed(rng, pos_mix, sample_id + j), "is_target": True})
        spans.append(span)
    for j in range(num_distractors):
        dk, dv = rand_word_phrase(rng), rand_number(rng)
        records.append({"line": niah_record(dk, dv), "pos": rng.uniform(0.03, 0.97), "is_target": False})
    question = f"What are all values associated with the key phrase {key}?"
    return {
        "records": records,
        "question": question,
        "answer": ", ".join(values),
        "target_keys": [key],
        "target_values": values,
        "target_record_spans": spans,
        "aggregation_spans": [],
        "loss_type": "ranking",
    }


def build_niah_multiquery(
    rng: random.Random,
    sample_id: int,
    num_distractors: int,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    keys = [rand_word_phrase(rng) for _ in range(4)]
    values = [rand_number(rng) for _ in range(4)]
    records: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []
    for j, (key, value) in enumerate(zip(keys, values)):
        line, span = wrap_marked(sample_id, j, "target", niah_record(key, value))
        records.append({"line": line, "pos": choose_position_mixed(rng, pos_mix, sample_id + j), "is_target": True})
        spans.append(span)
    for j in range(num_distractors):
        dk, dv = rand_word_phrase(rng), rand_number(rng)
        records.append({"line": niah_record(dk, dv), "pos": rng.uniform(0.03, 0.97), "is_target": False})
    question = "What are the values associated with the following key phrases: " + "; ".join(keys) + "?"
    return {
        "records": records,
        "question": question,
        "answer": ", ".join(values),
        "target_keys": keys,
        "target_values": values,
        "target_record_spans": spans,
        "aggregation_spans": [],
        "loss_type": "ranking",
    }


def build_vt(
    rng: random.Random,
    sample_id: int,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    vars_ = [f"VAR_{rand_lower_word(rng, 4, 6).upper()}_{j}" for j in range(3)]
    final_values: Dict[str, str] = {}
    records: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []
    idx = 0
    for step in range(rng.randint(12, 22)):
        var = rng.choice(vars_)
        value = f"VAL_{rand_number(rng, 5)}"
        final_values[var] = value
        line = f"At update step {step}, variable {var} is set to {value}."
        records.append({"line": line, "pos": rng.uniform(0.03, 0.97), "is_target": False})
    # Add explicitly marked final confirmations, which serve as evidence of final states.
    for var in vars_:
        value = final_values.get(var, f"VAL_{rand_number(rng, 5)}")
        final_values[var] = value
        line = f"The final value of variable {var} is {value}."
        wrapped, span = wrap_marked(sample_id, idx, "target", line)
        records.append({"line": wrapped, "pos": choose_position_mixed(rng, pos_mix, sample_id + idx), "is_target": True})
        spans.append(span)
        idx += 1
    question = "What are the final values of variables " + ", ".join(vars_) + "?"
    answer = ", ".join([f"{v}={final_values[v]}" for v in vars_])
    return {
        "records": records,
        "question": question,
        "answer": answer,
        "target_keys": vars_,
        "target_values": [final_values[v] for v in vars_],
        "target_record_spans": spans,
        "aggregation_spans": [],
        "loss_type": "ranking",
    }


def build_word_aggregation(
    rng: random.Random,
    sample_id: int,
    task_type: str,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    vocab = [rand_code_word(rng) for _ in range(80)]
    top_words = [rand_code_word(rng) for _ in range(3)]
    if task_type == "cwe":
        # common vs uncommon: common words repeat heavily.
        freqs = {top_words[0]: 36, top_words[1]: 32, top_words[2]: 28}
        question = "Which three common coded words appear repeatedly in the coded text?"
    else:
        # fwe: explicit top-frequency task.
        freqs = {top_words[0]: 40, top_words[1]: 30, top_words[2]: 24}
        question = "What are the three most frequent words in the coded text?"

    tokens: List[str] = []
    for w, c in freqs.items():
        tokens.extend([w] * c)
    for w in vocab:
        tokens.extend([w] * rng.randint(1, 4))
    rng.shuffle(tokens)

    records: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []
    seg_size = 32
    idx = 0
    for start in range(0, len(tokens), seg_size):
        seg = tokens[start:start + seg_size]
        line = " ".join(seg) + "."
        if any(w in seg for w in top_words):
            wrapped, span = wrap_marked(sample_id, idx, "aggregate", line)
            records.append({"line": wrapped, "pos": choose_position_mixed(rng, pos_mix, sample_id + idx), "is_target": True})
            spans.append(span)
            idx += 1
        else:
            records.append({"line": line, "pos": rng.uniform(0.03, 0.97), "is_target": False})
    return {
        "records": records,
        "question": question,
        "answer": ", ".join(top_words),
        "target_keys": top_words,
        "target_values": top_words,
        "target_record_spans": [],
        "aggregation_spans": spans,
        "loss_type": "aggregation",
    }


def build_qa(
    rng: random.Random,
    sample_id: int,
    task_type: str,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    entity = rand_word_phrase(rng, 2)
    answer = rand_number(rng, 6)
    if task_type == "qa_1":
        question = f"According to the context, what identifier is assigned to {entity}?"
        evidence = f"In the reference passage, the identifier assigned to {entity} is {answer}."
    else:
        question = f"According to the context, what final identifier is assigned to {entity} after the update?"
        old = rand_number(rng, 6)
        evidence = f"Earlier notes list {entity} with identifier {old}, but the later confirmed final identifier for {entity} is {answer}."
    line, span = wrap_marked(sample_id, 0, "target", evidence)
    records = [{"line": line, "pos": choose_position_mixed(rng, pos_mix, sample_id), "is_target": True}]
    # Add QA-like distractor facts.
    for j in range(12):
        de = rand_word_phrase(rng, 2)
        dv = rand_number(rng, 6)
        records.append({"line": f"The identifier assigned to {de} is {dv}.", "pos": rng.uniform(0.03, 0.97), "is_target": False})
    return {
        "records": records,
        "question": question,
        "answer": answer,
        "target_keys": [entity],
        "target_values": [answer],
        "target_record_spans": [span],
        "aggregation_spans": [],
        "loss_type": "ranking",
    }


def build_dense_general(rng: random.Random, sample_id: int) -> Dict[str, Any]:
    return {
        "records": [],
        "question": "Continue focusing on the long context. What is the main content type of the context?",
        "answer": "long context",
        "target_keys": [],
        "target_values": [],
        "target_record_spans": [],
        "aggregation_spans": [],
        "loss_type": "dense",
    }


def build_task(
    rng: random.Random,
    sample_id: int,
    task_type: str,
    num_distractors: int,
    pos_mix: Dict[str, float],
) -> Dict[str, Any]:
    if task_type in {"niah_single_1", "niah_single_2", "niah_single_3"}:
        return build_niah_single(rng, sample_id, task_type, num_distractors, pos_mix)
    if task_type in {"niah_multikey_1", "niah_multikey_2", "niah_multikey_3"}:
        return build_niah_multikey(rng, sample_id, task_type, num_distractors, pos_mix)
    if task_type == "niah_multivalue":
        return build_niah_multivalue(rng, sample_id, num_distractors, pos_mix)
    if task_type == "niah_multiquery":
        return build_niah_multiquery(rng, sample_id, num_distractors, pos_mix)
    if task_type == "vt":
        return build_vt(rng, sample_id, pos_mix)
    if task_type in {"cwe", "fwe"}:
        return build_word_aggregation(rng, sample_id, task_type, pos_mix)
    if task_type in {"qa_1", "qa_2"}:
        return build_qa(rng, sample_id, task_type, pos_mix)
    if task_type == "dense_general":
        return build_dense_general(rng, sample_id)
    raise ValueError(f"Unsupported task_type={task_type}")


# ----------------------------- main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--nolima_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--haystack_subdirs", nargs="+", default=["rand_shuffle_long", "rand_shuffle"])
    parser.add_argument("--num_samples", type=int, default=24000)
    parser.add_argument("--min_seq_length", type=int, default=4096)
    parser.add_argument("--max_seq_length", type=int, default=9448)
    parser.add_argument("--num_distractor_needles", type=int, default=32)
    parser.add_argument("--position_mix", default="uniform:0.55,edge:0.25,bimodal:0.20")
    parser.add_argument(
        "--task_mix",
        default=(
            "niah_single_1:0.06,niah_single_2:0.06,niah_single_3:0.06,"
            "niah_multikey_1:0.04,niah_multikey_2:0.07,niah_multikey_3:0.07,"
            "niah_multivalue:0.05,niah_multiquery:0.05,vt:0.10,"
            "cwe:0.09,fwe:0.09,qa_1:0.08,qa_2:0.10,dense_general:0.08"
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_attempt_multiplier", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    haystack_texts = collect_haystack_texts(args.nolima_root, args.haystack_subdirs)
    pool = build_haystack_token_pool(tokenizer, haystack_texts)
    task_mix = parse_mix(args.task_mix)
    pos_mix = parse_mix(args.position_mix)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_counts: Counter[str] = Counter()
    written = 0
    attempts = 0
    max_attempts = args.num_samples * args.max_attempt_multiplier

    with out_path.open("w", encoding="utf-8") as f:
        while written < args.num_samples and attempts < max_attempts:
            attempts += 1
            sample_id = attempts
            task_type = choose_from_mix(rng, task_mix)
            # Reserve prompt+records budget. The final untruncated prompt-only
            # length check below confirms both bounds.
            base_tokens = rng.randint(max(512, args.min_seq_length - 768), max(1024, args.max_seq_length - 512))
            base_text = sample_base_text(rng, pool, tokenizer, base_tokens)
            spec = build_task(
                rng=rng,
                sample_id=sample_id,
                task_type=task_type,
                num_distractors=args.num_distractor_needles,
                pos_mix=pos_mix,
            )
            context = insert_records_into_base(base_text, spec["records"])
            messages = make_messages(context, spec["question"], spec["answer"])
            n_tokens = chat_token_len(
                tokenizer,
                messages,
                args.max_seq_length,
                list(spec["target_record_spans"]) + list(spec["aggregation_spans"]),
            )
            if n_tokens < args.min_seq_length or n_tokens > args.max_seq_length:
                continue
            example = {
                "messages": messages,
                "meta": {
                    "source": "NoLiMa_RULER_Mix_synthetic_v2",
                    "sample_id": sample_id,
                    "task_type": task_type,
                    "loss_type": spec["loss_type"],
                    "seq_len_prompt": n_tokens,
                    "target_keys": spec["target_keys"],
                    "target_values": spec["target_values"],
                    "target_record_spans": spec["target_record_spans"],
                    "aggregation_spans": spec["aggregation_spans"],
                    "num_distractor_needles": args.num_distractor_needles,
                    "position_mix": args.position_mix,
                },
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
            task_counts[task_type] += 1
            if written % 1000 == 0:
                print(f"[write] {written}/{args.num_samples} task_counts={dict(task_counts)}")

    if written < args.num_samples:
        raise RuntimeError(f"Only wrote {written}/{args.num_samples} after {attempts} attempts")

    print("=" * 80)
    print(f"Done. output={out_path}")
    print(f"num_samples={written}")
    print(f"attempts={attempts}")
    print(f"task_counts={dict(task_counts)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
