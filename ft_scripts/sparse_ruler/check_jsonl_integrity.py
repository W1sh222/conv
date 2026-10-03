#!/usr/bin/env python3
"""Validate a generated RULER-Mix JSONL before Hugging Face ingestion.

The dataset builder publishes files atomically, but this small check catches a
stale/truncated file (or malformed final record) before ``datasets`` creates an
Arrow cache from it.  It intentionally reads one JSON record at a time so it
does not duplicate the whole long-context dataset in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        raise FileNotFoundError(path)

    count = 0
    last_line = b""
    with path.open("rb") as handle:
        for count, raw in enumerate(handle, start=1):
            last_line = raw
            if not raw.endswith(b"\n"):
                raise ValueError(
                    f"{path}: record {count} is not newline terminated; "
                    "the file may have been truncated"
                )
            try:
                record = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - include record context
                raise ValueError(f"{path}: invalid JSON at record {count}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: record {count} is not a JSON object")

    if count != args.expected:
        raise ValueError(
            f"{path}: expected {args.expected} records, found {count}"
        )
    if args.expected > 0 and not last_line:
        raise ValueError(f"{path}: empty final record")
    print(f"[data] JSONL integrity OK: {path} ({count} records)")


if __name__ == "__main__":
    main()
