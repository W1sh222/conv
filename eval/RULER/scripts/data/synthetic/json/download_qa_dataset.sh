#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

download() {
    local output="$1"
    shift

    if command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=60 "$@" -O "${output}.tmp"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --connect-timeout 60 "$@" -o "${output}.tmp"
    else
        echo "Neither wget nor curl is installed; cannot download ${output}." >&2
        exit 1
    fi

    python - "${output}.tmp" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
    mv -f "${output}.tmp" "${output}"
}

if [[ ! -s squad.json ]]; then
    download squad.json \
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
else
    echo "squad.json already exists; skipping."
fi

if [[ ! -s hotpotqa.json ]]; then
    if ! download hotpotqa.json \
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"; then
        rm -f hotpotqa.json.tmp
        download hotpotqa.json \
            "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"
    fi
else
    echo "hotpotqa.json already exists; skipping."
fi

echo "RULER QA datasets are ready in ${SCRIPT_DIR}."
