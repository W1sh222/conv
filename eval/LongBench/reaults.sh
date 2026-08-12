#!/usr/bin/env bash
set -euo pipefail

# LongBench 根目录
LONGBENCH_DIR="/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/eval/LongBench"

# 你的预测结果目录
DEFAULT_RESULT_DIR="${LONGBENCH_DIR}/pred/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct/minference"

# 可以通过第一个参数覆盖默认结果目录
RESULT_DIR="${1:-$DEFAULT_RESULT_DIR}"

# eval.py 里用的是 f"{path}{filename}"，所以必须保证结尾有 /
case "$RESULT_DIR" in
    */) ;;
    *) RESULT_DIR="${RESULT_DIR}/" ;;
esac

cd "$LONGBENCH_DIR"

echo "Result dir: $RESULT_DIR"

# 默认如果没有 result.json，就先跑 eval.py
# REFRESH=1 时强制重新评测
# NO_EVAL=1 时跳过 eval.py，只读取已有 result.json
if [[ "${NO_EVAL:-0}" != "1" ]]; then
    if [[ "${REFRESH:-0}" == "1" || ! -f "${RESULT_DIR}/result.json" ]]; then
        echo "Running eval.py ..."
        python -u eval.py --results_path "$RESULT_DIR"
    else
        echo "Found existing result.json, skip eval.py."
    fi
fi

python show_longbench_scores.py \
    --result_dir "$RESULT_DIR" \
    --method conv