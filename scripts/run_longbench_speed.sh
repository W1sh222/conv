#!/usr/bin/env bash
set -euo pipefail
set -x
export CUDA_VISIBLE_DEVICES=0

models="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct"

# XAttention
methods="conv"

# Baselines
# methods="full flex minference conv"

# tasks="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum vcsum multi_news trec triviaqa samsum lsht lcc repobench-p"
# tasks="vcsum multi_news trec triviaqa samsum lsht lcc repobench-p narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique"
tasks="narrativeqa"

for model in $models; do
    for task in $tasks; do
        for method in $methods; do
            bash scripts/longbench.sh "$model" "$task" "$method"
        done
    done
done

cd eval/LongBench

for model in $models; do
    python -u eval.py --model "$model"
done