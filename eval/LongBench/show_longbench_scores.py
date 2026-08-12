#!/usr/bin/env python3
import os
import json
import argparse
from statistics import mean


TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "vcsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing jsonl prediction files and result.json",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="xattn",
        help="Method name to filter, e.g. xattn, full, flex, minference",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASKS),
        help="Comma-separated task names",
    )
    return parser.parse_args()


def normalize_result_dir(path: str) -> str:
    return path if path.endswith("/") else path + "/"


def get_task_name(filename: str) -> str:
    """
    eval.py 里的逻辑是：
        dataset = filename.split("-")[0]

    例如：
        2wikimqa-xattn-stride=8.jsonl -> 2wikimqa
    """
    return filename.split("-")[0].replace(".jsonl", "")


def main():
    args = parse_args()
    result_dir = normalize_result_dir(args.result_dir)
    method = args.method
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    result_path = os.path.join(result_dir, "result.json")

    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Cannot find result.json: {result_path}\n"
            f"Please run eval.py first, for example:\n"
            f"python -u eval.py --results_path {result_dir}"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    print()
    print("| task | file | score |")
    print("|---|---|---|")

    numeric_scores = []
    missing_tasks = []

    for task in tasks:
        matched = []

        for filename, score in scores.items():
            if not filename.endswith(".jsonl"):
                continue

            file_task = get_task_name(filename)

            if file_task != task:
                continue

            # 只展示指定 method，比如 xattn
            if method and f"-{method}" not in filename:
                continue

            matched.append((filename, score))

        if not matched:
            missing_tasks.append(task)
            print(f"| {task} | MISSING | - |")
            continue

        matched.sort(key=lambda x: x[0])

        for filename, score in matched:
            print(f"| {task} | {filename} | {score} |")

            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))

    print()

    if numeric_scores:
        print(f"Average over found numeric scores: {mean(numeric_scores):.2f}")
    else:
        print("Average over found numeric scores: N/A")

    if missing_tasks:
        print()
        print("Missing tasks:")
        for task in missing_tasks:
            print(f"- {task}")


if __name__ == "__main__":
    main()