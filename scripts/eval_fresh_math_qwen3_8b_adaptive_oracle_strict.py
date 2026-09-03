from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_answer_cluster_consensus import (
    extract_answer,
    normalize_answer,
)


CANDIDATE_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_candidates.jsonl"
)
CANDIDATE_MANIFEST_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_adaptive_k16.json"
)
LABEL_PATH = (
    ROOT
    / "data/external/fresh_math_2026/"
    "sealed_labels.jsonl"
)
OUTPUT_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_adaptive_oracle_strict.json"
)

FREEZE_TAG = (
    "fresh-math-2026-qwen3-adaptive-k16-candidates-v1"
)
K_VALUES = [1, 4, 8, 16]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(name):
    return subprocess.check_output(
        ["git", "rev-list", "-n", "1", name],
        cwd=ROOT,
        text=True,
    ).strip()


def pass_at_k(n, correct, k):
    if correct <= 0:
        return 0.0
    if n - correct < k:
        return 1.0

    failure = 1.0
    for offset in range(k):
        failure *= (
            (n - correct - offset)
            / (n - offset)
        )
    return 1.0 - failure


def majority_prediction(rows):
    parsed = [
        row["normalized_prediction"]
        for row in rows
        if row["normalized_prediction"] is not None
    ]
    if not parsed:
        return None, 0, 0

    counts = Counter(parsed)
    largest = max(counts.values())
    tied = {
        answer
        for answer, count in counts.items()
        if count == largest
    }

    # 固定规则：最大支持簇；并列时选择最早出现的簇。
    selected = next(
        row["normalized_prediction"]
        for row in rows
        if row["normalized_prediction"] in tied
    )
    return selected, largest, len(tied)


def summarize(question_records):
    if not question_records:
        raise RuntimeError("没有问题记录")

    candidate_count = sum(
        len(item["candidates"])
        for item in question_records
    )
    parsed_count = sum(
        item["parsed_candidates"]
        for item in question_records
    )
    correct_count = sum(
        item["correct_candidates"]
        for item in question_records
    )

    n_values = {
        len(item["candidates"])
        for item in question_records
    }
    if n_values != {16}:
        raise RuntimeError(
            f"候选数不一致：{n_values}"
        )

    standard_pass = {}
    prefix_pass = {}

    for k in K_VALUES:
        standard_pass[f"pass_at_{k}"] = float(
            np.mean([
                pass_at_k(
                    16,
                    item["correct_candidates"],
                    k,
                )
                for item in question_records
            ])
        )
        prefix_pass[f"prefix_pass_at_{k}"] = float(
            np.mean([
                any(
                    candidate["correct"]
                    for candidate
                    in item["candidates"]
                    if candidate[
                        "candidate_index"
                    ] < k
                )
                for item in question_records
            ])
        )

    all_wrong = sum(
        item["correct_candidates"] == 0
        for item in question_records
    )
    all_correct = sum(
        item["correct_candidates"] == 16
        for item in question_records
    )
    mixed = sum(
        0 < item["correct_candidates"] < 16
        for item in question_records
    )

    majority_correct = sum(
        item["majority_correct"]
        for item in question_records
    )
    majority_ties = sum(
        item["majority_tie_count"] > 1
        for item in question_records
    )

    positive_support = [
        item["correct_candidates"]
        for item in question_records
        if item["correct_candidates"] > 0
    ]

    parse_methods = Counter()
    correct_by_method = Counter()
    for item in question_records:
        for candidate in item["candidates"]:
            method = candidate["parse_method"]
            parse_methods[method] += 1
            if candidate["correct"]:
                correct_by_method[method] += 1

    return {
        "questions": len(question_records),
        "candidates": candidate_count,
        "candidate_parse_rate": (
            parsed_count / candidate_count
        ),
        "candidate_accuracy": (
            correct_count / candidate_count
        ),
        "accuracy_among_parsed": (
            correct_count / parsed_count
            if parsed_count
            else 0.0
        ),
        "standard_pass_at_k": standard_pass,
        "fixed_prefix_pass_at_k": prefix_pass,
        "oracle_pass_at_16": (
            1.0 - all_wrong / len(question_records)
        ),
        "question_composition": {
            "all_wrong": all_wrong,
            "mixed": mixed,
            "all_correct": all_correct,
            "mixed_question_coverage": (
                mixed / len(question_records)
            ),
        },
        "majority_cluster": {
            "top1": (
                majority_correct
                / len(question_records)
            ),
            "tie_rate": (
                majority_ties
                / len(question_records)
            ),
        },
        "correct_cluster_support": (
            {
                "questions_with_correct": len(
                    positive_support
                ),
                "min": min(positive_support),
                "median": float(
                    np.median(positive_support)
                ),
                "max": max(positive_support),
                "mean": float(
                    np.mean(positive_support)
                ),
            }
            if positive_support
            else {
                "questions_with_correct": 0,
                "min": None,
                "median": None,
                "max": None,
                "mean": None,
            }
        ),
        "parse_methods": dict(parse_methods),
        "correct_by_parse_method": dict(
            correct_by_method
        ),
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"为避免覆盖首次 reveal，已停止："
            f"{OUTPUT_PATH}"
        )

    candidate_manifest = json.loads(
        CANDIDATE_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    candidate_sha256 = sha256_file(
        CANDIDATE_PATH
    )

    if candidate_sha256 != candidate_manifest[
        "output_sha256"
    ]:
        raise RuntimeError(
            "候选文件与冻结哈希不一致"
        )

    candidates = read_jsonl(CANDIDATE_PATH)

    # 从这里开始首次读取冻结标签。
    labels = read_jsonl(LABEL_PATH)

    label_by_uid = {
        row["question_uid"]: row
        for row in labels
    }
    if len(label_by_uid) != 63:
        raise RuntimeError(
            f"标签题数异常：{len(label_by_uid)}"
        )

    groups = defaultdict(list)

    for row in candidates:
        uid = row["question_uid"]
        if uid not in label_by_uid:
            raise RuntimeError(
                f"候选无法连接标签：{uid}"
            )

        prediction, method = extract_answer(
            row["solution_text"],
            "math",
        )

        # Fresh Math 的 last_number 仅是覆盖率诊断，
        # 不能作为正式答案解析结果。
        if method == "last_number":
            prediction = None
            method = "last_number_rejected"

        gold = normalize_answer(
            label_by_uid[uid]["answer"],
            "math",
        )
        if gold is None:
            raise RuntimeError(
                f"标准答案无法规范化：{uid}"
            )

        groups[uid].append({
            "candidate_index": int(
                row["candidate_index"]
            ),
            "normalized_prediction": prediction,
            "parse_method": method,
            "correct": prediction == gold,
        })

    if set(groups) != set(label_by_uid):
        raise RuntimeError(
            "候选问题与标签问题集合不一致"
        )

    question_records = []

    for uid, rows in groups.items():
        rows.sort(
            key=lambda item: item[
                "candidate_index"
            ]
        )
        if [
            row["candidate_index"]
            for row in rows
        ] != list(range(16)):
            raise RuntimeError(
                f"候选索引异常：{uid}"
            )

        gold = normalize_answer(
            label_by_uid[uid]["answer"],
            "math",
        )
        majority, support, tie_count = (
            majority_prediction(rows)
        )
        correct_candidates = sum(
            row["correct"]
            for row in rows
        )

        question_records.append({
            "question_uid": uid,
            "source_dataset": label_by_uid[uid][
                "source_dataset"
            ],
            "problem_id": label_by_uid[uid][
                "problem_id"
            ],
            "parsed_candidates": sum(
                row["normalized_prediction"]
                is not None
                for row in rows
            ),
            "correct_candidates": (
                correct_candidates
            ),
            "majority_support": support,
            "majority_tie_count": tie_count,
            "majority_correct": majority == gold,
            "candidates": rows,
        })

    question_records.sort(
        key=lambda item: (
            item["source_dataset"],
            str(item["problem_id"]),
        )
    )

    datasets = {}
    for dataset_name in [
        "AIME_2026",
        "HMMT_FEB_2026",
    ]:
        subset = [
            item
            for item in question_records
            if item["source_dataset"]
            == dataset_name
        ]
        datasets[dataset_name] = summarize(
            subset
        )

    pooled = summarize(question_records)

    macro = {}
    for metric in [
        "candidate_parse_rate",
        "candidate_accuracy",
        "oracle_pass_at_16",
    ]:
        macro[metric] = float(np.mean([
            datasets[name][metric]
            for name in datasets
        ]))

    for k in K_VALUES:
        key = f"pass_at_{k}"
        macro[key] = float(np.mean([
            datasets[name][
                "standard_pass_at_k"
            ][key]
            for name in datasets
        ]))

    macro["majority_cluster_top1"] = float(
        np.mean([
            datasets[name][
                "majority_cluster"
            ]["top1"]
            for name in datasets
        ])
    )
    macro["mixed_question_coverage"] = float(
        np.mean([
            datasets[name][
                "question_composition"
            ]["mixed_question_coverage"]
            for name in datasets
        ])
    )

    output = {
        "version": (
            "fresh_math_2026_qwen3_8b_adaptive_oracle_strict_v1"
        ),
        "label_reveal": {
            "revealed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "candidate_freeze_tag": FREEZE_TAG,
            "candidate_freeze_commit": (
                git_revision(FREEZE_TAG)
            ),
            "candidate_sha256": candidate_sha256,
            "sealed_label_sha256": sha256_file(
                LABEL_PATH
            ),
            "method_or_rm_scores_used": False,
            "hyperparameters_selected": False,
        },
        "parser": {
            "source": (
                "scripts/"
                "audit_answer_cluster_consensus.py"
            ),
            "family": "math",
            "rule": (
                "same frozen parser used by final "
                "answer-cluster experiments; "
                "last_number fallback rejected"
            ),
        },
        "datasets": datasets,
        "pooled": pooled,
        "dataset_macro": macro,
        "question_records": [
            {
                key: value
                for key, value in item.items()
                if key != "candidates"
            }
            for item in question_records
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== Track B 严格解析 Oracle 复核 =====")
    for name, result in datasets.items():
        print()
        print(name)
        print(json.dumps(
            {
                "questions": result["questions"],
                "candidate_parse_rate": (
                    result["candidate_parse_rate"]
                ),
                "candidate_accuracy": (
                    result["candidate_accuracy"]
                ),
                "standard_pass_at_k": (
                    result["standard_pass_at_k"]
                ),
                "oracle_pass_at_16": (
                    result["oracle_pass_at_16"]
                ),
                "question_composition": (
                    result[
                        "question_composition"
                    ]
                ),
                "majority_cluster": (
                    result["majority_cluster"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        ))

    print()
    print("===== Dataset Macro =====")
    print(json.dumps(
        macro,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("结果：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
