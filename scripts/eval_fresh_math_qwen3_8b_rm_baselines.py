from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

from audit_answer_cluster_consensus import (
    normalize_answer,
)


CANDIDATE_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_candidates.jsonl"
)
SCORES_PATH = (
    ROOT
    / "data/cache/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_rm_1p7b/"
    "scores_f32.npy"
)
LABEL_PATH = (
    ROOT
    / "data/external/fresh_math_2026/"
    "sealed_labels.jsonl"
)
OUTPUT_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_rm_baselines.json"
)

EXPECTED_CANDIDATE_SHA256 = (
    "9af1176c0b092608f413d90889315c03"
    "ef4ad5d3d994ad8746dadd5e131f4257"
)
EXPECTED_SCORE_SHA256 = (
    "c0f6776946aab3283c667d2d503d3027"
    "470419309734c833f11353ab59e555e4"
)
EXPECTED_CANDIDATES = 1008
EXPECTED_QUESTIONS = 63
EXPECTED_K = 16

TAU = 4.0
K_VALUES = [1, 4, 8, 16]
METHODS = [
    "raw_rm",
    "majority_rm_tiebreak",
    "weighted_tau4",
]


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
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_logsumexp(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    maximum = float(np.max(values))
    return float(
        maximum
        + np.log(
            np.exp(values - maximum).sum()
        )
    )


def cluster_key(candidate):
    if candidate["answer"] is None:
        # 未解析候选必须各自成为单例簇，
        # 不能把所有未解析回答合并。
        return (
            "__unparsed__",
            candidate["candidate_index"],
        )
    return ("answer", candidate["answer"])


def group_subset(question, subset):
    groups = defaultdict(list)
    for local_index in subset:
        candidate = question[
            "candidates"
        ][local_index]
        groups[
            cluster_key(candidate)
        ].append(candidate)
    return groups


def best_raw_candidate(candidates):
    return max(
        candidates,
        key=lambda item: (
            item["score"],
            -item["candidate_index"],
        ),
    )


def select_candidate(
    question,
    subset,
    method,
):
    subset = tuple(subset)

    if method == "raw_rm":
        return best_raw_candidate([
            question["candidates"][index]
            for index in subset
        ])

    groups = group_subset(
        question,
        subset,
    )

    if method == "majority_rm_tiebreak":
        selected_group = max(
            groups.values(),
            key=lambda group: (
                len(group),
                max(
                    item["score"]
                    for item in group
                ),
                -min(
                    item["candidate_index"]
                    for item in group
                ),
            ),
        )
        return best_raw_candidate(
            selected_group
        )

    if method == "weighted_tau4":
        def group_rank(group):
            aggregate = (
                TAU
                * stable_logsumexp([
                    item["score"] / TAU
                    for item in group
                ])
            )
            return (
                aggregate,
                max(
                    item["score"]
                    for item in group
                ),
                -min(
                    item["candidate_index"]
                    for item in group
                ),
            )

        selected_group = max(
            groups.values(),
            key=group_rank,
        )
        return best_raw_candidate(
            selected_group
        )

    raise KeyError(method)


def unique_majority(question):
    groups = group_subset(
        question,
        range(len(question["candidates"])),
    )
    maximum = max(
        len(group)
        for group in groups.values()
    )
    tied = [
        group
        for group in groups.values()
        if len(group) == maximum
    ]

    if len(tied) != 1:
        optimistic_correct = any(
            any(
                item["correct"]
                for item in group
            )
            for group in tied
        )
        return {
            "tie": True,
            "selected": None,
            "correct": False,
            "optimistic_correct": (
                optimistic_correct
            ),
        }

    selected = best_raw_candidate(tied[0])
    return {
        "tie": False,
        "selected": selected,
        "correct": bool(
            selected["correct"]
        ),
        "optimistic_correct": bool(
            selected["correct"]
        ),
    }


def candidate_rank_key(
    question,
    candidate,
    method,
):
    if method == "raw_rm":
        # Pair-Macro Strict 中同分不借助
        # 候选索引打破平局。
        return (
            float(candidate["score"]),
        )

    groups = group_subset(
        question,
        range(len(question["candidates"])),
    )
    group = groups[
        cluster_key(candidate)
    ]

    if method == "majority_rm_tiebreak":
        return (
            float(len(group)),
            float(candidate["score"]),
        )

    if method == "weighted_tau4":
        aggregate = (
            TAU
            * stable_logsumexp([
                item["score"] / TAU
                for item in group
            ])
        )
        return (
            aggregate,
            float(candidate["score"]),
        )

    raise KeyError(method)


def pair_metric(question, method):
    positives = [
        item
        for item in question["candidates"]
        if item["correct"]
    ]
    negatives = [
        item
        for item in question["candidates"]
        if not item["correct"]
    ]

    if not positives or not negatives:
        return None

    wins = 0
    total = 0

    for positive in positives:
        positive_key = candidate_rank_key(
            question,
            positive,
            method,
        )
        for negative in negatives:
            negative_key = candidate_rank_key(
                question,
                negative,
                method,
            )
            wins += int(
                positive_key > negative_key
            )
            total += 1

    return wins / total


def pass_probability(
    candidate_count,
    correct_count,
    k,
):
    if correct_count == 0:
        return 0.0
    if k >= candidate_count:
        return 1.0
    if candidate_count - correct_count < k:
        return 1.0

    return 1.0 - (
        math.comb(
            candidate_count - correct_count,
            k,
        )
        / math.comb(candidate_count, k)
    )


def exact_budget_metrics(
    question,
    k,
):
    candidate_count = len(
        question["candidates"]
    )
    subsets = combinations(
        range(candidate_count),
        k,
    )

    correct = {
        method: 0
        for method in METHODS
    }
    total = 0

    for subset in subsets:
        total += 1
        for method in METHODS:
            selected = select_candidate(
                question,
                subset,
                method,
            )
            correct[method] += int(
                selected["correct"]
            )

    return {
        method: correct[method] / total
        for method in METHODS
    }


def risk_relative_to_raw(
    questions,
    method,
):
    raw_correct = 0
    raw_wrong = 0
    corrections = 0
    damages = 0
    changed = 0

    for question in questions:
        raw = select_candidate(
            question,
            range(EXPECTED_K),
            "raw_rm",
        )
        selected = select_candidate(
            question,
            range(EXPECTED_K),
            method,
        )

        raw_is_correct = bool(
            raw["correct"]
        )
        selected_is_correct = bool(
            selected["correct"]
        )

        raw_correct += int(raw_is_correct)
        raw_wrong += int(not raw_is_correct)
        corrections += int(
            (not raw_is_correct)
            and selected_is_correct
        )
        damages += int(
            raw_is_correct
            and (not selected_is_correct)
        )
        changed += int(
            raw["candidate_index"]
            != selected["candidate_index"]
        )

    return {
        "changed_questions": changed,
        "switch_rate": changed / len(questions),
        "corrections": corrections,
        "damages": damages,
        "net_corrected": (
            corrections - damages
        ),
        "correction_rate": (
            corrections / raw_wrong
            if raw_wrong else 0.0
        ),
        "damage_rate": (
            damages / raw_correct
            if raw_correct else 0.0
        ),
    }


def summarize_questions(
    questions,
):
    candidate_count = sum(
        len(question["candidates"])
        for question in questions
    )
    parsed_count = sum(
        item["answer"] is not None
        for question in questions
        for item in question["candidates"]
    )
    correct_count = sum(
        item["correct"]
        for question in questions
        for item in question["candidates"]
    )

    composition = Counter()
    unique_majority_correct = 0
    optimistic_majority_correct = 0
    majority_ties = 0

    top1 = {
        method: 0
        for method in METHODS
    }
    per_question = []

    pair_values = {
        method: []
        for method in METHODS
    }

    budgets = {
        method: {
            k: []
            for k in K_VALUES
        }
        for method in METHODS
    }
    pass_at_k = {
        k: []
        for k in K_VALUES
    }

    for question in questions:
        number_correct = sum(
            item["correct"]
            for item in question["candidates"]
        )

        if number_correct == 0:
            composition["all_wrong"] += 1
        elif number_correct == EXPECTED_K:
            composition["all_correct"] += 1
        else:
            composition["mixed"] += 1

        majority = unique_majority(
            question
        )
        majority_ties += int(
            majority["tie"]
        )
        unique_majority_correct += int(
            majority["correct"]
        )
        optimistic_majority_correct += int(
            majority["optimistic_correct"]
        )

        selected_record = {}

        for method in METHODS:
            selected = select_candidate(
                question,
                range(EXPECTED_K),
                method,
            )
            top1[method] += int(
                selected["correct"]
            )
            selected_record[method] = {
                "candidate_index": (
                    selected["candidate_index"]
                ),
                "correct": bool(
                    selected["correct"]
                ),
                "answer": selected["answer"],
                "score": float(
                    selected["score"]
                ),
            }

            pair_value = pair_metric(
                question,
                method,
            )
            if pair_value is not None:
                pair_values[method].append(
                    pair_value
                )

        for k in K_VALUES:
            pass_at_k[k].append(
                pass_probability(
                    EXPECTED_K,
                    number_correct,
                    k,
                )
            )
            exact = exact_budget_metrics(
                question,
                k,
            )
            for method in METHODS:
                budgets[method][k].append(
                    exact[method]
                )

        per_question.append({
            "question_uid": question[
                "question_uid"
            ],
            "source_dataset": question[
                "source_dataset"
            ],
            "problem_id": question[
                "problem_id"
            ],
            "parsed_candidates": sum(
                item["answer"] is not None
                for item in question["candidates"]
            ),
            "correct_candidates": (
                number_correct
            ),
            "oracle_correct": (
                number_correct > 0
            ),
            "unique_majority_tie": (
                majority["tie"]
            ),
            "unique_majority_correct": (
                majority["correct"]
            ),
            "selected": selected_record,
        })

    question_count = len(questions)

    methods = {}
    for method in METHODS:
        methods[method] = {
            "top1": (
                top1[method]
                / question_count
            ),
            "top1_correct_questions": (
                top1[method]
            ),
            "pair_macro_strict": (
                float(np.mean(
                    pair_values[method]
                ))
                if pair_values[method]
                else None
            ),
            "pair_questions": len(
                pair_values[method]
            ),
            "best_at_k": {
                f"best_at_{k}": float(
                    np.mean(
                        budgets[method][k]
                    )
                )
                for k in K_VALUES
            },
        }

        if method != "raw_rm":
            methods[method][
                "relative_to_raw_rm"
            ] = risk_relative_to_raw(
                questions,
                method,
            )

    return {
        "questions": question_count,
        "candidates": candidate_count,
        "candidate_parse_rate": (
            parsed_count / candidate_count
        ),
        "candidate_accuracy": (
            correct_count / candidate_count
        ),
        "question_composition": {
            "all_wrong": composition[
                "all_wrong"
            ],
            "mixed": composition["mixed"],
            "all_correct": composition[
                "all_correct"
            ],
        },
        "oracle_pass_at_16": (
            sum(
                question[
                    "correct_candidates"
                ] > 0
                for question in per_question
            )
            / question_count
        ),
        "standard_pass_at_k": {
            f"pass_at_{k}": float(
                np.mean(pass_at_k[k])
            )
            for k in K_VALUES
        },
        "unique_majority": {
            "top1": (
                unique_majority_correct
                / question_count
            ),
            "correct_questions": (
                unique_majority_correct
            ),
            "optimistic_tie_top1": (
                optimistic_majority_correct
                / question_count
            ),
            "tie_rate": (
                majority_ties
                / question_count
            ),
            "tie_questions": majority_ties,
        },
        "methods": methods,
        "per_question": per_question,
    }


def main():
    started = time.time()

    if (
        sha256_file(CANDIDATE_PATH)
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            "候选文件 SHA256 不匹配"
        )
    if (
        sha256_file(SCORES_PATH)
        != EXPECTED_SCORE_SHA256
    ):
        raise RuntimeError(
            "RM 分数 SHA256 不匹配"
        )

    rows = read_jsonl(CANDIDATE_PATH)
    labels = read_jsonl(LABEL_PATH)
    scores = np.asarray(
        np.load(SCORES_PATH),
        dtype=np.float32,
    )

    if len(rows) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"候选数错误：{len(rows)}"
        )
    if scores.shape != (
        EXPECTED_CANDIDATES,
    ):
        raise RuntimeError(
            f"分数 shape 错误："
            f"{scores.shape}"
        )

    gold_by_uid = {}
    for row in labels:
        uid = str(row["question_uid"])
        if uid in gold_by_uid:
            raise RuntimeError(
                f"重复标签：{uid}"
            )
        gold_by_uid[uid] = normalize_answer(
            row["answer"],
            "math",
        )

    grouped_rows = defaultdict(list)

    for row_index, (row, score) in enumerate(
        zip(rows, scores)
    ):
        uid = str(row["question_uid"])
        if uid not in gold_by_uid:
            raise RuntimeError(
                f"缺少标签：{uid}"
            )

        raw_answer = row.get(
            "parsed_answer_generation_audit"
        )
        answer = (
            normalize_answer(
                raw_answer,
                "math",
            )
            if raw_answer is not None
            else None
        )

        grouped_rows[uid].append({
            "row_index": row_index,
            "candidate_index": int(
                row["candidate_index"]
            ),
            "answer": answer,
            "correct": bool(
                answer is not None
                and answer
                == gold_by_uid[uid]
            ),
            "score": float(score),
        })

    if len(grouped_rows) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"问题数错误："
            f"{len(grouped_rows)}"
        )

    metadata = {}
    for row in rows:
        uid = str(row["question_uid"])
        metadata.setdefault(uid, {
            "question_uid": uid,
            "source_dataset": str(
                row["source_dataset"]
            ),
            "problem_id": row[
                "problem_id"
            ],
        })

    questions = []
    for uid, candidates in grouped_rows.items():
        candidates.sort(
            key=lambda item: item[
                "candidate_index"
            ]
        )
        if [
            item["candidate_index"]
            for item in candidates
        ] != list(range(EXPECTED_K)):
            raise RuntimeError(
                f"{uid} 候选索引错误"
            )

        questions.append({
            **metadata[uid],
            "candidates": candidates,
        })

    dataset_groups = defaultdict(list)
    for question in questions:
        dataset_groups[
            question["source_dataset"]
        ].append(question)

    results = {
        name: summarize_questions(
            dataset_questions
        )
        for name, dataset_questions
        in sorted(dataset_groups.items())
    }
    pooled = summarize_questions(
        questions
    )

    # 复现冻结生成解析结果，防止标签、顺序或
    # 解析口径在 RM 评估中发生漂移。
    assert pooled["candidates"] == 1008
    assert (
        pooled["candidate_parse_rate"]
        == 924 / 1008
    )
    assert (
        pooled["candidate_accuracy"]
        == 146 / 1008
    )
    assert (
        pooled["oracle_pass_at_16"]
        == 23 / 63
    )
    assert (
        pooled["unique_majority"][
            "correct_questions"
        ]
        == 13
    )
    assert (
        pooled["unique_majority"][
            "tie_questions"
        ]
        == 7
    )
    assert pooled[
        "question_composition"
    ] == {
        "all_wrong": 40,
        "mixed": 21,
        "all_correct": 2,
    }

    output = {
        "version": (
            "fresh_math_2026_qwen3_8b_"
            "rm_baselines_v2"
        ),
        "evaluation_status": (
            "post_track_a_reveal_exploratory"
        ),
        "labels_loaded_for_evaluation": True,
        "labels_used_for_generation": False,
        "candidate_parser": (
            "frozen parsed_answer_"
            "generation_audit"
        ),
        "candidate_file": str(
            CANDIDATE_PATH.relative_to(ROOT)
        ),
        "candidate_sha256": (
            EXPECTED_CANDIDATE_SHA256
        ),
        "score_file": str(
            SCORES_PATH.relative_to(ROOT)
        ),
        "score_sha256": EXPECTED_SCORE_SHA256,
        "reward_model": (
            "Skywork-Reward-V2-Qwen3-1.7B"
        ),
        "weighted_consensus_tau": TAU,
        "budget_protocol": (
            "exact expectation over every "
            "uniform subset without replacement"
        ),
        "pair_protocol": (
            "macro strict over mixed-label "
            "questions only; exact score ties "
            "count as non-wins"
        ),
        "supersedes": (
            "v1 used candidate-index tie-breaking "
            "inside the reported strict Pair metric"
        ),
        "datasets": results,
        "pooled": pooled,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "git_head": git_head(),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    atomic_json(OUTPUT_PATH, output)

    print("===== Fresh Math RM 基线 =====")
    for name, result in [
        *results.items(),
        ("POOLED", pooled),
    ]:
        print()
        print(name)
        print(json.dumps({
            "questions": result["questions"],
            "candidate_accuracy": result[
                "candidate_accuracy"
            ],
            "oracle_pass_at_16": result[
                "oracle_pass_at_16"
            ],
            "unique_majority": result[
                "unique_majority"
            ],
            "raw_rm": result["methods"][
                "raw_rm"
            ],
            "majority_rm_tiebreak": (
                result["methods"][
                    "majority_rm_tiebreak"
                ]
            ),
            "weighted_tau4": (
                result["methods"][
                    "weighted_tau4"
                ]
            ),
        }, ensure_ascii=False, indent=2))

    print()
    print("结果：", OUTPUT_PATH)
    print(
        "耗时秒：",
        output["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
