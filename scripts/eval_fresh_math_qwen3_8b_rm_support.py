from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import subprocess
import time

import numpy as np

from audit_answer_cluster_consensus import (
    normalize_answer,
)


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = (
    ROOT
    / "outputs/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_candidates.jsonl"
)
RAW_SCORE_PATH = (
    ROOT
    / "data/cache/fresh_math_2026/"
    "qwen3_8b_adaptive_k16_rm_1p7b/"
    "scores_f32.npy"
)
PREDICTION_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_"
    "rm_support_predictions.json"
)
BASELINE_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_"
    "rm_baselines.json"
)
LABEL_PATH = (
    ROOT
    / "data/external/fresh_math_2026/"
    "sealed_labels.jsonl"
)
OUTPUT_PATH = (
    ROOT
    / "data/manifests/"
    "fresh_math_2026_qwen3_8b_"
    "rm_support_evaluation.json"
)

EXPECTED_CANDIDATE_SHA256 = (
    "9af1176c0b092608f413d90889315c03"
    "ef4ad5d3d994ad8746dadd5e131f4257"
)
EXPECTED_RAW_SCORE_SHA256 = (
    "c0f6776946aab3283c667d2d503d3027"
    "470419309734c833f11353ab59e555e4"
)
EXPECTED_METHOD_SCORE_SHA256 = (
    "11d17068f84a9964360d1b3b5b311661"
    "b43852ed386bfcb929b6d9c22577238e"
)

K_VALUES = [1, 4, 8, 16]
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260902


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


def top_candidate(candidates, scores):
    local_index = int(np.argmax(scores))
    return candidates[local_index], local_index


def pair_strict(candidates, scores):
    positives = [
        index
        for index, candidate
        in enumerate(candidates)
        if candidate["correct"]
    ]
    negatives = [
        index
        for index, candidate
        in enumerate(candidates)
        if not candidate["correct"]
    ]

    if not positives or not negatives:
        return None

    wins = 0
    total = 0

    for positive in positives:
        for negative in negatives:
            wins += int(
                scores[positive]
                > scores[negative]
            )
            total += 1

    return wins / total


def selector_probability(
    candidates,
    scores,
    k,
):
    candidate_count = len(candidates)

    order = sorted(
        range(candidate_count),
        key=lambda index: (
            -float(scores[index]),
            candidates[index][
                "candidate_index"
            ],
        ),
    )

    denominator = math.comb(
        candidate_count,
        k,
    )
    probability = 0.0

    for rank, local_index in enumerate(order):
        if not candidates[local_index][
            "correct"
        ]:
            continue

        lower_count = (
            candidate_count - rank - 1
        )
        if lower_count < k - 1:
            continue

        probability += (
            math.comb(
                lower_count,
                k - 1,
            )
            / denominator
        )

    return probability


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


def build_question_records(
    rows,
    raw_scores,
    method_scores,
    gold_by_uid,
    frozen_selections,
):
    grouped = defaultdict(list)

    for global_index, row in enumerate(rows):
        uid = str(row["question_uid"])
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

        grouped[uid].append({
            "global_index": global_index,
            "candidate_index": int(
                row["candidate_index"]
            ),
            "answer": answer,
            "correct": bool(
                answer is not None
                and answer
                == gold_by_uid[uid]
            ),
            "raw_score": float(
                raw_scores[global_index]
            ),
            "method_score": float(
                method_scores[global_index]
            ),
            "source_dataset": str(
                row["source_dataset"]
            ),
            "problem_id": row["problem_id"],
        })

    records = []

    for uid, candidates in grouped.items():
        candidates.sort(
            key=lambda item: item[
                "candidate_index"
            ]
        )

        if [
            item["candidate_index"]
            for item in candidates
        ] != list(range(16)):
            raise RuntimeError(
                f"{uid}: 候选索引错误"
            )

        raw_vector = np.asarray([
            item["raw_score"]
            for item in candidates
        ], dtype=np.float32)
        method_vector = np.asarray([
            item["method_score"]
            for item in candidates
        ], dtype=np.float32)

        raw_selected, raw_local = (
            top_candidate(
                candidates,
                raw_vector,
            )
        )
        method_selected, method_local = (
            top_candidate(
                candidates,
                method_vector,
            )
        )

        frozen = frozen_selections[uid]
        if (
            method_selected[
                "candidate_index"
            ]
            != frozen[
                "selected_candidate_index"
            ]
        ):
            raise RuntimeError(
                f"{uid}: 冻结选择与方法分数不一致"
            )
        if (
            raw_selected[
                "candidate_index"
            ]
            != frozen[
                "raw_candidate_index"
            ]
        ):
            raise RuntimeError(
                f"{uid}: 冻结 Raw 选择不一致"
            )

        correct_count = sum(
            item["correct"]
            for item in candidates
        )

        raw_pair = pair_strict(
            candidates,
            raw_vector,
        )
        method_pair = pair_strict(
            candidates,
            method_vector,
        )

        records.append({
            "question_uid": uid,
            "source_dataset": candidates[0][
                "source_dataset"
            ],
            "problem_id": candidates[0][
                "problem_id"
            ],
            "parsed_candidates": sum(
                item["answer"] is not None
                for item in candidates
            ),
            "correct_candidates": correct_count,
            "oracle_correct": (
                correct_count > 0
            ),
            "raw_candidate_index": (
                raw_selected[
                    "candidate_index"
                ]
            ),
            "method_candidate_index": (
                method_selected[
                    "candidate_index"
                ]
            ),
            "switched": (
                raw_local != method_local
            ),
            "raw_correct": bool(
                raw_selected["correct"]
            ),
            "method_correct": bool(
                method_selected["correct"]
            ),
            "top1_delta": (
                int(method_selected["correct"])
                - int(raw_selected["correct"])
            ),
            "raw_pair": raw_pair,
            "method_pair": method_pair,
            "pair_delta": (
                method_pair - raw_pair
                if raw_pair is not None
                else None
            ),
            "pass_at_k": {
                str(k): pass_probability(
                    16,
                    correct_count,
                    k,
                )
                for k in K_VALUES
            },
            "raw_best_at_k": {
                str(k): selector_probability(
                    candidates,
                    raw_vector,
                    k,
                )
                for k in K_VALUES
            },
            "method_best_at_k": {
                str(k): selector_probability(
                    candidates,
                    method_vector,
                    k,
                )
                for k in K_VALUES
            },
        })

    return records


def summarize(records):
    question_count = len(records)

    raw_correct = sum(
        row["raw_correct"]
        for row in records
    )
    method_correct = sum(
        row["method_correct"]
        for row in records
    )
    corrections = sum(
        (not row["raw_correct"])
        and row["method_correct"]
        for row in records
    )
    damages = sum(
        row["raw_correct"]
        and (not row["method_correct"])
        for row in records
    )
    switches = sum(
        row["switched"]
        for row in records
    )

    pair_rows = [
        row
        for row in records
        if row["raw_pair"] is not None
    ]

    return {
        "questions": question_count,
        "candidate_parse_rate": (
            sum(
                row["parsed_candidates"]
                for row in records
            )
            / (question_count * 16)
        ),
        "candidate_accuracy": (
            sum(
                row["correct_candidates"]
                for row in records
            )
            / (question_count * 16)
        ),
        "oracle_pass_at_16": (
            sum(
                row["oracle_correct"]
                for row in records
            )
            / question_count
        ),
        "raw_rm": {
            "top1": (
                raw_correct / question_count
            ),
            "correct_questions": raw_correct,
            "pair_macro_strict": (
                float(np.mean([
                    row["raw_pair"]
                    for row in pair_rows
                ]))
            ),
            "pair_questions": len(
                pair_rows
            ),
            "best_at_k": {
                f"best_at_{k}": float(
                    np.mean([
                        row[
                            "raw_best_at_k"
                        ][str(k)]
                        for row in records
                    ])
                )
                for k in K_VALUES
            },
        },
        "rm_support": {
            "top1": (
                method_correct
                / question_count
            ),
            "correct_questions": (
                method_correct
            ),
            "top1_delta": (
                (method_correct - raw_correct)
                / question_count
            ),
            "pair_macro_strict": (
                float(np.mean([
                    row["method_pair"]
                    for row in pair_rows
                ]))
            ),
            "pair_delta": float(
                np.mean([
                    row["pair_delta"]
                    for row in pair_rows
                ])
            ),
            "pair_questions": len(
                pair_rows
            ),
            "best_at_k": {
                f"best_at_{k}": float(
                    np.mean([
                        row[
                            "method_best_at_k"
                        ][str(k)]
                        for row in records
                    ])
                )
                for k in K_VALUES
            },
            "best_at_k_delta": {
                f"best_at_{k}_delta": float(
                    np.mean([
                        row[
                            "method_best_at_k"
                        ][str(k)]
                        - row[
                            "raw_best_at_k"
                        ][str(k)]
                        for row in records
                    ])
                )
                for k in K_VALUES
            },
            "switches": switches,
            "switch_rate": (
                switches / question_count
            ),
            "corrections": corrections,
            "damages": damages,
            "net_corrected": (
                corrections - damages
            ),
            "correction_rate": (
                corrections
                / (question_count - raw_correct)
                if raw_correct < question_count
                else 0.0
            ),
            "damage_rate": (
                damages / raw_correct
                if raw_correct else 0.0
            ),
        },
    }


def bootstrap(records):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )
    question_count = len(records)

    top1_delta = np.asarray([
        row["top1_delta"]
        for row in records
    ], dtype=np.float64)

    best_deltas = {
        k: np.asarray([
            row["method_best_at_k"][str(k)]
            - row["raw_best_at_k"][str(k)]
            for row in records
        ], dtype=np.float64)
        for k in K_VALUES
    }

    pair_delta = np.asarray([
        row["pair_delta"]
        for row in records
        if row["pair_delta"] is not None
    ], dtype=np.float64)

    indices = rng.integers(
        0,
        question_count,
        size=(
            BOOTSTRAP_SAMPLES,
            question_count,
        ),
    )
    top_boot = np.mean(
        top1_delta[indices],
        axis=1,
    )

    result = {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "top1_delta": {
            "point": float(
                np.mean(top1_delta)
            ),
            "ci95": [
                float(np.percentile(
                    top_boot, 2.5
                )),
                float(np.percentile(
                    top_boot, 97.5
                )),
            ],
            "probability_positive": float(
                np.mean(top_boot > 0)
            ),
        },
        "best_at_k_delta": {},
    }

    for k, values in best_deltas.items():
        boot = np.mean(
            values[indices],
            axis=1,
        )
        result["best_at_k_delta"][
            f"best_at_{k}_delta"
        ] = {
            "point": float(
                np.mean(values)
            ),
            "ci95": [
                float(np.percentile(
                    boot, 2.5
                )),
                float(np.percentile(
                    boot, 97.5
                )),
            ],
        }

    if len(pair_delta):
        pair_indices = rng.integers(
            0,
            len(pair_delta),
            size=(
                BOOTSTRAP_SAMPLES,
                len(pair_delta),
            ),
        )
        pair_boot = np.mean(
            pair_delta[pair_indices],
            axis=1,
        )
        result["pair_delta"] = {
            "point": float(
                np.mean(pair_delta)
            ),
            "ci95": [
                float(np.percentile(
                    pair_boot, 2.5
                )),
                float(np.percentile(
                    pair_boot, 97.5
                )),
            ],
            "questions": len(pair_delta),
        }

    return result


def main():
    started = time.time()

    if (
        sha256_file(CANDIDATE_PATH)
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            "候选 SHA256 不匹配"
        )
    if (
        sha256_file(RAW_SCORE_PATH)
        != EXPECTED_RAW_SCORE_SHA256
    ):
        raise RuntimeError(
            "Raw RM SHA256 不匹配"
        )

    prediction = json.loads(
        PREDICTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    if prediction["fresh_labels_loaded"]:
        raise RuntimeError(
            "预测阶段读取了 Fresh 标签"
        )
    if (
        prediction["method_scores"]["sha256"]
        != EXPECTED_METHOD_SCORE_SHA256
    ):
        raise RuntimeError(
            "冻结方法分数 SHA256 不匹配"
        )

    rows = read_jsonl(CANDIDATE_PATH)
    raw_scores = np.asarray(
        np.load(RAW_SCORE_PATH),
        dtype=np.float32,
    )
    method_scores = np.asarray(
        prediction["method_scores"]["values"],
        dtype=np.float32,
    )

    labels = read_jsonl(LABEL_PATH)
    gold_by_uid = {
        str(row["question_uid"]):
        normalize_answer(
            row["answer"],
            "math",
        )
        for row in labels
    }

    frozen_selections = {
        str(row["question_uid"]): row
        for row in prediction["selections"]
    }

    records = build_question_records(
        rows,
        raw_scores,
        method_scores,
        gold_by_uid,
        frozen_selections,
    )

    if len(records) != 63:
        raise RuntimeError(
            f"问题数错误：{len(records)}"
        )

    groups = defaultdict(list)
    for row in records:
        groups[
            row["source_dataset"]
        ].append(row)

    datasets = {
        name: summarize(items)
        for name, items
        in sorted(groups.items())
    }
    pooled = summarize(records)
    bootstrap_result = bootstrap(records)

    # 再次复现已经冻结的生成器与 Raw RM 基线。
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
        pooled["raw_rm"][
            "correct_questions"
        ]
        == 14
    )
    assert (
        pooled["rm_support"]["switches"]
        == prediction["switch_count"]
        == 13
    )

    baseline = json.loads(
        BASELINE_PATH.read_text(
            encoding="utf-8"
        )
    )

    output = {
        "version": (
            "fresh_math_2026_qwen3_8b_"
            "rm_support_evaluation_v1"
        ),
        "evaluation_status": (
            "post_track_a_reveal_exploratory"
        ),
        "prediction_frozen_before_evaluation": True,
        "prediction_manifest": str(
            PREDICTION_PATH.relative_to(ROOT)
        ),
        "prediction_git_tag": (
            "fresh-math-2026-qwen3-"
            "rm-support-predictions-v1"
        ),
        "method_score_sha256": (
            EXPECTED_METHOD_SCORE_SHA256
        ),
        "labels_loaded_only_for_evaluation": True,
        "datasets": datasets,
        "pooled": pooled,
        "paired_bootstrap": bootstrap_result,
        "baseline_comparison": {
            "unique_majority_top1": (
                baseline["pooled"][
                    "unique_majority"
                ]["top1"]
            ),
            "raw_rm_top1": (
                baseline["pooled"][
                    "methods"
                ]["raw_rm"]["top1"]
            ),
            "weighted_tau4_top1": (
                baseline["pooled"][
                    "methods"
                ]["weighted_tau4"][
                    "top1"
                ]
            ),
            "oracle_pass_at_16": (
                baseline["pooled"][
                    "oracle_pass_at_16"
                ]
            ),
        },
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

    print(
        "===== Fresh Math 冻结 "
        "RM-support 评价 ====="
    )

    for name, result in [
        *datasets.items(),
        ("POOLED", pooled),
    ]:
        print()
        print(name)
        print(json.dumps({
            "questions": result["questions"],
            "oracle_pass_at_16": (
                result["oracle_pass_at_16"]
            ),
            "raw_rm": result["raw_rm"],
            "rm_support": (
                result["rm_support"]
            ),
        }, ensure_ascii=False, indent=2))

    print()
    print("===== 配对 Bootstrap =====")
    print(json.dumps(
        bootstrap_result,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("结果：", OUTPUT_PATH)
    print(
        "耗时秒：",
        output["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
