from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import comb
import hashlib
import json
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    ROOT / "data/processed/arc_challenge_v1/"
    "qwen2_7b_medium_pilot_questions.jsonl"
)
PRIMARY_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_medium_pilot_candidates.jsonl"
)
RECOVERED_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_medium_pilot_recovered_candidates.jsonl"
)
LABEL_PATHS = {
    "train": (
        ROOT / "data/external/arc_challenge_v1/"
        "train_labels.jsonl"
    ),
    "pilot": (
        ROOT / "data/external/arc_challenge_v1/"
        "pilot_labels.jsonl"
    ),
}
OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "arc_challenge_qwen2_7b_medium_pilot_"
    "label_evaluation_v1.json"
)

EXPECTED = {
    "questions_sha256": (
        "1594563ee904f8c6d31a9d1eca81a769"
        "7d2220c7be48a602183868bfaa0bfd92"
    ),
    "primary_sha256": (
        "eb613189dc6ade7cddea4e937cb8512c"
        "5bfa4a70b95435f67462d914bd6399d0"
    ),
    "recovered_sha256": (
        "4ec477bcaef3fc7146508ade766ce44a"
        "77bf549adbeaedbecf83dae39c256a91"
    ),
}

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
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
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


def pass_probability(correct, total, k):
    if k > total:
        return None
    if correct <= 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return float(
        1.0
        - comb(total - correct, k)
        / comb(total, k)
    )


def prediction(row):
    value = row.get(
        "parsed_choice_index_generation_audit"
    )
    if value is None:
        return None
    return int(value)


def build_gold(questions):
    question_by_uid = {
        str(row["question_uid"]): row
        for row in questions
    }
    target_uids = set(question_by_uid)
    gold = {}

    for split, path in LABEL_PATHS.items():
        for row in read_jsonl(path):
            uid = str(row["question_uid"])
            if uid not in target_uids:
                continue

            if uid in gold:
                raise RuntimeError(
                    f"重复标签：{uid}"
                )

            expected_split = str(
                question_by_uid[uid][
                    "logical_split"
                ]
            )
            if expected_split != split:
                raise RuntimeError(
                    f"标签划分错误：{uid}"
                )

            answer_index = int(
                row["answer_index"]
            )
            answer_label = str(
                row["answer_label"]
            )
            labels = [
                str(value)
                for value in question_by_uid[
                    uid
                ]["choice_labels"]
            ]

            if not (
                0 <= answer_index < len(labels)
            ):
                raise RuntimeError(
                    f"答案索引越界：{uid}"
                )
            if labels[answer_index] != answer_label:
                raise RuntimeError(
                    f"答案标签和索引不一致：{uid}"
                )

            gold[uid] = answer_index

    if set(gold) != target_uids:
        raise RuntimeError(
            "Train/Pilot 标签覆盖不完整"
        )

    return question_by_uid, gold


def prepare_groups(rows, question_by_uid):
    groups = defaultdict(list)
    identities = set()

    for row in rows:
        uid = str(row["question_uid"])
        index = int(row["candidate_index"])
        key = (uid, index)

        if uid not in question_by_uid:
            raise RuntimeError(
                f"候选含未知问题：{uid}"
            )
        if key in identities:
            raise RuntimeError(
                f"候选身份重复：{key}"
            )

        identities.add(key)
        groups[uid].append(row)

    for uid, members in groups.items():
        members.sort(
            key=lambda row: int(
                row["candidate_index"]
            )
        )
        indices = [
            int(row["candidate_index"])
            for row in members
        ]
        if indices != list(range(16)):
            raise RuntimeError(
                f"{uid}: candidate_index 不完整"
            )

    if (
        len(groups) != 128
        or len(rows) != 2048
    ):
        raise RuntimeError(
            "候选数量不符合 128×16"
        )

    return dict(groups), identities


def evaluate(groups, gold, selected_uids):
    candidate_correct = []
    parsed_flags = []
    question_details = {}
    cluster_counts = []

    for uid in selected_uids:
        members = groups[uid]
        target = gold[uid]

        predictions = [
            prediction(row)
            for row in members
        ]
        parsed = [
            value is not None
            for value in predictions
        ]
        correct = [
            value == target
            for value in predictions
        ]

        parsed_count = int(sum(parsed))
        correct_count = int(sum(correct))
        total = len(members)

        parsed_flags.extend(parsed)
        candidate_correct.extend(correct)

        clusters = defaultdict(list)
        for row, value in zip(
            members,
            predictions,
        ):
            candidate_index = int(
                row["candidate_index"]
            )
            if value is None:
                cluster_key = (
                    f"unparsed:{candidate_index}"
                )
            else:
                cluster_key = f"choice:{value}"

            clusters[cluster_key].append(
                candidate_index
            )

        support = {
            key: len(indices)
            for key, indices in clusters.items()
        }
        top_support = max(support.values())
        tied = sorted(
            [
                key
                for key, count in support.items()
                if count == top_support
            ],
            key=lambda key: min(clusters[key]),
        )

        correct_key = f"choice:{target}"
        unique_majority_correct = (
            len(tied) == 1
            and tied[0] == correct_key
        )
        optimistic_tie_correct = (
            correct_key in tied
        )
        deterministic_choice = tied[0]
        deterministic_correct = (
            deterministic_choice == correct_key
        )

        parsed_choices = {
            value
            for value in predictions
            if value is not None
        }
        cluster_counts.append(
            len(parsed_choices)
        )

        question_details[uid] = {
            "parsed": parsed_count,
            "correct": correct_count,
            "total": total,
            "oracle": correct_count > 0,
            "all_wrong": correct_count == 0,
            "all_correct": correct_count == total,
            "mixed": 0 < correct_count < total,
            "top_support": top_support,
            "tie": len(tied) > 1,
            "unique_majority_correct": (
                unique_majority_correct
            ),
            "optimistic_tie_correct": (
                optimistic_tie_correct
            ),
            "deterministic_majority_correct": (
                deterministic_correct
            ),
            "deterministic_cluster": (
                deterministic_choice
            ),
            "parsed_choice_clusters": (
                len(parsed_choices)
            ),
        }

    details = list(question_details.values())
    parsed_total = int(sum(parsed_flags))
    correct_total = int(sum(candidate_correct))
    candidate_total = len(candidate_correct)
    question_total = len(details)

    result = {
        "questions": question_total,
        "candidates": candidate_total,
        "parsed_candidates": parsed_total,
        "unparsed_candidates": (
            candidate_total - parsed_total
        ),
        "candidate_parse_rate": float(
            parsed_total / candidate_total
        ),
        "candidate_accuracy": float(
            correct_total / candidate_total
        ),
        "parsed_candidate_accuracy": float(
            correct_total / max(parsed_total, 1)
        ),
        "correct_candidates": correct_total,
        "standard_pass_at_k": {},
        "oracle_pass_at_16": float(
            np.mean([
                item["oracle"]
                for item in details
            ])
        ),
        "question_composition": {
            "all_wrong": int(sum(
                item["all_wrong"]
                for item in details
            )),
            "mixed": int(sum(
                item["mixed"]
                for item in details
            )),
            "all_correct": int(sum(
                item["all_correct"]
                for item in details
            )),
            "mixed_question_coverage": float(
                np.mean([
                    item["mixed"]
                    for item in details
                ])
            ),
        },
        "majority_cluster": {
            "unique_majority_top1": float(
                np.mean([
                    item[
                        "unique_majority_correct"
                    ]
                    for item in details
                ])
            ),
            "deterministic_first_tie_top1": (
                float(np.mean([
                    item[
                        "deterministic_majority_correct"
                    ]
                    for item in details
                ]))
            ),
            "optimistic_tie_top1": float(
                np.mean([
                    item[
                        "optimistic_tie_correct"
                    ]
                    for item in details
                ])
            ),
            "tie_rate": float(
                np.mean([
                    item["tie"]
                    for item in details
                ])
            ),
            "tie_questions": int(sum(
                item["tie"]
                for item in details
            )),
        },
        "choice_diversity": {
            "min_clusters": int(
                min(cluster_counts)
            ),
            "median_clusters": float(
                np.median(cluster_counts)
            ),
            "mean_clusters": float(
                np.mean(cluster_counts)
            ),
            "max_clusters": int(
                max(cluster_counts)
            ),
            "multi_cluster_questions": int(
                sum(
                    value > 1
                    for value in cluster_counts
                )
            ),
            "multi_cluster_coverage": float(
                np.mean([
                    value > 1
                    for value in cluster_counts
                ])
            ),
        },
    }

    for k in K_VALUES:
        result["standard_pass_at_k"][
            f"pass_at_{k}"
        ] = float(np.mean([
            pass_probability(
                item["correct"],
                item["total"],
                k,
            )
            for item in details
        ]))

    return result, question_details


def compare_versions(
    primary_details,
    recovered_details,
):
    uids = sorted(primary_details)

    def transition(field):
        corrections = 0
        damages = 0
        unchanged_correct = 0
        unchanged_wrong = 0

        for uid in uids:
            before = bool(
                primary_details[uid][field]
            )
            after = bool(
                recovered_details[uid][field]
            )

            if not before and after:
                corrections += 1
            elif before and not after:
                damages += 1
            elif before:
                unchanged_correct += 1
            else:
                unchanged_wrong += 1

        return {
            "corrections": corrections,
            "damages": damages,
            "net_corrected": (
                corrections - damages
            ),
            "unchanged_correct": (
                unchanged_correct
            ),
            "unchanged_wrong": (
                unchanged_wrong
            ),
        }

    return {
        "correct_candidate_delta": int(sum(
            recovered_details[uid]["correct"]
            - primary_details[uid]["correct"]
            for uid in uids
        )),
        "parsed_candidate_delta": int(sum(
            recovered_details[uid]["parsed"]
            - primary_details[uid]["parsed"]
            for uid in uids
        )),
        "oracle_transition": transition(
            "oracle"
        ),
        "unique_majority_transition": (
            transition(
                "unique_majority_correct"
            )
        ),
        "deterministic_majority_transition": (
            transition(
                "deterministic_majority_correct"
            )
        ),
        "questions_with_correct_count_change": int(
            sum(
                primary_details[uid]["correct"]
                != recovered_details[uid][
                    "correct"
                ]
                for uid in uids
            )
        ),
        "questions_with_majority_cluster_change": int(
            sum(
                primary_details[uid][
                    "deterministic_cluster"
                ]
                != recovered_details[uid][
                    "deterministic_cluster"
                ]
                for uid in uids
            )
        ),
    }


def recovery_candidate_audit(
    primary_rows,
    recovered_rows,
    gold,
):
    primary_by_key = {
        (
            str(row["question_uid"]),
            int(row["candidate_index"]),
        ): row
        for row in primary_rows
    }

    successful = []
    failed = []

    for row in recovered_rows:
        if not row.get(
            "recovery_attempted",
            False,
        ):
            continue

        key = (
            str(row["question_uid"]),
            int(row["candidate_index"]),
        )
        before = primary_by_key[key]

        record = {
            "uid": key[0],
            "candidate_index": key[1],
            "split": str(
                row["logical_split"]
            ),
            "before_prediction": (
                prediction(before)
            ),
            "after_prediction": (
                prediction(row)
            ),
            "after_correct": (
                prediction(row)
                == gold[key[0]]
            ),
        }

        if row.get(
            "recovery_success",
            False,
        ):
            successful.append(record)
        else:
            failed.append(record)

    if len(successful) != 69:
        raise RuntimeError(
            "成功恢复数量不是 69"
        )
    if len(failed) != 8:
        raise RuntimeError(
            "失败恢复数量不是 8"
        )
    if any(
        item["before_prediction"] is not None
        for item in successful + failed
    ):
        raise RuntimeError(
            "恢复目标中存在 Primary 已解析候选"
        )

    return {
        "attempted": len(
            successful + failed
        ),
        "success": len(successful),
        "failed": len(failed),
        "successful_recovery_correct": int(
            sum(
                item["after_correct"]
                for item in successful
            )
        ),
        "successful_recovery_accuracy": float(
            np.mean([
                item["after_correct"]
                for item in successful
            ])
        ),
        "by_split": {
            split: {
                "attempted": int(sum(
                    item["split"] == split
                    for item
                    in successful + failed
                )),
                "success": int(sum(
                    item["split"] == split
                    for item in successful
                )),
                "correct": int(sum(
                    item["split"] == split
                    and item["after_correct"]
                    for item in successful
                )),
            }
            for split in ["train", "pilot"]
        },
    }


def main():
    print(
        "===== ARC Medium Pilot "
        "首次 Train/Pilot 标签评估 ====="
    )
    print("读取 Test 标签：False")
    print(
        "标签仅用于冻结候选后的评价：True"
    )

    actual_hashes = {
        "questions_sha256": sha256_file(
            QUESTIONS_PATH
        ),
        "primary_sha256": sha256_file(
            PRIMARY_PATH
        ),
        "recovered_sha256": sha256_file(
            RECOVERED_PATH
        ),
    }

    if actual_hashes != EXPECTED:
        raise RuntimeError(
            "冻结输入 SHA256 不匹配："
            f"{actual_hashes}"
        )

    questions = read_jsonl(
        QUESTIONS_PATH
    )
    primary_rows = read_jsonl(
        PRIMARY_PATH
    )
    recovered_rows = read_jsonl(
        RECOVERED_PATH
    )

    question_by_uid, gold = build_gold(
        questions
    )
    primary_groups, primary_identities = (
        prepare_groups(
            primary_rows,
            question_by_uid,
        )
    )
    recovered_groups, recovered_identities = (
        prepare_groups(
            recovered_rows,
            question_by_uid,
        )
    )

    if (
        primary_identities
        != recovered_identities
    ):
        raise RuntimeError(
            "Primary/Recovered 候选身份不同"
        )

    split_uids = {
        split: sorted([
            uid
            for uid, row
            in question_by_uid.items()
            if str(row["logical_split"])
            == split
        ])
        for split in ["train", "pilot"]
    }
    split_uids["pooled"] = sorted(
        question_by_uid
    )

    systems = {}
    details = {
        "primary": {},
        "recovered": {},
    }

    for system_name, groups in [
        ("primary", primary_groups),
        ("recovered", recovered_groups),
    ]:
        systems[system_name] = {}

        for split, uids in split_uids.items():
            metrics, question_details = (
                evaluate(
                    groups,
                    gold,
                    uids,
                )
            )
            systems[system_name][
                split
            ] = metrics
            details[system_name][
                split
            ] = question_details

    comparison = {
        split: compare_versions(
            details["primary"][split],
            details["recovered"][split],
        )
        for split in split_uids
    }

    recovery_audit = (
        recovery_candidate_audit(
            primary_rows,
            recovered_rows,
            gold,
        )
    )

    pilot = systems[
        "recovered"
    ]["pilot"]
    pilot_oracle_gap = (
        pilot["oracle_pass_at_16"]
        - pilot["candidate_accuracy"]
    )

    screening_checks = {
        "parse_rate_at_least_0p99": (
            pilot["candidate_parse_rate"]
            >= 0.99
        ),
        "candidate_accuracy_not_saturated": (
            0.10
            <= pilot["candidate_accuracy"]
            <= 0.90
        ),
        "oracle_gap_at_least_0p05": (
            pilot_oracle_gap >= 0.05
        ),
        "mixed_coverage_at_least_0p25": (
            pilot[
                "question_composition"
            ][
                "mixed_question_coverage"
            ] >= 0.25
        ),
        "multi_cluster_coverage_at_least_0p50": (
            pilot[
                "choice_diversity"
            ][
                "multi_cluster_coverage"
            ] >= 0.50
        ),
    }

    proceed = all(
        screening_checks.values()
    )

    result = {
        "version": (
            "arc_challenge_qwen2_7b_"
            "medium_pilot_label_evaluation_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_status": (
            "post_freeze_train_pilot_"
            "label_reveal"
        ),
        "labels_loaded": True,
        "label_splits_loaded": [
            "train",
            "pilot",
        ],
        "test_split_used": False,
        "sealed_test_labels_loaded": False,
        "inputs": {
            "questions": str(
                QUESTIONS_PATH.relative_to(
                    ROOT
                )
            ),
            "primary_candidates": str(
                PRIMARY_PATH.relative_to(
                    ROOT
                )
            ),
            "recovered_candidates": str(
                RECOVERED_PATH.relative_to(
                    ROOT
                )
            ),
            **actual_hashes,
            "label_sha256": {
                split: sha256_file(path)
                for split, path
                in LABEL_PATHS.items()
            },
        },
        "systems": systems,
        "recovery_candidate_audit": (
            recovery_audit
        ),
        "primary_vs_recovered": (
            comparison
        ),
        "pilot_screening": {
            "oracle_minus_candidate_accuracy": (
                pilot_oracle_gap
            ),
            "checks": screening_checks,
            "decision": (
                "proceed_to_full_arc_protocol"
                if proceed
                else
                "do_not_scale_before_review"
            ),
            "note": (
                "Descriptive screening decision; "
                "not a preregistered hypothesis test."
            ),
        },
    }

    atomic_json(OUTPUT_PATH, result)

    for split in [
        "train",
        "pilot",
        "pooled",
    ]:
        print()
        print("=" * 76)
        print(split.upper())

        for system in [
            "primary",
            "recovered",
        ]:
            print()
            print(system)
            print(json.dumps(
                systems[system][split],
                ensure_ascii=False,
                indent=2,
            ))

        print()
        print("Primary -> Recovered")
        print(json.dumps(
            comparison[split],
            ensure_ascii=False,
            indent=2,
        ))

    print()
    print("=" * 76)
    print("===== 恢复候选正确性 =====")
    print(json.dumps(
        recovery_audit,
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("===== 是否扩展到全量 ARC =====")
    print(json.dumps(
        result["pilot_screening"],
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("结果：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
