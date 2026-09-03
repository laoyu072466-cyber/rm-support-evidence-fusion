from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import explore_answer_cluster_evidence_smoke as cluster
import configure_arc_multi_reward_train_pilot_v1 as base
import configure_arc_multi_reward_train_pilot_v2 as v2


CONFIG_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_train_pilot_config_v2.json"
)
CANDIDATE_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_full_k16_recovered_candidates.jsonl"
)
TEST_LABEL_PATH = (
    ROOT / "data/external/arc_challenge_v1/"
    "sealed_test_labels.jsonl"
)
OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_frozen_test_v1.json"
)

CONFIG_TAG = (
    "arc-multi-reward-train-pilot-config-v2"
)
PROTOCOL_TAG = (
    "arc-multi-reward-frozen-test-protocol-v1"
)

EXPECTED_CANDIDATE_SHA256 = (
    "eb72914f82678c4738fab24ea055a62b"
    "00f8c207372d3f3ba85d785591267191"
)

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260910
K_VALUES = [1, 4, 8, 16]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def atomic_json(path, value):
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            base.jsonable(value),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_tagged_file(tag, path):
    relative = str(path.relative_to(ROOT))
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{tag}:{relative}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    require(
        completed.returncode == 0,
        f"无法从标签读取：{tag}:{relative}",
    )
    require(
        completed.stdout == path.read_bytes(),
        f"当前文件不同于冻结标签：{relative}",
    )


def preflight(require_protocol_tag):
    print(
        "===== ARC 冻结 Test 评价预检 ====="
    )
    print("配置训练：False")
    print("超参数搜索：False")
    print("Test 标签读取：False")

    require(
        not OUTPUT_PATH.exists(),
        "Test 结果已经存在，拒绝覆盖或重跑",
    )
    require(
        CONFIG_PATH.exists(),
        "缺少冻结 v2 配置",
    )
    require(
        CANDIDATE_PATH.exists(),
        "缺少恢复候选",
    )
    require(
        TEST_LABEL_PATH.exists(),
        "缺少密封 Test 标签文件",
    )
    require(
        sha256_file(CANDIDATE_PATH)
        == EXPECTED_CANDIDATE_SHA256,
        "候选 SHA256 不一致",
    )

    verify_tagged_file(
        CONFIG_TAG,
        CONFIG_PATH,
    )

    if require_protocol_tag:
        verify_tagged_file(
            PROTOCOL_TAG,
            Path(__file__).resolve(),
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )
    require(
        config.get("version")
        == (
            "arc_multi_reward_"
            "train_pilot_config_v2"
        ),
        "配置版本错误",
    )
    require(
        config["protocol"][
            "test_labels_loaded"
        ] is False,
        "配置阶段读取过 Test 标签",
    )
    require(
        config["protocol"][
            "test_candidates_used_for_training"
        ] is False,
        "Test 候选参与过训练",
    )
    require(
        config["protocol"][
            "test_candidates_used_for_selection"
        ] is False,
        "Test 候选参与过选择",
    )

    rows = read_jsonl(CANDIDATE_PATH)
    require(
        len(rows) == 41440,
        "候选数量错误",
    )

    test_rows = [
        row
        for row in rows
        if row["logical_split"] == "test"
    ]
    test_uids = {
        str(row["question_uid"])
        for row in test_rows
    }

    require(
        len(test_rows) == 18752,
        "Test 候选数量错误",
    )
    require(
        len(test_uids) == 1172,
        "Test 题目数量错误",
    )
    require(
        all(
            sum(
                str(row["question_uid"]) == uid
                for row in test_rows
            ) == 16
            for uid in test_uids
        ),
        "Test 每题候选数不为 16",
    )

    for model_key, record in (
        config["models"].items()
    ):
        score_path = (
            ROOT / record["score_file"]
        )
        require(
            score_path.exists(),
            f"{model_key}: 缺少评分数组",
        )
        require(
            sha256_file(score_path)
            == record["score_sha256"],
            f"{model_key}: 分数 SHA256 不一致",
        )
        scores = np.load(
            score_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        require(
            scores.shape == (41440,),
            f"{model_key}: 分数形状错误",
        )
        require(
            bool(np.all(np.isfinite(
                scores
            ))),
            f"{model_key}: 存在非有限分数",
        )
        require(
            len(record["fitted_models"]) == 3,
            f"{model_key}: 冻结模型数量错误",
        )

    print("冻结配置标签：", CONFIG_TAG)
    print("配置 SHA256：", sha256_file(CONFIG_PATH))
    print("候选 SHA256：", sha256_file(CANDIDATE_PATH))
    print("Test 问题：", len(test_uids))
    print("Test 候选：", len(test_rows))
    print(
        "ARC_FROZEN_TEST_PREFLIGHT_READY"
    )

    return config, rows


def load_test_labels(expected_uids):
    rows = read_jsonl(TEST_LABEL_PATH)
    labels = {}

    for row in rows:
        uid = str(row["question_uid"])
        require(
            uid not in labels,
            f"Test 标签 UID 重复：{uid}",
        )
        labels[uid] = int(
            row["answer_index"]
        )

    require(
        len(labels) == 1172,
        "Test 标签数量错误",
    )
    require(
        set(labels) == expected_uids,
        "Test 标签与候选 UID 不匹配",
    )

    return labels


def restore_models(records):
    return [
        {
            "weights": np.asarray(
                record["weights"],
                dtype=np.float32,
            ),
            "mean": np.asarray(
                record["mean"],
                dtype=np.float32,
            ),
            "std": np.asarray(
                record["std"],
                dtype=np.float32,
            ),
            "regularization": float(
                record["regularization"]
            ),
            "seed": int(record["seed"]),
        }
        for record in records
    ]


def learned_scores(
    dataset,
    fitted_models,
    beta,
    threshold,
):
    predictions = np.empty(
        len(dataset["labels"]),
        dtype=np.float32,
    )

    for question in dataset["questions"]:
        learned = cluster.model_cluster_scores(
            question,
            base.FEATURE_INDICES,
            fitted_models,
        )
        learned = cluster.ordinary_z(
            learned
        )

        raw_cluster = np.asarray([
            item["rm_max"]
            for item in question["clusters"]
        ], dtype=np.float32)
        raw_cluster = cluster.ordinary_z(
            raw_cluster
        )

        hybrid = (
            raw_cluster
            + float(beta) * learned
        )

        local_scores = (
            cluster.candidate_scores_from_clusters(
                question,
                hybrid,
                float(threshold),
            )
        )
        predictions[
            question["indices"]
        ] = local_scores

    require(
        bool(np.all(np.isfinite(
            predictions
        ))),
        "融合分数含非有限值",
    )
    return predictions


def majority_scores(dataset):
    predictions = np.empty(
        len(dataset["labels"]),
        dtype=np.float32,
    )

    for question in dataset["questions"]:
        supports = np.asarray([
            len(item["members"])
            for item in question["clusters"]
        ], dtype=np.float32)
        cluster_rm = np.asarray([
            item["rm_max"]
            for item in question["clusters"]
        ], dtype=np.float32)

        cluster_values = (
            supports
            + 1e-4
            * cluster.ordinary_z(cluster_rm)
        )
        candidate_rm = cluster.ordinary_z(
            question["rm"]
        )

        local = np.empty(
            len(question["indices"]),
            dtype=np.float32,
        )

        for cluster_index, item in enumerate(
            question["clusters"]
        ):
            for member in item["members"]:
                local[int(member)] = (
                    cluster_values[
                        cluster_index
                    ]
                    + 1e-6
                    * candidate_rm[int(member)]
                )

        predictions[
            question["indices"]
        ] = local

    return predictions


def method_metrics(
    dataset,
    method_scores,
    raw_scores,
):
    question_ids = [
        str(row["question_uid"])
        for row in dataset["rows"]
    ]
    result = v2.arc_ranking_metrics(
        dataset["labels"],
        method_scores,
        question_ids,
        raw_scores,
    )
    return result


def pass_probability(labels, k):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )
    n = len(labels)
    correct = int(np.sum(labels))

    if k > n:
        return math.nan
    if correct == 0:
        return 0.0
    if n - correct < k:
        return 1.0

    return float(
        1.0
        - math.comb(n - correct, k)
        / math.comb(n, k)
    )


def selector_probability(
    labels,
    scores,
    k,
):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )
    n = len(labels)

    if k > n:
        return math.nan

    order = np.argsort(
        -scores,
        kind="stable",
    )
    ranked_labels = labels[order]
    denominator = math.comb(n, k)
    probability = 0.0

    for rank, label in enumerate(
        ranked_labels
    ):
        remaining = n - rank - 1
        if remaining < k - 1:
            break

        chosen_as_best = (
            math.comb(
                remaining,
                k - 1,
            )
            / denominator
        )
        probability += (
            int(label) * chosen_as_best
        )

    return float(probability)


def budget_metrics(dataset, scores):
    result = {}

    for k in K_VALUES:
        pass_values = []
        selector_values = []

        for question in dataset["questions"]:
            indices = question["indices"]
            labels = dataset["labels"][
                indices
            ]
            local_scores = scores[indices]

            pass_values.append(
                pass_probability(labels, k)
            )
            selector_values.append(
                selector_probability(
                    labels,
                    local_scores,
                    k,
                )
            )

        result[f"k{k}"] = {
            "eligible_questions": len(
                pass_values
            ),
            "pass_at_k": float(
                np.mean(pass_values)
            ),
            "best_at_k": float(
                np.mean(selector_values)
            ),
        }

    return result


def pair_value(labels, scores):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )
    positive = scores[labels == 1]
    negative = scores[labels == 0]

    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return math.nan

    return float(np.mean(
        positive[:, None]
        > negative[None, :]
    ))


def question_records(
    dataset,
    raw_scores,
    method_scores,
):
    records = []

    for question in dataset["questions"]:
        indices = question["indices"]
        labels = dataset["labels"][indices]
        raw = raw_scores[indices]
        method = method_scores[indices]

        raw_choice = int(np.argmax(raw))
        method_choice = int(
            np.argmax(method)
        )
        raw_correct = int(
            labels[raw_choice] == 1
        )
        method_correct = int(
            labels[method_choice] == 1
        )

        record = {
            "top1_delta": (
                method_correct - raw_correct
            ),
            "raw_correct": raw_correct,
            "raw_wrong": int(
                raw_correct == 0
            ),
            "damage": int(
                raw_correct == 1
                and method_correct == 0
            ),
            "correction": int(
                raw_correct == 0
                and method_correct == 1
            ),
            "pair_delta": (
                pair_value(labels, method)
                - pair_value(labels, raw)
            ),
        }

        for k in [4, 8, 16]:
            record[f"best_delta_{k}"] = (
                selector_probability(
                    labels,
                    method,
                    k,
                )
                - selector_probability(
                    labels,
                    raw,
                    k,
                )
            )

        records.append(record)

    return {
        key: np.asarray([
            record[key]
            for record in records
        ], dtype=np.float64)
        for key in records[0]
    }


def distribution_summary(
    values,
    point,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "point": float(point),
        "ci95": [
            float(np.percentile(
                values,
                2.5,
            )),
            float(np.percentile(
                values,
                97.5,
            )),
        ],
        "probability_positive": float(
            np.mean(values > 0)
        ),
    }


def paired_bootstrap(arrays, seed):
    rng = np.random.default_rng(seed)
    count = len(arrays["top1_delta"])
    distributions = {
        "top1_delta": [],
        "pair_delta": [],
        "damage_rate": [],
        "correction_rate": [],
        "best_at_4_delta": [],
        "best_at_8_delta": [],
        "best_at_16_delta": [],
    }

    remaining = BOOTSTRAP_SAMPLES

    while remaining > 0:
        batch = min(250, remaining)
        indices = rng.integers(
            0,
            count,
            size=(batch, count),
        )

        distributions[
            "top1_delta"
        ].append(np.mean(
            arrays["top1_delta"][indices],
            axis=1,
        ))
        distributions[
            "pair_delta"
        ].append(np.nanmean(
            arrays["pair_delta"][indices],
            axis=1,
        ))

        raw_correct = np.sum(
            arrays["raw_correct"][indices],
            axis=1,
        )
        raw_wrong = np.sum(
            arrays["raw_wrong"][indices],
            axis=1,
        )

        distributions[
            "damage_rate"
        ].append(
            np.sum(
                arrays["damage"][indices],
                axis=1,
            )
            / np.maximum(raw_correct, 1)
        )
        distributions[
            "correction_rate"
        ].append(
            np.sum(
                arrays["correction"][indices],
                axis=1,
            )
            / np.maximum(raw_wrong, 1)
        )

        for k in [4, 8, 16]:
            distributions[
                f"best_at_{k}_delta"
            ].append(np.mean(
                arrays[
                    f"best_delta_{k}"
                ][indices],
                axis=1,
            ))

        remaining -= batch

    distributions = {
        key: np.concatenate(value)
        for key, value
        in distributions.items()
    }

    points = {
        "top1_delta": float(np.mean(
            arrays["top1_delta"]
        )),
        "pair_delta": float(np.nanmean(
            arrays["pair_delta"]
        )),
        "damage_rate": float(
            np.sum(arrays["damage"])
            / max(
                np.sum(
                    arrays["raw_correct"]
                ),
                1,
            )
        ),
        "correction_rate": float(
            np.sum(arrays["correction"])
            / max(
                np.sum(
                    arrays["raw_wrong"]
                ),
                1,
            )
        ),
        "best_at_4_delta": float(
            np.mean(
                arrays["best_delta_4"]
            )
        ),
        "best_at_8_delta": float(
            np.mean(
                arrays["best_delta_8"]
            )
        ),
        "best_at_16_delta": float(
            np.mean(
                arrays["best_delta_16"]
            )
        ),
    }

    return {
        key: distribution_summary(
            distributions[key],
            points[key],
        )
        for key in distributions
    }


def add_deltas(method, raw):
    result = dict(method)
    result["top1_delta"] = (
        method["top1"] - raw["top1"]
    )
    result["pair_delta"] = (
        method["pair_macro_strict"]
        - raw["pair_macro_strict"]
    )

    result["budget"] = {
        key: {
            **value,
            "raw_best_at_k": (
                raw["budget"][key][
                    "best_at_k"
                ]
            ),
            "best_at_k_delta": (
                value["best_at_k"]
                - raw["budget"][key][
                    "best_at_k"
                ]
            ),
        }
        for key, value
        in method["budget"].items()
    }
    return result


def majority_failure_audit(
    dataset,
    raw_scores,
    majority,
):
    corrections = 0
    damages = 0
    majority_wrong_with_correct_pool = 0

    for question in dataset["questions"]:
        indices = question["indices"]
        labels = dataset["labels"][indices]
        raw_choice = int(np.argmax(
            raw_scores[indices]
        ))
        majority_choice = int(np.argmax(
            majority[indices]
        ))

        raw_correct = int(
            labels[raw_choice] == 1
        )
        majority_correct = int(
            labels[majority_choice] == 1
        )

        corrections += int(
            raw_correct == 0
            and majority_correct == 1
        )
        damages += int(
            raw_correct == 1
            and majority_correct == 0
        )
        majority_wrong_with_correct_pool += int(
            majority_correct == 0
            and np.any(labels == 1)
        )

    return {
        "corrections_of_raw": corrections,
        "damages_of_raw": damages,
        "net_corrected": (
            corrections - damages
        ),
        "wrong_with_correct_candidate_pool": (
            majority_wrong_with_correct_pool
        ),
    }


def gate_audit(
    dataset,
    raw_scores,
    ungated,
    gated,
):
    result = Counter()

    for question in dataset["questions"]:
        indices = question["indices"]
        labels = dataset["labels"][indices]

        raw_choice = int(np.argmax(
            raw_scores[indices]
        ))
        proposal = int(np.argmax(
            ungated[indices]
        ))
        gate_choice = int(np.argmax(
            gated[indices]
        ))

        raw_correct = int(
            labels[raw_choice] == 1
        )
        proposal_correct = int(
            labels[proposal] == 1
        )
        gate_correct = int(
            labels[gate_choice] == 1
        )

        if proposal == raw_choice:
            continue

        result["proposal_switches"] += 1

        if gate_choice == proposal:
            result["accepted_switches"] += 1
            if (
                raw_correct == 0
                and proposal_correct == 1
            ):
                result[
                    "accepted_corrections"
                ] += 1
            elif (
                raw_correct == 1
                and proposal_correct == 0
            ):
                result[
                    "accepted_damages"
                ] += 1
            else:
                result[
                    "accepted_neutral"
                ] += 1
        elif gate_choice == raw_choice:
            result["rejected_switches"] += 1
            if (
                raw_correct == 1
                and proposal_correct == 0
            ):
                result[
                    "prevented_damages"
                ] += 1
            elif (
                raw_correct == 0
                and proposal_correct == 1
            ):
                result[
                    "blocked_corrections"
                ] += 1
            else:
                result[
                    "rejected_neutral"
                ] += 1
        else:
            raise RuntimeError(
                "Gate 选择既不是 Raw 也不是 Proposal"
            )

        result[
            "net_questions_saved_by_gate"
        ] += gate_correct - proposal_correct

    result["proposal_switch_rate"] = (
        result["proposal_switches"]
        / len(dataset["questions"])
    )
    return dict(result)


def test_dataset_summary(dataset):
    parsed = sum(
        base.parsed_choice_index(row)
        is not None
        for row in dataset["rows"]
    )

    composition = Counter()
    for question in dataset["questions"]:
        labels = question["labels"]
        if np.all(labels == 0):
            composition["all_wrong"] += 1
        elif np.all(labels == 1):
            composition["all_correct"] += 1
        else:
            composition["mixed"] += 1

    return {
        "questions": len(
            dataset["questions"]
        ),
        "candidates": len(
            dataset["rows"]
        ),
        "parsed_candidates": parsed,
        "parse_rate": (
            parsed / len(dataset["rows"])
        ),
        "candidate_accuracy": float(
            np.mean(dataset["labels"])
        ),
        "question_composition": dict(
            composition
        ),
        "pair_eligible_questions": int(
            composition["mixed"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    args = parser.parse_args()

    config, all_rows = preflight(
        require_protocol_tag=(
            not args.preflight_only
        )
    )

    if args.preflight_only:
        print(
            "Preflight-only 完成；"
            "没有读取 Test 标签。"
        )
        return

    started = time.time()

    print()
    print(
        "===== 首次读取 ARC 密封 Test 标签 ====="
    )
    print("配置训练：False")
    print("超参数搜索：False")
    print(
        "使用配置标签：",
        CONFIG_TAG,
    )

    expected_test_uids = {
        str(row["question_uid"])
        for row in all_rows
        if row["logical_split"] == "test"
    }
    test_gold = load_test_labels(
        expected_test_uids
    )

    print("Test 标签关联：1172/1172")
    print("标签值打印：False")

    base.EXPECTED_QUESTIONS["test"] = 1172
    cluster.ranking_metrics = (
        v2.arc_ranking_metrics
    )

    result = {
        "version": (
            "arc_multi_reward_"
            "frozen_test_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "task": "ARC-Challenge",
        "protocol": {
            "configuration_tag": CONFIG_TAG,
            "evaluation_protocol_tag": (
                PROTOCOL_TAG
            ),
            "configuration_sha256": (
                sha256_file(CONFIG_PATH)
            ),
            "candidate_sha256": (
                sha256_file(CANDIDATE_PATH)
            ),
            "train_labels_used_for_test": False,
            "pilot_labels_used_for_test": False,
            "test_labels_loaded": True,
            "test_labels_used_only_for_evaluation": (
                True
            ),
            "retraining_after_test_access": False,
            "retuning_after_test_access": False,
            "test_label_values_printed": False,
            "bootstrap_samples": (
                BOOTSTRAP_SAMPLES
            ),
            "bootstrap_seed": (
                BOOTSTRAP_SEED
            ),
        },
        "test_labels": {
            "path": str(
                TEST_LABEL_PATH.relative_to(
                    ROOT
                )
            ),
            "sha256": sha256_file(
                TEST_LABEL_PATH
            ),
            "questions": len(test_gold),
            "values_printed": False,
        },
        "models": {},
    }

    raw_strength = []
    fusion_gain = []

    for model_number, (
        model_key,
        frozen,
    ) in enumerate(
        config["models"].items()
    ):
        print()
        print("=" * 76)
        print(frozen["model_name"])

        all_scores = np.load(
            ROOT / frozen["score_file"],
            mmap_mode="r",
            allow_pickle=False,
        )

        dataset, _ = base.build_dataset(
            f"{model_key}_FROZEN_TEST",
            "test",
            all_rows,
            all_scores,
            test_gold,
        )

        fitted = restore_models(
            frozen["fitted_models"]
        )
        selected = frozen["selected"]

        raw_scores = np.asarray(
            dataset["rm_scores"],
            dtype=np.float32,
        )
        majority = majority_scores(dataset)
        ungated = learned_scores(
            dataset,
            fitted,
            selected["beta"],
            0.0,
        )
        gated = learned_scores(
            dataset,
            fitted,
            selected["beta"],
            selected["threshold"],
        )

        raw = method_metrics(
            dataset,
            raw_scores,
            raw_scores,
        )
        raw["budget"] = budget_metrics(
            dataset,
            raw_scores,
        )

        majority_result = method_metrics(
            dataset,
            majority,
            raw_scores,
        )
        majority_result["budget"] = (
            budget_metrics(
                dataset,
                majority,
            )
        )
        majority_result = add_deltas(
            majority_result,
            raw,
        )

        ungated_result = method_metrics(
            dataset,
            ungated,
            raw_scores,
        )
        ungated_result["budget"] = (
            budget_metrics(
                dataset,
                ungated,
            )
        )
        ungated_result = add_deltas(
            ungated_result,
            raw,
        )

        gated_result = method_metrics(
            dataset,
            gated,
            raw_scores,
        )
        gated_result["budget"] = (
            budget_metrics(
                dataset,
                gated,
            )
        )
        gated_result = add_deltas(
            gated_result,
            raw,
        )

        records = question_records(
            dataset,
            raw_scores,
            gated,
        )
        bootstrap = paired_bootstrap(
            records,
            BOOTSTRAP_SEED + model_number,
        )

        model_result = {
            "model_name": frozen[
                "model_name"
            ],
            "frozen_configuration": {
                key: selected[key]
                for key in [
                    "regularization",
                    "beta",
                    "threshold",
                ]
            },
            "dataset": test_dataset_summary(
                dataset
            ),
            "methods": {
                "raw_rm": raw,
                "majority_rm_tiebreak": (
                    majority_result
                ),
                "ungated_hybrid": (
                    ungated_result
                ),
                "frozen_gate": gated_result,
            },
            "paired_bootstrap_gate_vs_raw": (
                bootstrap
            ),
            "majority_failure": (
                majority_failure_audit(
                    dataset,
                    raw_scores,
                    majority,
                )
            ),
            "gate_mechanism": gate_audit(
                dataset,
                raw_scores,
                ungated,
                gated,
            ),
        }
        result["models"][model_key] = (
            model_result
        )

        raw_strength.append(raw["top1"])
        fusion_gain.append(
            gated_result["top1_delta"]
        )

        print(
            f"Raw={raw['top1']:.6f} | "
            f"Majority="
            f"{majority_result['top1']:.6f} | "
            f"Ungated="
            f"{ungated_result['top1']:.6f} | "
            f"Gate="
            f"{gated_result['top1']:.6f}"
        )
        print(
            f"ΔTop1="
            f"{gated_result['top1_delta']:+.6f} | "
            f"ΔPair="
            f"{gated_result['pair_delta']:+.6f} | "
            f"damage="
            f"{gated_result['damage_rate']:.6f} | "
            f"correction="
            f"{gated_result['correction_rate']:.6f}"
        )
        print(
            "95% CI ΔTop1：",
            bootstrap["top1_delta"]["ci95"],
        )

    raw_array = np.asarray(
        raw_strength,
        dtype=np.float64,
    )
    gain_array = np.asarray(
        fusion_gain,
        dtype=np.float64,
    )

    pearson = pearsonr(
        raw_array,
        gain_array,
    )
    spearman = spearmanr(
        raw_array,
        gain_array,
    )

    ordered = sorted(
        zip(
            raw_array.tolist(),
            gain_array.tolist(),
        ),
        key=lambda item: item[0],
    )

    result["reward_strength_gain_trend"] = {
        "models": len(raw_array),
        "strength_proxy": "raw_test_top1",
        "gain": "frozen_gate_top1_delta",
        "pearson": float(
            pearson.statistic
        ),
        "spearman": float(
            spearman.statistic
        ),
        "strict_inverse_monotonic": all(
            ordered[index][1]
            > ordered[index + 1][1]
            for index in range(
                len(ordered) - 1
            )
        ),
        "ordered_weak_to_strong": [
            {
                "raw_top1": raw,
                "top1_delta": gain,
            }
            for raw, gain in ordered
        ],
        "interpretation_scope": (
            "descriptive three-model ARC test; "
            "not an independent significance test"
        ),
    }

    model_results = list(
        result["models"].values()
    )
    result["macro_across_reward_models"] = {
        "raw_top1": float(np.mean([
            item["methods"]["raw_rm"][
                "top1"
            ]
            for item in model_results
        ])),
        "majority_top1": float(np.mean([
            item["methods"][
                "majority_rm_tiebreak"
            ]["top1"]
            for item in model_results
        ])),
        "ungated_top1": float(np.mean([
            item["methods"][
                "ungated_hybrid"
            ]["top1"]
            for item in model_results
        ])),
        "frozen_gate_top1": float(np.mean([
            item["methods"][
                "frozen_gate"
            ]["top1"]
            for item in model_results
        ])),
        "frozen_gate_top1_delta": float(
            np.mean([
                item["methods"][
                    "frozen_gate"
                ]["top1_delta"]
                for item in model_results
            ])
        ),
        "frozen_gate_pair_delta": float(
            np.mean([
                item["methods"][
                    "frozen_gate"
                ]["pair_delta"]
                for item in model_results
            ])
        ),
    }

    result["elapsed_seconds"] = (
        time.time() - started
    )
    result["decision"] = (
        "freeze_one_shot_arc_test_results"
    )

    atomic_json(OUTPUT_PATH, result)

    print()
    print("=" * 76)
    print("===== ARC Test 宏平均 =====")
    print(json.dumps(
        result[
            "macro_across_reward_models"
        ],
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("===== 奖励强度—融合增益 =====")
    print(json.dumps(
        result[
            "reward_strength_gain_trend"
        ],
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print(
        "ARC_FROZEN_TEST_EVALUATION_COMPLETE"
    )
    print("结果：", OUTPUT_PATH)
    print(
        "耗时秒：",
        round(result["elapsed_seconds"], 3),
    )


if __name__ == "__main__":
    main()
