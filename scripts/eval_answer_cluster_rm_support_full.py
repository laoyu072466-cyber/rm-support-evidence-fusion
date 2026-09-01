from pathlib import Path
from collections import defaultdict
import json
import math
import sys
import time

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import explore_answer_cluster_evidence_smoke as cluster


DATA_ROOT = ROOT / "data/processed/prototype_v2"
RM_ROOT = (
    ROOT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
OUTPUT = (
    ROOT
    / "data/manifests/answer_cluster_rm_support_full_v1.json"
)

DOMAINS = {
    "GSM8K": {
        "family": "gsm",
        "train": (
            "gsm_train.jsonl",
            "gsm_train",
        ),
        "pilot": (
            "gsm_pilot_validation.jsonl",
            "gsm_pilot",
        ),
        "tests": {
            "GSM8K_ID": (
                "gsm_id_test_mixed.jsonl",
                "gsm_id_test",
            ),
            "SVAMP_OOD": (
                "svamp_ood_mixed.jsonl",
                "svamp_ood",
            ),
        },
    },
    "MATH": {
        "family": "math",
        "train": (
            "math_train.jsonl",
            "math_train",
        ),
        "pilot": (
            "math_pilot_validation.jsonl",
            "math_pilot",
        ),
        "tests": {
            "MATH_ID": (
                "math_id_test_mixed.jsonl",
                "math_id_test",
            ),
        },
    },
}

FEATURE_INDICES = list(range(8))


def load_full_dataset(
    display_name,
    family,
    filename,
    rm_prefix,
):
    rows = cluster.read_jsonl(
        DATA_ROOT / filename
    )
    rm_scores = np.asarray(
        np.load(
            RM_ROOT / f"{rm_prefix}.scores_f32.npy"
        ),
        dtype=np.float32,
    )

    if len(rows) != len(rm_scores):
        raise RuntimeError(
            f"{display_name}: 行数与 RM 分数不一致"
        )

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int8,
    )

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["question_uid"])].append(
            index
        )

    candidate_count = len(rows)

    # rm_support 只使用前八个特征。
    # 以下占位不会参与模型训练。
    dataset = {
        "name": display_name,
        "family": family,
        "rows": rows,
        "metadata": rows,
        "labels": labels,
        "rm_scores": rm_scores,
        "hidden": np.zeros(
            (candidate_count, 2),
            dtype=np.float32,
        ),
        "nll": np.zeros(
            (candidate_count, 5),
            dtype=np.float32,
        ),
        "prompt_agreement": np.ones(
            candidate_count,
            dtype=np.float32,
        ),
        "groups": dict(groups),
    }

    cluster.build_questions(dataset)

    untrainable_questions = sum(
        not question["cluster_trainable"]
        for question in dataset["questions"]
    )
    dataset["untrainable_questions"] = (
        untrainable_questions
    )

    print(
        f"{display_name}: "
        f"问题={len(dataset['questions'])}, "
        f"候选={candidate_count}, "
        f"混合答案簇={dataset['mixed_clusters']}, "
        f"不可构造正负簇问题="
        f"{untrainable_questions}",
        flush=True,
    )

    return dataset


def exact_logsumexp(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    maximum = float(np.max(values))
    return float(
        maximum
        + math.log(
            float(
                np.exp(values - maximum).sum()
            )
        )
    )


def evaluate_rule(dataset, rule):
    labels = []
    method_scores = []
    raw_scores = []
    question_ids = []

    for question in dataset["questions"]:
        if rule == "majority":
            support = np.asarray([
                item["features"][2]
                for item in question["clusters"]
            ], dtype=np.float32)

            rm_max = np.asarray([
                item["rm_max"]
                for item in question["clusters"]
            ], dtype=np.float32)

            cluster_scores = (
                cluster.ordinary_z(support)
                + 1e-4 * cluster.ordinary_z(rm_max)
            )

        elif rule == "weighted_tau4":
            cluster_scores = np.asarray([
                exact_logsumexp(
                    question["rm"][item["members"]]
                    / 4.0
                )
                for item in question["clusters"]
            ], dtype=np.float32)

        else:
            raise ValueError(rule)

        candidate_scores = (
            cluster.candidate_scores_from_clusters(
                question,
                cluster_scores,
                threshold=0.0,
            )
        )

        labels.extend(question["labels"].tolist())
        method_scores.extend(candidate_scores.tolist())
        raw_scores.extend(question["rm"].tolist())
        question_ids.extend(
            [question["uid"]]
            * len(question["labels"])
        )

    return cluster.ranking_metrics(
        np.asarray(labels, dtype=np.int8),
        np.asarray(
            method_scores,
            dtype=np.float32,
        ),
        question_ids,
        np.asarray(raw_scores, dtype=np.float32),
    )


def with_delta(metrics, raw):
    result = dict(metrics)
    result["raw_top1"] = raw["top1"]
    result["raw_pair_macro_strict"] = raw[
        "pair_macro_strict"
    ]
    result["top1_delta"] = (
        metrics["top1"] - raw["top1"]
    )
    result["pair_delta"] = (
        metrics["pair_macro_strict"]
        - raw["pair_macro_strict"]
    )
    return result


def run_domain(domain_name, spec):
    print()
    print("=" * 76)
    print(domain_name)

    train = load_full_dataset(
        f"{domain_name}_TRAIN",
        spec["family"],
        *spec["train"],
    )
    pilot = load_full_dataset(
        f"{domain_name}_PILOT",
        spec["family"],
        *spec["pilot"],
    )
    tests = {
        name: load_full_dataset(
            name,
            spec["family"],
            *split_spec,
        )
        for name, split_spec
        in spec["tests"].items()
    }

    selected, models, grid = (
        cluster.choose_configuration(
            train,
            pilot,
            FEATURE_INDICES,
        )
    )

    print()
    print("Pilot 选择：")
    print(json.dumps(
        {
            key: selected[key]
            for key in [
                "regularization",
                "beta",
                "threshold",
                "top1",
                "pair_macro_strict",
                "damage_rate",
                "correction_rate",
                "switch_rate",
                "damage_constraint_fallback",
            ]
        },
        ensure_ascii=False,
        indent=2,
    ))

    datasets = {
        "PILOT": pilot,
        **tests,
    }
    evaluations = {}

    for name, dataset in datasets.items():
        raw = cluster.evaluate_raw(dataset)
        majority = evaluate_rule(
            dataset,
            "majority",
        )
        weighted = evaluate_rule(
            dataset,
            "weighted_tau4",
        )
        learned = cluster.evaluate_cluster_model(
            dataset,
            FEATURE_INDICES,
            models,
            selected["beta"],
            selected["threshold"],
        )

        evaluations[name] = {
            "raw_rm": raw,
            "majority": with_delta(
                majority,
                raw,
            ),
            "weighted_tau4": with_delta(
                weighted,
                raw,
            ),
            "learned_rm_support": with_delta(
                learned,
                raw,
            ),
        }

        print()
        print(name)

        for method in [
            "majority",
            "weighted_tau4",
            "learned_rm_support",
        ]:
            item = evaluations[name][method]
            print(
                f"  {method}: "
                f"Top1={item['raw_top1']:.6f}"
                f"->{item['top1']:.6f} "
                f"({item['top1_delta']:+.6f}), "
                f"Pair="
                f"{item['raw_pair_macro_strict']:.6f}"
                f"->{item['pair_macro_strict']:.6f} "
                f"({item['pair_delta']:+.6f}), "
                f"Damage={item['damage_rate']:.6f}, "
                f"Correction="
                f"{item['correction_rate']:.6f}, "
                f"Net="
                f"{item['net_corrected_questions']}"
            )

    return {
        "family": spec["family"],
        "feature_indices": FEATURE_INDICES,
        "features": [
            cluster.FEATURE_NAMES[index]
            for index in FEATURE_INDICES
        ],
        "selected": selected,
        "evaluations": evaluations,
        "grid": grid,
    }


def macro_summary(results):
    test_rows = []

    for domain_result in results.values():
        for name, methods in domain_result[
            "evaluations"
        ].items():
            if name == "PILOT":
                continue
            test_rows.append(methods)

    summary = {}

    for method in [
        "raw_rm",
        "majority",
        "weighted_tau4",
        "learned_rm_support",
    ]:
        top1 = []
        pair = []
        damage = []

        for methods in test_rows:
            item = methods[method]
            top1.append(item["top1"])
            pair.append(
                item["pair_macro_strict"]
            )
            damage.append(
                item.get("damage_rate", 0.0)
            )

        summary[method] = {
            "macro_top1": float(
                np.mean(top1)
            ),
            "macro_pair_macro_strict": float(
                np.mean(pair)
            ),
            "macro_damage_rate": float(
                np.mean(damage)
            ),
        }

    raw = summary["raw_rm"]

    for method, item in summary.items():
        item["macro_top1_delta"] = (
            item["macro_top1"]
            - raw["macro_top1"]
        )
        item["macro_pair_delta"] = (
            item["macro_pair_macro_strict"]
            - raw["macro_pair_macro_strict"]
        )

    return summary


def main():
    started = time.time()

    results = {
        domain_name: run_domain(
            domain_name,
            spec,
        )
        for domain_name, spec in DOMAINS.items()
    }

    macro = macro_summary(results)

    output = {
        "version": "answer_cluster_rm_support_full_v1",
        "scope": (
            "full train and pilot; existing full "
            "GSM8K ID, MATH ID and SVAMP OOD tests"
        ),
        "selection": {
            "train": "fit low-dimensional model",
            "pilot": (
                "select regularization, beta and "
                "threshold under damage constraint"
            ),
            "test": "evaluation only",
            "pilot_damage_limit": (
                cluster.PILOT_DAMAGE_LIMIT
            ),
        },
        "results": results,
        "test_macro": macro,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("===== 全量测试宏平均 =====")

    for method, item in macro.items():
        print(
            f"{method}: "
            f"Top1={item['macro_top1']:.6f} "
            f"({item['macro_top1_delta']:+.6f}), "
            f"Pair="
            f"{item['macro_pair_macro_strict']:.6f} "
            f"({item['macro_pair_delta']:+.6f}), "
            f"Damage={item['macro_damage_rate']:.6f}"
        )

    print()
    print("结果：", OUTPUT)
    print("耗时秒：", output["elapsed_seconds"])


if __name__ == "__main__":
    main()
