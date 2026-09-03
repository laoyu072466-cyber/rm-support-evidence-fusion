from pathlib import Path
from collections import defaultdict
import json
import math
import sys

import numpy as np


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import explore_answer_cluster_evidence_smoke as cluster
import configure_arc_multi_reward_train_pilot_v1 as base


OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_train_pilot_config_v2.json"
)
V1_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_train_pilot_config_v1.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def arc_ranking_metrics(
    labels,
    method_scores,
    question_ids,
    raw_scores,
):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )
    method_scores = np.asarray(
        method_scores,
        dtype=np.float64,
    )
    raw_scores = np.asarray(
        raw_scores,
        dtype=np.float64,
    )

    require(
        len(labels)
        == len(method_scores)
        == len(raw_scores)
        == len(question_ids),
        "ARC ranking_metrics 输入长度不一致",
    )
    require(
        bool(np.all(np.isfinite(
            method_scores
        ))),
        "ARC method scores 含非有限值",
    )
    require(
        bool(np.all(np.isfinite(
            raw_scores
        ))),
        "ARC raw scores 含非有限值",
    )
    require(
        set(np.unique(labels)).issubset(
            {0, 1}
        ),
        "ARC 标签必须为 0/1",
    )

    groups = defaultdict(list)
    for index, uid in enumerate(
        question_ids
    ):
        groups[str(uid)].append(index)

    top1 = []
    pair_values = []
    raw_correct_flags = []
    method_correct_flags = []
    switches = []

    for uid, index_list in groups.items():
        indices = np.asarray(
            index_list,
            dtype=np.int64,
        )
        local_labels = labels[indices]
        local_method = method_scores[indices]
        local_raw = raw_scores[indices]

        method_choice = int(
            np.argmax(local_method)
        )
        raw_choice = int(
            np.argmax(local_raw)
        )

        method_correct = int(
            local_labels[method_choice] == 1
        )
        raw_correct = int(
            local_labels[raw_choice] == 1
        )

        top1.append(method_correct)
        method_correct_flags.append(
            method_correct
        )
        raw_correct_flags.append(raw_correct)
        switches.append(
            int(method_choice != raw_choice)
        )

        positive = local_method[
            local_labels == 1
        ]
        negative = local_method[
            local_labels == 0
        ]

        if (
            len(positive) > 0
            and len(negative) > 0
        ):
            strict_wins = (
                positive[:, None]
                > negative[None, :]
            )
            pair_values.append(
                float(np.mean(strict_wins))
            )

    require(
        len(groups) > 0,
        "ARC 没有问题分组",
    )
    require(
        len(pair_values) > 0,
        "ARC 没有可比较的正确—错误候选对",
    )

    raw_correct_flags = np.asarray(
        raw_correct_flags,
        dtype=np.int8,
    )
    method_correct_flags = np.asarray(
        method_correct_flags,
        dtype=np.int8,
    )

    damages = int(np.sum(
        (raw_correct_flags == 1)
        & (method_correct_flags == 0)
    ))
    corrections = int(np.sum(
        (raw_correct_flags == 0)
        & (method_correct_flags == 1)
    ))

    raw_correct_count = int(np.sum(
        raw_correct_flags == 1
    ))
    raw_wrong_count = int(np.sum(
        raw_correct_flags == 0
    ))

    return {
        "questions": len(groups),
        "top1": float(np.mean(top1)),
        "pair_macro_strict": float(
            np.mean(pair_values)
        ),
        "pair_eligible_questions": int(
            len(pair_values)
        ),
        "damage_rate": float(
            damages
            / max(raw_correct_count, 1)
        ),
        "correction_rate": float(
            corrections
            / max(raw_wrong_count, 1)
        ),
        "switch_rate": float(
            np.mean(switches)
        ),
        "net_corrected_questions": int(
            corrections - damages
        ),
    }


def same_configuration(left, right):
    fields = [
        "regularization",
        "beta",
        "threshold",
    ]
    return all(
        float(left[field])
        == float(right[field])
        for field in fields
    )


def validate_grid(record):
    grid = record["selection_grid"]

    require(
        len(grid) == 100,
        "配置网格数量不是 100",
    )
    require(
        all(
            math.isfinite(float(
                item["pair_macro_strict"]
            ))
            for item in grid
        ),
        "v2 网格仍有非有限 Pairwise",
    )

    eligible = [
        item
        for item in grid
        if float(item["damage_rate"])
        <= float(
            cluster.PILOT_DAMAGE_LIMIT
        )
    ]

    if eligible:
        pool = eligible
    else:
        minimum_damage = min(
            float(item["damage_rate"])
            for item in grid
        )
        pool = [
            item
            for item in grid
            if float(item["damage_rate"])
            == minimum_damage
        ]

    expected = max(
        pool,
        key=lambda item: (
            float(item["top1"]),
            float(
                item["pair_macro_strict"]
            ),
            -float(item["damage_rate"]),
            -float(item["switch_rate"]),
        ),
    )

    require(
        same_configuration(
            expected,
            record["selected"],
        ),
        "冻结配置不符合确定性决胜规则",
    )

    return {
        "grid_entries": len(grid),
        "finite_pair_entries": sum(
            math.isfinite(float(
                item["pair_macro_strict"]
            ))
            for item in grid
        ),
        "pair_eligible_questions": int(
            record["selected"].get(
                "pair_eligible_questions",
                0,
            )
        ),
        "selection_verified": True,
    }


def main():
    print(
        "===== ARC 多 RM Train/Pilot v2 配置 ====="
    )
    print("读取标签：Train + Pilot")
    print("读取 Test 标签：False")
    print(
        "修复：ARC 正确候选—错误候选严格 Pairwise"
    )

    v1 = json.loads(
        V1_PATH.read_text(encoding="utf-8")
    )

    cluster.ranking_metrics = (
        arc_ranking_metrics
    )
    base.cluster.ranking_metrics = (
        arc_ranking_metrics
    )
    base.OUTPUT_PATH = OUTPUT_PATH

    base.main()

    manifest = json.loads(
        OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )
    manifest["version"] = (
        "arc_multi_reward_"
        "train_pilot_config_v2"
    )
    manifest["supersedes"] = (
        "arc_multi_reward_"
        "train_pilot_config_v1"
    )
    manifest["metric_compatibility_fix"] = {
        "reason": (
            "The generic evaluator produced "
            "no finite ARC pairwise values."
        ),
        "scope": (
            "ARC Train/Pilot configuration only"
        ),
        "definition": (
            "For each eligible question, compute "
            "the fraction of all correct-candidate "
            "and incorrect-candidate pairs for which "
            "the correct candidate has a strictly "
            "higher score; macro-average by question."
        ),
        "ties": "count_as_non_wins",
        "test_labels_used": False,
        "v1_preserved": True,
    }

    validation = {}
    comparison = {}

    for model_key, record in (
        manifest["models"].items()
    ):
        validation[model_key] = (
            validate_grid(record)
        )

        old = v1["models"][
            model_key
        ]["selected"]
        new = record["selected"]

        comparison[model_key] = {
            "v1": {
                key: old.get(key)
                for key in [
                    "regularization",
                    "beta",
                    "threshold",
                    "top1",
                    "damage_rate",
                    "switch_rate",
                ]
            },
            "v2": {
                key: new.get(key)
                for key in [
                    "regularization",
                    "beta",
                    "threshold",
                    "top1",
                    "pair_macro_strict",
                    "pair_eligible_questions",
                    "damage_rate",
                    "switch_rate",
                ]
            },
            "configuration_changed": (
                not same_configuration(
                    old,
                    new,
                )
            ),
        }

    manifest[
        "v2_grid_validation"
    ] = validation
    manifest[
        "v1_v2_selection_comparison"
    ] = comparison
    manifest["decision"] = (
        "freeze_v2_configurations_before_"
        "first_test_label_access"
    )

    base.atomic_json(
        OUTPUT_PATH,
        manifest,
    )

    print()
    print("=" * 76)
    print("===== v1 → v2 配置比较 =====")
    print(json.dumps(
        comparison,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print(json.dumps(
        validation,
        ensure_ascii=False,
        indent=2,
    ))
    print(
        "ARC_TRAIN_PILOT_CONFIG_V2_FREEZE_READY"
    )
    print("结果：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
