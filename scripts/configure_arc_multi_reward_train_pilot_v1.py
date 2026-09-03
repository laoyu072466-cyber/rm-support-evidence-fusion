from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import sys
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import explore_answer_cluster_evidence_smoke as cluster


CANDIDATE_PATH = (
    ROOT / "outputs/arc_challenge_v1/"
    "qwen2_7b_full_k16_recovered_candidates.jsonl"
)
SCORE_AUDIT_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_score_audit_v1.json"
)
OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "arc_multi_reward_train_pilot_config_v1.json"
)

CANDIDATE_SHA256 = (
    "eb72914f82678c4738fab24ea055a62b"
    "00f8c207372d3f3ba85d785591267191"
)

FEATURE_INDICES = list(range(8))

FEATURE_NAMES = [
    "log_cluster_support",
    "cluster_support_fraction",
    "log_unique_solution_count",
    "max_within_question_relative_rm",
    "mean_within_question_relative_rm",
    "logmeanexp_within_question_relative_rm",
    "std_within_question_relative_rm",
    "contains_raw_rm_top1",
]

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

EXPECTED_QUESTIONS = {
    "train": 1119,
    "pilot": 299,
}

MODELS = {
    "qwen3_1p7b": {
        "name": (
            "Skywork-Reward-V2-Qwen3-1.7B"
        ),
        "score_file": (
            ROOT / "data/cache/"
            "arc_reward_scores_v1/"
            "Skywork-Reward-V2-Qwen3-1.7B/"
            "arc_full_recovered.scores_f32.npy"
        ),
        "score_sha256": (
            "e2b92df8becba8fa4d9dd4606d5ea4a"
            "5b89a6ae7eb78aa5529af600b4a5d94b0"
        ),
    },
    "internlm2_1p8b": {
        "name": "InternLM2-1.8B-Reward",
        "score_file": (
            ROOT / "data/cache/"
            "arc_reward_scores_v1/"
            "InternLM2-1.8B-Reward/"
            "arc_full_recovered.scores_f32.npy"
        ),
        "score_sha256": (
            "16785a1da4b7da38884137fedb43c39b"
            "1874763fdc54659d9b7ff1449482a31c"
        ),
    },
    "armorm_8b": {
        "name": "ArmoRM-Llama3-8B-v0.1",
        "score_file": (
            ROOT / "data/cache/"
            "arc_reward_scores_v1/"
            "ArmoRM-Llama3-8B-v0.1/"
            "arc_full_recovered.scores_f32.npy"
        ),
        "score_sha256": (
            "09472611d6e1a4129f18772850c9f16e"
            "d2b38e44d1d8dc0d1cb0d9da1e7388cc"
        ),
    },
}


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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            jsonable(item)
            for item in value
        ]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stable_logmeanexp(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    maximum = float(np.max(values))
    return float(
        maximum
        + math.log(
            float(
                np.exp(
                    values - maximum
                ).mean()
            )
        )
    )


def canonical_solution(text):
    function = getattr(
        cluster,
        "canonical_solution",
        None,
    )
    if function is not None:
        return function(text)
    return " ".join(str(text).lower().split())


def parsed_choice_index(row):
    value = row.get(
        "parsed_choice_index_generation_audit"
    )

    if value is not None:
        try:
            value = int(value)
            if (
                0
                <= value
                < len(row["choice_labels"])
            ):
                return value
        except (TypeError, ValueError):
            pass

    answer = row.get(
        "parsed_answer_generation_audit"
    )
    if answer is None:
        return None

    answer = str(answer).strip()
    labels = [
        str(item).strip()
        for item in row["choice_labels"]
    ]

    exact = [
        index
        for index, label in enumerate(labels)
        if answer == label
    ]
    if len(exact) == 1:
        return exact[0]

    upper = answer.upper()
    folded = [
        index
        for index, label in enumerate(labels)
        if upper == label.upper()
    ]
    if len(folded) == 1:
        return folded[0]

    return None


def load_label_map(split):
    path = LABEL_PATHS[split]
    require(
        path.exists(),
        f"缺少 {split} 标签文件",
    )

    rows = read_jsonl(path)
    result = {}

    for row in rows:
        uid = str(row["question_uid"])
        require(
            uid not in result,
            f"{split}: 重复标签 UID：{uid}",
        )
        result[uid] = int(
            row["answer_index"]
        )

    require(
        len(result)
        == EXPECTED_QUESTIONS[split],
        f"{split}: 标签数量错误 "
        f"{len(result)}",
    )

    return result, {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "values_printed": False,
    }


def build_questions(dataset):
    questions = []
    mixed_clusters = 0

    for uid, index_list in dataset[
        "groups"
    ].items():
        indices = np.asarray(
            index_list,
            dtype=np.int64,
        )
        rm = dataset["rm_scores"][indices]
        labels = dataset["labels"][indices]

        rm_relative = cluster.robust_z(rm)
        answer_groups = defaultdict(list)

        for local_index, global_index in enumerate(
            indices
        ):
            row = dataset["rows"][
                int(global_index)
            ]
            parsed = parsed_choice_index(row)

            if parsed is None:
                answer = (
                    "__unparsed__"
                    f"{uid}__"
                    f"{row['candidate_index']}"
                )
            else:
                answer = f"choice_{parsed}"

            answer_groups[answer].append(
                local_index
            )

        raw_local = int(np.argmax(rm))
        raw_answer = next(
            answer
            for answer, members
            in answer_groups.items()
            if raw_local in members
        )

        cluster_items = []
        total_candidates = len(indices)

        for answer, member_list in (
            answer_groups.items()
        ):
            members = np.asarray(
                member_list,
                dtype=np.int64,
            )
            member_labels = labels[members]
            unique_labels = set(
                member_labels.tolist()
            )

            if len(unique_labels) > 1:
                mixed_clusters += 1

            member_rm_relative = (
                rm_relative[members]
            )

            unique_solutions = {
                canonical_solution(
                    dataset["rows"][
                        int(indices[member])
                    ]["solution_text"]
                )
                for member in members
            }

            features = np.asarray([
                math.log1p(len(members)),
                len(members) / total_candidates,
                math.log1p(
                    len(unique_solutions)
                ),
                float(np.max(
                    member_rm_relative
                )),
                float(np.mean(
                    member_rm_relative
                )),
                stable_logmeanexp(
                    member_rm_relative
                ),
                float(np.std(
                    member_rm_relative
                )),
                float(answer == raw_answer),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ], dtype=np.float32)

            cluster_items.append({
                "answer": answer,
                "members": members,
                "label": int(
                    np.mean(member_labels)
                    >= 0.5
                ),
                "features": features,
                "rm_max": float(
                    np.max(rm[members])
                ),
                "rm_logmass_tau4": (
                    stable_logmeanexp(
                        rm[members] / 4.0
                    )
                    + math.log(len(members))
                ),
            })

        has_positive = any(
            item["label"] == 1
            for item in cluster_items
        )
        has_negative = any(
            item["label"] == 0
            for item in cluster_items
        )

        raw_cluster = next(
            index
            for index, item
            in enumerate(cluster_items)
            if raw_local in item["members"]
        )

        questions.append({
            "uid": uid,
            "indices": indices,
            "labels": labels,
            "rm": rm,
            "clusters": cluster_items,
            "raw_local": raw_local,
            "raw_cluster": raw_cluster,
            "cluster_trainable": (
                has_positive
                and has_negative
            ),
        })

    dataset["questions"] = questions
    dataset["mixed_clusters"] = (
        mixed_clusters
    )


def build_dataset(
    name,
    split,
    all_rows,
    all_scores,
    gold,
):
    rows = []
    scores = []
    labels = []
    groups = defaultdict(list)
    seen_questions = set()
    parsed_count = 0

    for global_index, row in enumerate(
        all_rows
    ):
        if row["logical_split"] != split:
            continue

        uid = str(row["question_uid"])
        require(
            uid in gold,
            f"{split}: 缺少标签 {uid}",
        )

        local_index = len(rows)
        parsed = parsed_choice_index(row)

        rows.append(row)
        scores.append(
            float(all_scores[global_index])
        )
        labels.append(
            int(
                parsed is not None
                and parsed == gold[uid]
            )
        )
        groups[uid].append(local_index)
        seen_questions.add(uid)

        if parsed is not None:
            parsed_count += 1

    require(
        len(seen_questions)
        == EXPECTED_QUESTIONS[split],
        f"{split}: 题目数错误",
    )
    require(
        len(rows)
        == EXPECTED_QUESTIONS[split] * 16,
        f"{split}: 候选数错误",
    )
    require(
        seen_questions == set(gold),
        f"{split}: 标签关联不完整",
    )

    dataset = {
        "name": name,
        "family": "ARC_CHALLENGE",
        "rows": rows,
        "labels": np.asarray(
            labels,
            dtype=np.int8,
        ),
        "rm_scores": np.asarray(
            scores,
            dtype=np.float32,
        ),
        "groups": dict(groups),
    }
    build_questions(dataset)

    untrainable = sum(
        not question["cluster_trainable"]
        for question in dataset["questions"]
    )

    print(
        f"{name}: "
        f"问题={len(dataset['questions'])}, "
        f"候选={len(rows)}, "
        f"解析={parsed_count}/{len(rows)}, "
        f"混合簇={dataset['mixed_clusters']}, "
        f"不可训练问题={untrainable}",
        flush=True,
    )

    return dataset, {
        "questions": len(
            dataset["questions"]
        ),
        "candidates": len(rows),
        "parsed_candidates": parsed_count,
        "parse_rate": (
            parsed_count / len(rows)
        ),
        "correct_candidates": int(
            np.sum(dataset["labels"])
        ),
        "candidate_accuracy": float(
            np.mean(dataset["labels"])
        ),
        "mixed_clusters": int(
            dataset["mixed_clusters"]
        ),
        "untrainable_questions": int(
            untrainable
        ),
    }


def raw_metrics(dataset):
    return cluster.evaluate_raw(dataset)


def model_metrics(
    dataset,
    models,
    beta,
    threshold,
):
    return cluster.evaluate_cluster_model(
        dataset,
        FEATURE_INDICES,
        models,
        beta,
        threshold,
    )


def serialize_models(models):
    return [
        {
            "weights": model["weights"],
            "mean": model["mean"],
            "std": model["std"],
            "regularization": model[
                "regularization"
            ],
            "seed": model["seed"],
        }
        for model in models
    ]


def main():
    started = time.time()

    print(
        "===== ARC 多 RM Train/Pilot 配置 ====="
    )
    print(
        "读取标签：Train + Pilot"
    )
    print("读取 Test 标签：False")
    print(
        "Test 候选不参与训练或选择：True"
    )

    require(
        sha256_file(CANDIDATE_PATH)
        == CANDIDATE_SHA256,
        "候选 SHA256 不一致",
    )

    score_audit = json.loads(
        SCORE_AUDIT_PATH.read_text(
            encoding="utf-8"
        )
    )
    require(
        score_audit.get("labels_loaded")
        is False,
        "评分审计标签状态错误",
    )
    require(
        score_audit.get(
            "sealed_test_labels_loaded"
        ) is False,
        "评分阶段读取了 Test 标签",
    )

    all_rows = read_jsonl(
        CANDIDATE_PATH
    )
    require(
        len(all_rows) == 41440,
        "候选数量错误",
    )

    train_gold, train_label_record = (
        load_label_map("train")
    )
    pilot_gold, pilot_label_record = (
        load_label_map("pilot")
    )

    result = {
        "version": (
            "arc_multi_reward_"
            "train_pilot_config_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "task": "ARC-Challenge",
        "candidate_source": {
            "path": str(
                CANDIDATE_PATH.relative_to(
                    ROOT
                )
            ),
            "sha256": CANDIDATE_SHA256,
            "questions": 2590,
            "candidates": 41440,
        },
        "protocol": {
            "train_labels_loaded": True,
            "pilot_labels_loaded": True,
            "test_labels_loaded": False,
            "test_candidates_used_for_training": (
                False
            ),
            "test_candidates_used_for_selection": (
                False
            ),
            "selection": (
                "fit on ARC train; choose on "
                "ARC validation/pilot under "
                "the existing damage constraint"
            ),
        },
        "label_sources": {
            "train": train_label_record,
            "pilot": pilot_label_record,
        },
        "implementation": {
            "cluster_module": (
                "scripts/"
                "explore_answer_cluster_"
                "evidence_smoke.py"
            ),
            "cluster_module_sha256": (
                sha256_file(
                    ROOT / "scripts/"
                    "explore_answer_cluster_"
                    "evidence_smoke.py"
                )
            ),
            "feature_indices": (
                FEATURE_INDICES
            ),
            "feature_names": FEATURE_NAMES,
            "unused_feature_slots_zeroed": (
                list(range(8, 14))
            ),
            "prompt_agreement_constant": 1.0,
            "regularization_grid": getattr(
                cluster,
                "REGULARIZATION",
            ),
            "beta_grid": getattr(
                cluster,
                "BETA_GRID",
            ),
            "threshold_grid": getattr(
                cluster,
                "THRESHOLD_GRID",
            ),
            "seeds": getattr(
                cluster,
                "SEEDS",
            ),
            "pilot_damage_limit": getattr(
                cluster,
                "PILOT_DAMAGE_LIMIT",
            ),
        },
        "models": {},
    }

    for model_key, spec in MODELS.items():
        print()
        print("=" * 76)
        print(spec["name"])

        require(
            sha256_file(spec["score_file"])
            == spec["score_sha256"],
            f"{model_key}: 分数哈希不一致",
        )

        all_scores = np.load(
            spec["score_file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        require(
            all_scores.shape == (41440,),
            f"{model_key}: 分数形状错误",
        )
        require(
            bool(np.all(np.isfinite(
                all_scores
            ))),
            f"{model_key}: 分数非有限",
        )

        train, train_summary = (
            build_dataset(
                f"{model_key}_TRAIN",
                "train",
                all_rows,
                all_scores,
                train_gold,
            )
        )
        pilot, pilot_summary = (
            build_dataset(
                f"{model_key}_PILOT",
                "pilot",
                all_rows,
                all_scores,
                pilot_gold,
            )
        )

        selected, fitted, grid = (
            cluster.choose_configuration(
                train,
                pilot,
                FEATURE_INDICES,
            )
        )

        raw = raw_metrics(pilot)
        ungated = model_metrics(
            pilot,
            fitted,
            selected["beta"],
            0.0,
        )
        gated = model_metrics(
            pilot,
            fitted,
            selected["beta"],
            selected["threshold"],
        )

        print(
            "选择："
            f"reg={selected['regularization']}, "
            f"beta={selected['beta']}, "
            f"threshold={selected['threshold']}"
        )
        print(
            "Pilot："
            f"Raw Top1={raw['top1']:.6f}, "
            f"Ungated={ungated['top1']:.6f}, "
            f"Gate={gated['top1']:.6f}, "
            f"Δ={gated['top1'] - raw['top1']:+.6f}, "
            f"damage={gated['damage_rate']:.6f}"
        )

        result["models"][model_key] = {
            "model_name": spec["name"],
            "score_file": str(
                spec["score_file"].relative_to(
                    ROOT
                )
            ),
            "score_sha256": (
                spec["score_sha256"]
            ),
            "train": train_summary,
            "pilot": pilot_summary,
            "selected": selected,
            "pilot_metrics": {
                "raw": raw,
                "ungated_hybrid": ungated,
                "frozen_gate": gated,
            },
            "fitted_models": (
                serialize_models(fitted)
            ),
            "selection_grid": grid,
        }

    result["elapsed_seconds"] = (
        time.time() - started
    )
    result["decision"] = (
        "freeze_all_three_configurations_"
        "before_first_test_label_access"
    )

    atomic_json(OUTPUT_PATH, result)

    print()
    print("=" * 76)
    print("===== 配置汇总 =====")

    summary = {
        key: {
            "model_name": value[
                "model_name"
            ],
            "selected": {
                field: value["selected"][field]
                for field in [
                    "regularization",
                    "beta",
                    "threshold",
                    "top1",
                    "pair_macro_strict",
                    "damage_rate",
                    "switch_rate",
                ]
                if field in value["selected"]
            },
            "raw_pilot_top1": value[
                "pilot_metrics"
            ]["raw"]["top1"],
            "gated_pilot_top1": value[
                "pilot_metrics"
            ]["frozen_gate"]["top1"],
        }
        for key, value
        in result["models"].items()
    }

    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ))
    print(
        "ARC_TRAIN_PILOT_CONFIG_FREEZE_READY"
    )
    print("结果：", OUTPUT_PATH)
    print(
        "耗时秒：",
        round(result["elapsed_seconds"], 3),
    )


if __name__ == "__main__":
    main()
