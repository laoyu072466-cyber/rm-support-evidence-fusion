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
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answer_cluster_generator_full as evaluation
import explore_answer_cluster_evidence_smoke as cluster


OUTPUT_PATH = (
    ROOT / "data/manifests/"
    "reliability_aware_math_risk_gate_train_pilot_v2.json"
)
PROTOCOL_TAG = (
    "reliability-aware-math-risk-gate-protocol-v2"
)

FEATURE_INDICES = list(range(8))
REGULARIZATION_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]
ACCEPTANCE_THRESHOLDS = [
    0.000,
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.300,
    0.500,
    1.000,
]

FEATURE_NAMES = [
    "raw_rm_margin_vs_runnerup",
    "proposal_hybrid_advantage_over_raw",
    "proposal_hybrid_margin_vs_runnerup",
    "learned_proposal_minus_raw",
    "rm_proposal_minus_raw",
    "proposal_support_fraction",
    "raw_support_fraction",
    "proposal_minus_raw_support",
    "maximum_support_fraction",
    "top_two_support_margin",
    "normalized_vote_entropy",
    "raw_is_plurality",
    "proposal_is_plurality",
    "log_cluster_count",
    "ensemble_proposal_margin_std",
]

MODEL_ROOTS = {
    "qwen3_1p7b": {
        "name": (
            "Skywork-Reward-V2-Qwen3-1.7B"
        ),
        "root": (
            ROOT / "data/cache/"
            "trajectory_features_v1/"
            "Skywork-Reward-V2-Qwen3-1.7B/"
            "layer_28"
        ),
    },
    "qwen3_4b": {
        "name": (
            "Skywork-Reward-V2-Qwen3-4B"
        ),
        "root": (
            ROOT / "data/cache/"
            "reward_scores_full_v1/"
            "Skywork-Reward-V2-Qwen3-4B"
        ),
    },
    "llama_8b_v2": {
        "name": (
            "Skywork-Reward-V2-Llama-3.1-8B"
        ),
        "root": (
            ROOT / "data/cache/"
            "reward_scores_full_v1/"
            "Skywork-Reward-V2-Llama-3.1-8B"
        ),
    },
    "armorm_8b": {
        "name": "ArmoRM-Llama3-8B-v0.1",
        "root": (
            ROOT / "data/cache/"
            "reward_scores_full_v1/"
            "ArmoRM-Llama3-8B-v0.1"
        ),
    },
    "internlm2_1p8b": {
        "name": "InternLM2-1.8B-Reward",
        "root": (
            ROOT / "data/cache/"
            "reward_scores_full_v1/"
            "InternLM2-1.8B-Reward"
        ),
    },
}


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


def atomic_json(path, value):
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


def verify_protocol_tag():
    relative = str(
        Path(__file__).resolve().relative_to(
            ROOT
        )
    )
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{PROTOCOL_TAG}:{relative}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    require(
        completed.returncode == 0,
        "无法读取冻结协议标签",
    )
    require(
        completed.stdout
        == Path(__file__).read_bytes(),
        "当前训练脚本不同于冻结协议",
    )


def preflight(require_frozen):
    print(
        "===== Reliability-Aware Math Harm-Risk Gate 预检 ====="
    )
    print("允许标签：GSM8K/MATH Train + Pilot")
    print("读取任何 Test 数据：False")
    print("加载奖励模型：False")

    require(
        not OUTPUT_PATH.exists(),
        "训练结果已经存在，拒绝覆盖",
    )

    if require_frozen:
        verify_protocol_tag()

    expected = {
        "gsm_train": 24221,
        "gsm_pilot": 2985,
        "math_train": 16229,
        "math_pilot": 2040,
    }

    for model_key, spec in (
        MODEL_ROOTS.items()
    ):
        for prefix, expected_count in (
            expected.items()
        ):
            path = (
                spec["root"]
                / f"{prefix}.scores_f32.npy"
            )
            require(
                path.exists(),
                f"{model_key}: 缺少 {prefix}",
            )
            values = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
            require(
                values.shape
                == (expected_count,),
                f"{model_key}/{prefix}: "
                "形状错误",
            )
            require(
                bool(np.all(np.isfinite(
                    values
                ))),
                f"{model_key}/{prefix}: "
                "非有限分数",
            )

    for domain_name, spec in (
        evaluation.DOMAINS.items()
    ):
        for role in ["train", "pilot"]:
            filename, prefix = spec[role]
            require(
                (
                    evaluation.DATA_ROOT
                    / filename
                ).exists(),
                f"{domain_name}: 缺少 {role}",
            )

            generator_root = (
                evaluation.GEN_ROOT
                / spec["model"]
            )
            for suffix in [
                "terminal_hidden_f16.npy",
                "token_nll_f32.npy",
                "labels_i8.npy",
                "metadata.jsonl",
            ]:
                require(
                    (
                        generator_root
                        / f"{prefix}.{suffix}"
                    ).exists(),
                    f"{domain_name}/{role}: "
                    f"缺少 {suffix}",
                )

    print("奖励模型：", len(MODEL_ROOTS))
    print("数学域：", list(
        evaluation.DOMAINS
    ))
    print(
        "Reliability 特征：",
        len(FEATURE_NAMES),
    )
    print(
        "配置数量：",
        len(REGULARIZATION_GRID)
        * len(ACCEPTANCE_THRESHOLDS),
    )
    print(
        "RELIABILITY_MATH_PROTOCOL_READY"
    )


def frozen_models(models):
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


def max_other(values, index):
    if len(values) <= 1:
        return float(values[index])
    mask = np.ones(
        len(values),
        dtype=bool,
    )
    mask[index] = False
    return float(np.max(values[mask]))


def reliability_features(
    question,
    fitted_models,
    learned,
    rm_cluster,
    hybrid,
    proposal_cluster,
):
    raw_cluster = question["raw_cluster"]

    supports = np.asarray([
        len(item["members"])
        for item in question["clusters"]
    ], dtype=np.float64)
    fractions = (
        supports / np.sum(supports)
    )

    if len(fractions) > 1:
        entropy = float(
            -np.sum(
                fractions
                * np.log(
                    np.maximum(
                        fractions,
                        1e-12,
                    )
                )
            )
            / math.log(len(fractions))
        )
        sorted_support = np.sort(
            fractions
        )[::-1]
        support_margin = float(
            sorted_support[0]
            - sorted_support[1]
        )
    else:
        entropy = 0.0
        support_margin = 1.0

    matrix = np.stack([
        item["features"][
            FEATURE_INDICES
        ]
        for item in question["clusters"]
    ]).astype(np.float32)

    ensemble_margins = []

    for model in fitted_models:
        normalized = (
            matrix - model["mean"]
        ) / model["std"]
        seed_scores = (
            normalized @ model["weights"]
        )
        seed_scores = cluster.ordinary_z(
            seed_scores
        )
        ensemble_margins.append(
            float(
                seed_scores[proposal_cluster]
                - seed_scores[raw_cluster]
            )
        )

    maximum_support = float(
        np.max(fractions)
    )

    return np.asarray([
        float(
            rm_cluster[raw_cluster]
            - max_other(
                rm_cluster,
                raw_cluster,
            )
        ),
        float(
            hybrid[proposal_cluster]
            - hybrid[raw_cluster]
        ),
        float(
            hybrid[proposal_cluster]
            - max_other(
                hybrid,
                proposal_cluster,
            )
        ),
        float(
            learned[proposal_cluster]
            - learned[raw_cluster]
        ),
        float(
            rm_cluster[proposal_cluster]
            - rm_cluster[raw_cluster]
        ),
        float(fractions[proposal_cluster]),
        float(fractions[raw_cluster]),
        float(
            fractions[proposal_cluster]
            - fractions[raw_cluster]
        ),
        maximum_support,
        support_margin,
        entropy,
        float(
            fractions[raw_cluster]
            == maximum_support
        ),
        float(
            fractions[proposal_cluster]
            == maximum_support
        ),
        math.log1p(len(fractions)),
        float(np.std(ensemble_margins)),
    ], dtype=np.float64)


def prepare_decisions(
    dataset,
    fitted_models,
    beta,
):
    accepted_scores = np.empty(
        len(dataset["labels"]),
        dtype=np.float32,
    )
    rejected_scores = np.empty(
        len(dataset["labels"]),
        dtype=np.float32,
    )
    switches = []

    for question_number, question in enumerate(
        dataset["questions"]
    ):
        learned = cluster.model_cluster_scores(
            question,
            FEATURE_INDICES,
            fitted_models,
        )
        learned = cluster.ordinary_z(
            learned
        )

        rm_cluster = np.asarray([
            item["rm_max"]
            for item in question["clusters"]
        ], dtype=np.float32)
        rm_cluster = cluster.ordinary_z(
            rm_cluster
        )

        hybrid = (
            rm_cluster
            + float(beta) * learned
        )

        accepted_local = (
            cluster.candidate_scores_from_clusters(
                question,
                hybrid,
                threshold=0.0,
            )
        )
        rejected_local = (
            cluster.candidate_scores_from_clusters(
                question,
                hybrid,
                threshold=float("inf"),
            )
        )

        indices = question["indices"]
        accepted_scores[indices] = (
            accepted_local
        )
        rejected_scores[indices] = (
            rejected_local
        )

        raw_choice = int(
            question["raw_local"]
        )
        proposal_choice = int(
            np.argmax(accepted_local)
        )

        if proposal_choice == raw_choice:
            continue

        proposal_cluster = next(
            cluster_index
            for cluster_index, item
            in enumerate(question["clusters"])
            if proposal_choice in item["members"]
        )

        raw_correct = int(
            question["labels"][
                raw_choice
            ] == 1
        )
        proposal_correct = int(
            question["labels"][
                proposal_choice
            ] == 1
        )

        switches.append({
            "question_number": (
                question_number
            ),
            "indices": indices,
            "features": reliability_features(
                question,
                fitted_models,
                learned,
                rm_cluster,
                hybrid,
                proposal_cluster,
            ),
            "raw_correct": raw_correct,
            "proposal_correct": (
                proposal_correct
            ),
            "utility": (
                proposal_correct
                - raw_correct
            ),
        })

    require(
        bool(np.all(np.isfinite(
            accepted_scores
        ))),
        "Ungated proposal 分数非有限",
    )
    require(
        bool(np.all(np.isfinite(
            rejected_scores
        ))),
        "Raw fallback 分数非有限",
    )

    return {
        "dataset": dataset,
        "accepted_scores": accepted_scores,
        "rejected_scores": rejected_scores,
        "switches": switches,
    }


def decision_statistics(prepared):
    utilities = Counter(
        item["utility"]
        for item in prepared["switches"]
    )
    return {
        "questions": len(
            prepared["dataset"]["questions"]
        ),
        "proposal_switches": len(
            prepared["switches"]
        ),
        "beneficial_switches": int(
            utilities[1]
        ),
        "harmful_switches": int(
            utilities[-1]
        ),
        "neutral_switches": int(
            utilities[0]
        ),
    }


def train_reliability_model(
    train_x,
    train_y,
    regularization,
):
    mean = np.mean(train_x, axis=0)
    std = np.maximum(
        np.std(train_x, axis=0),
        1e-6,
    )
    normalized = (
        train_x - mean
    ) / std

    model = LogisticRegression(
        C=float(regularization),
        solver="lbfgs",
        class_weight=None,
        max_iter=5000,
        random_state=20260911,
    )
    model.fit(normalized, train_y)

    return {
        "regularization": float(
            regularization
        ),
        "mean": mean,
        "std": std,
        "weights": (
            model.coef_[0].astype(
                np.float64
            )
        ),
        "intercept": float(
            model.intercept_[0]
        ),
        "iterations": int(
            np.max(model.n_iter_)
        ),
    }


def predict_probability(model, matrix):
    normalized = (
        matrix - model["mean"]
    ) / model["std"]
    logits = (
        normalized @ model["weights"]
        + model["intercept"]
    )
    logits = np.clip(
        logits,
        -40.0,
        40.0,
    )
    return (
        1.0 / (1.0 + np.exp(-logits))
    )


def apply_reliability_gate(
    prepared,
    reliability_model,
    threshold,
):
    output = prepared[
        "accepted_scores"
    ].copy()
    switches = prepared["switches"]

    if not switches:
        return output, []

    matrix = np.stack([
        item["features"]
        for item in switches
    ])
    probabilities = predict_probability(
        reliability_model,
        matrix,
    )

    decisions = []

    for item, probability in zip(
        switches,
        probabilities,
    ):
        accept = bool(
            probability <= threshold
        )

        if not accept:
            indices = item["indices"]
            output[indices] = prepared[
                "rejected_scores"
            ][indices]

        decisions.append({
            "probability": float(
                probability
            ),
            "accepted": accept,
            "utility": item["utility"],
        })

    return output, decisions


def standard_metrics(dataset, scores):
    return evaluation.standard_metrics(
        dataset,
        scores,
    )


def macro_metrics(metrics):
    fields = [
        "top1",
        "pair_macro_strict",
        "damage_rate",
        "correction_rate",
        "switch_rate",
    ]
    return {
        field: float(np.mean([
            value[field]
            for value in metrics.values()
        ]))
        for field in fields
    }


def add_delta(metrics, raw):
    result = dict(metrics)
    result["top1_delta"] = (
        metrics["top1"] - raw["top1"]
    )
    result["pair_delta"] = (
        metrics["pair_macro_strict"]
        - raw["pair_macro_strict"]
    )
    return result


def summarize_decisions(decisions):
    result = Counter()

    for item in decisions:
        if item["accepted"]:
            result["accepted"] += 1
            if item["utility"] == 1:
                result[
                    "accepted_corrections"
                ] += 1
            elif item["utility"] == -1:
                result[
                    "accepted_damages"
                ] += 1
            else:
                result[
                    "accepted_neutral"
                ] += 1
        else:
            result["rejected"] += 1
            if item["utility"] == 1:
                result[
                    "blocked_corrections"
                ] += 1
            elif item["utility"] == -1:
                result[
                    "prevented_damages"
                ] += 1
            else:
                result[
                    "rejected_neutral"
                ] += 1

    probabilities = [
        item["probability"]
        for item in decisions
    ]
    result["mean_harm_probability"] = (
        float(np.mean(probabilities))
        if probabilities
        else None
    )
    return dict(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    args = parser.parse_args()

    preflight(
        require_frozen=(
            not args.preflight_only
        )
    )

    if args.preflight_only:
        print(
            "Preflight-only 完成；"
            "未读取 Train/Pilot 标签。"
        )
        return

    started = time.time()

    print()
    print(
        "===== 训练 Reliability-Aware Math Harm-Risk Gate ====="
    )
    print("训练标签：GSM8K + MATH Train")
    print("配置标签：GSM8K + MATH Pilot")
    print("读取任何 Test：False")
    print("加载奖励模型：False")

    sources = {}
    base_records = {}
    train_feature_rows = []
    train_targets = []

    for model_key, model_spec in (
        MODEL_ROOTS.items()
    ):
        evaluation.RM_ROOT = (
            model_spec["root"]
        )

        for domain_name, domain_spec in (
            evaluation.DOMAINS.items()
        ):
            source_name = (
                f"{model_key}__{domain_name}"
            )

            print()
            print("=" * 76)
            print(
                model_spec["name"],
                "/",
                domain_name,
            )

            train = evaluation.load_dataset(
                f"{source_name}_TRAIN",
                domain_spec,
                *domain_spec["train"],
            )
            pilot = evaluation.load_dataset(
                f"{source_name}_PILOT",
                domain_spec,
                *domain_spec["pilot"],
            )

            selected, fitted, _ = (
                cluster.choose_configuration(
                    train,
                    pilot,
                    FEATURE_INDICES,
                )
            )

            prepared_train = (
                prepare_decisions(
                    train,
                    fitted,
                    selected["beta"],
                )
            )
            prepared_pilot = (
                prepare_decisions(
                    pilot,
                    fitted,
                    selected["beta"],
                )
            )

            for item in (
                prepared_train["switches"]
            ):
                train_feature_rows.append(
                    item["features"]
                )
                train_targets.append(
                    int(item["utility"] == -1)
                )

            raw_pilot = standard_metrics(
                pilot,
                pilot["rm_scores"],
            )
            ungated_pilot = (
                standard_metrics(
                    pilot,
                    prepared_pilot[
                        "accepted_scores"
                    ],
                )
            )
            fixed_scores = (
                evaluation.predict_learned(
                    pilot,
                    FEATURE_INDICES,
                    fitted,
                    selected["beta"],
                    selected["threshold"],
                )
            )
            fixed_pilot = standard_metrics(
                pilot,
                fixed_scores,
            )

            sources[source_name] = {
                "model_key": model_key,
                "model_name": (
                    model_spec["name"]
                ),
                "domain": domain_name,
                "pilot": pilot,
                "prepared_pilot": (
                    prepared_pilot
                ),
                "raw_metrics": raw_pilot,
                "ungated_metrics": (
                    ungated_pilot
                ),
                "fixed_metrics": fixed_pilot,
            }

            base_records[source_name] = {
                "model_key": model_key,
                "model_name": (
                    model_spec["name"]
                ),
                "domain": domain_name,
                "score_root": str(
                    model_spec["root"]
                    .relative_to(ROOT)
                ),
                "score_sha256": {
                    role: sha256_file(
                        model_spec["root"]
                        / (
                            f"{domain_spec[role][1]}"
                            ".scores_f32.npy"
                        )
                    )
                    for role in [
                        "train",
                        "pilot",
                    ]
                },
                "selected_base_fusion": {
                    key: selected[key]
                    for key in [
                        "regularization",
                        "beta",
                        "threshold",
                    ]
                },
                "fitted_base_models": (
                    frozen_models(fitted)
                ),
                "train_switches": (
                    decision_statistics(
                        prepared_train
                    )
                ),
                "pilot_switches": (
                    decision_statistics(
                        prepared_pilot
                    )
                ),
                "pilot_baselines": {
                    "raw_rm": raw_pilot,
                    "ungated_hybrid": (
                        ungated_pilot
                    ),
                    "fixed_gate": fixed_pilot,
                },
            }

            print(
                "Base："
                f"reg={selected['regularization']}, "
                f"beta={selected['beta']}, "
                f"threshold="
                f"{selected['threshold']}"
            )
            print(
                "Train switches：",
                decision_statistics(
                    prepared_train
                ),
            )

    train_x = np.stack(
        train_feature_rows
    ).astype(np.float64)
    train_y = np.asarray(
        train_targets,
        dtype=np.int8,
    )

    require(
        train_x.shape[1]
        == len(FEATURE_NAMES),
        "Reliability 特征维度错误",
    )
    require(
        set(train_y.tolist()) == {0, 1},
        "Reliability 训练缺少正类或负类",
    )

    print()
    print("===== Harm-Risk 训练样本 =====")
    print("all switches:", len(train_y))
    print("harmful:", int(np.sum(
        train_y == 1
    )))
    print("nonharmful:", int(np.sum(
        train_y == 0
    )))

    fitted_reliability = {}
    grid = []

    for regularization in (
        REGULARIZATION_GRID
    ):
        model = train_reliability_model(
            train_x,
            train_y,
            regularization,
        )
        fitted_reliability[
            regularization
        ] = model

        for threshold in (
            ACCEPTANCE_THRESHOLDS
        ):
            source_metrics = {}

            for source_name, source in (
                sources.items()
            ):
                scores, _ = (
                    apply_reliability_gate(
                        source[
                            "prepared_pilot"
                        ],
                        model,
                        threshold,
                    )
                )
                source_metrics[
                    source_name
                ] = standard_metrics(
                    source["pilot"],
                    scores,
                )

            macro = macro_metrics(
                source_metrics
            )
            grid.append({
                "regularization": float(
                    regularization
                ),
                "threshold": float(
                    threshold
                ),
                **macro,
            })

    eligible = [
        item
        for item in grid
        if item["damage_rate"]
        <= cluster.PILOT_DAMAGE_LIMIT
    ]

    if eligible:
        pool = eligible
        fallback = False
    else:
        minimum_damage = min(
            item["damage_rate"]
            for item in grid
        )
        pool = [
            item
            for item in grid
            if item["damage_rate"]
            == minimum_damage
        ]
        fallback = True

    selected = max(
        pool,
        key=lambda item: (
            item["top1"],
            item["pair_macro_strict"],
            -item["damage_rate"],
            -item["switch_rate"],
        ),
    ).copy()
    selected[
        "damage_constraint_fallback"
    ] = fallback
    selected[
        "maximum_harm_probability"
    ] = float(selected["threshold"])
    selected[
        "accept_all_baseline_in_grid"
    ] = True

    reliability_model = (
        fitted_reliability[
            selected["regularization"]
        ]
    )

    selected_source_results = {}

    for source_name, source in (
        sources.items()
    ):
        scores, decisions = (
            apply_reliability_gate(
                source["prepared_pilot"],
                reliability_model,
                selected["threshold"],
            )
        )
        metrics = standard_metrics(
            source["pilot"],
            scores,
        )

        selected_source_results[
            source_name
        ] = {
            "raw_rm": source[
                "raw_metrics"
            ],
            "ungated_hybrid": source[
                "ungated_metrics"
            ],
            "fixed_gate": source[
                "fixed_metrics"
            ],
            "risk_gate": (
                add_delta(
                    metrics,
                    source["raw_metrics"],
                )
            ),
            "risk_decisions": (
                summarize_decisions(
                    decisions
                )
            ),
        }

    raw_macro = macro_metrics({
        name: source["raw_metrics"]
        for name, source in sources.items()
    })
    ungated_macro = macro_metrics({
        name: source["ungated_metrics"]
        for name, source in sources.items()
    })
    fixed_macro = macro_metrics({
        name: source["fixed_metrics"]
        for name, source in sources.items()
    })
    reliability_macro = {
        key: selected[key]
        for key in [
            "top1",
            "pair_macro_strict",
            "damage_rate",
            "correction_rate",
            "switch_rate",
        ]
    }

    result = {
        "version": (
            "reliability_aware_math_risk_gate_"
            "train_pilot_v2"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": {
            "development_domains": [
                "GSM8K",
                "MATH",
            ],
            "reward_models": len(
                MODEL_ROOTS
            ),
            "source_combinations": len(
                sources
            ),
            "train_labels_used": True,
            "pilot_labels_used": True,
            "test_data_loaded": False,
            "test_labels_loaded": False,
            "base_proposal": (
                "domain- and RM-specific "
                "ungated RM-Support"
            ),
            "training_target": (
                "harmful versus non-harmful "
                "Raw-to-Proposal switches"
            ),
            "positive_class": "harmful_switch",
            "neutral_switches_used_for_fit": (
                True
            ),
            "class_weight": None,
            "decision_policy": (
                "accept proposal unless estimated "
                "harm probability exceeds threshold"
            ),
            "accept_all_threshold": 1.0,
            "pilot_damage_limit": (
                cluster.PILOT_DAMAGE_LIMIT
            ),
        },
        "reliability_features": (
            FEATURE_NAMES
        ),
        "training": {
            "all_switches": len(
                train_y
            ),
            "harmful": int(np.sum(
                train_y == 1
            )),
            "nonharmful": int(np.sum(
                train_y == 0
            )),
            "beneficial": int(sum(
                record["train_switches"][
                    "beneficial_switches"
                ]
                for record in (
                    base_records.values()
                )
            )),
            "neutral": int(sum(
                record["train_switches"][
                    "neutral_switches"
                ]
                for record in (
                    base_records.values()
                )
            )),
        },
        "selected": selected,
        "reliability_model": (
            reliability_model
        ),
        "coefficient_by_feature": {
            name: float(weight)
            for name, weight in zip(
                FEATURE_NAMES,
                reliability_model[
                    "weights"
                ],
            )
        },
        "pilot_macro": {
            "raw_rm": raw_macro,
            "ungated_hybrid": (
                ungated_macro
            ),
            "fixed_gate": fixed_macro,
            "risk_gate": (
                reliability_macro
            ),
            "risk_vs_raw_top1": (
                reliability_macro["top1"]
                - raw_macro["top1"]
            ),
            "risk_vs_fixed_top1": (
                reliability_macro["top1"]
                - fixed_macro["top1"]
            ),
            "risk_vs_ungated_top1": (
                reliability_macro["top1"]
                - ungated_macro["top1"]
            ),
        },
        "base_configurations": (
            base_records
        ),
        "selected_source_results": (
            selected_source_results
        ),
        "selection_grid": grid,
        "elapsed_seconds": (
            time.time() - started
        ),
        "decision": (
            "freeze_math_trained_harm_risk_"
            "gate_before_any_test_evaluation"
        ),
    }

    atomic_json(OUTPUT_PATH, result)

    print()
    print("=" * 76)
    print("===== Harm-Risk Gate 选择 =====")
    print(json.dumps(
        {
            "training": result["training"],
            "selected": selected,
            "pilot_macro": result[
                "pilot_macro"
            ],
            "largest_coefficients": sorted(
                result[
                    "coefficient_by_feature"
                ].items(),
                key=lambda item: abs(
                    item[1]
                ),
                reverse=True,
            )[:8],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print(
        "RELIABILITY_MATH_RISK_GATE_V2_FREEZE_READY"
    )
    print("结果：", OUTPUT_PATH)
    print(
        "耗时秒：",
        round(
            result["elapsed_seconds"],
            3,
        ),
    )


if __name__ == "__main__":
    main()
