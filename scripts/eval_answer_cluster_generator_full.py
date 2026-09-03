from pathlib import Path
from collections import defaultdict
import json
import math
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_answer_cluster_evidence_smoke as cluster


DATA_ROOT = ROOT / "data/processed/prototype_v2"
RM_ROOT = (
    ROOT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
GEN_ROOT = (
    ROOT / "data/cache/generator_cluster_features_v1"
)
OUTPUT = (
    ROOT
    / "data/manifests/"
    "answer_cluster_generator_full_v1.json"
)

K_VALUES = [1, 2, 4, 8]

ABLATIONS = {
    "rm_support": list(range(0, 8)),
    "rm_support_nll": list(range(0, 11)),
    # prompt_hidden_agreement 不存在于单提示全量缓存，
    # 因此使用到 hidden_separation 为止。
    "rm_support_nll_hidden": list(range(0, 14)),
}

DOMAINS = {
    "GSM8K": {
        "model": "Qwen2-1.5B",
        "family": "gsm",
        "layer": "block_17",
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
        "model": "Qwen2-7B",
        "family": "math",
        "layer": "block_19",
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


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def candidate_key(item):
    return (
        str(item["question_uid"]),
        int(item.get("candidate_index", -1)),
    )


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )
    return values / np.maximum(norm, 1e-8)


def load_dataset(
    display_name,
    domain_spec,
    filename,
    prefix,
):
    rows = read_jsonl(DATA_ROOT / filename)
    rm_scores = np.asarray(
        np.load(
            RM_ROOT / f"{prefix}.scores_f32.npy"
        ),
        dtype=np.float32,
    )

    generator_dir = (
        GEN_ROOT / domain_spec["model"]
    )
    hidden = np.asarray(
        np.load(
            generator_dir
            / f"{prefix}.terminal_hidden_f16.npy",
            mmap_mode="r",
        ),
        dtype=np.float32,
    )
    hidden = unit_rows(hidden)

    nll = np.asarray(
        np.load(
            generator_dir
            / f"{prefix}.token_nll_f32.npy"
        ),
        dtype=np.float32,
    )
    generator_labels = np.asarray(
        np.load(
            generator_dir
            / f"{prefix}.labels_i8.npy"
        ),
        dtype=np.int8,
    )
    generator_metadata = read_jsonl(
        generator_dir / f"{prefix}.metadata.jsonl"
    )

    candidate_count = len(rows)
    expected_keys = [
        candidate_key(row) for row in rows
    ]
    actual_keys = [
        candidate_key(item)
        for item in generator_metadata
    ]

    if expected_keys != actual_keys:
        raise RuntimeError(
            f"{display_name}: 生成特征顺序不一致"
        )

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int8,
    )

    if not (
        len(rm_scores)
        == len(hidden)
        == len(nll)
        == len(generator_labels)
        == candidate_count
    ):
        raise RuntimeError(
            f"{display_name}: 缓存长度不一致"
        )

    if not np.array_equal(
        labels,
        generator_labels,
    ):
        raise RuntimeError(
            f"{display_name}: 标签不一致"
        )

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["question_uid"])].append(
            index
        )

    dataset = {
        "name": display_name,
        "family": domain_spec["family"],
        "rows": rows,
        "metadata": generator_metadata,
        "labels": labels,
        "rm_scores": rm_scores,
        "hidden": hidden,
        "nll": nll,
        # 单提示缓存没有跨提示一致性特征。
        "prompt_agreement": np.ones(
            candidate_count,
            dtype=np.float32,
        ),
        "groups": dict(groups),
    }

    cluster.build_questions(dataset)

    untrainable = sum(
        not question["cluster_trainable"]
        for question in dataset["questions"]
    )

    print(
        f"{display_name}: "
        f"问题={len(dataset['questions'])}, "
        f"候选={candidate_count}, "
        f"混合簇={dataset['mixed_clusters']}, "
        f"不可训练问题={untrainable}",
        flush=True,
    )

    # 后续只需要已经聚合好的低维簇特征。
    dataset.pop("hidden")
    dataset.pop("nll")
    dataset.pop("prompt_agreement")

    return dataset


def exact_logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return float(
        maximum
        + math.log(
            float(np.exp(values - maximum).sum())
        )
    )


def predict_rule(dataset, rule):
    predictions = np.empty(
        len(dataset["labels"]),
        dtype=np.float32,
    )

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
                + 1e-4
                * cluster.ordinary_z(rm_max)
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

        local_scores = (
            cluster.candidate_scores_from_clusters(
                question,
                cluster_scores,
                threshold=0.0,
            )
        )
        predictions[
            question["indices"]
        ] = local_scores

    return predictions


def predict_learned(
    dataset,
    feature_indices,
    models,
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
            feature_indices,
            models,
        )
        learned = cluster.ordinary_z(learned)

        base = np.asarray([
            item["rm_max"]
            for item in question["clusters"]
        ], dtype=np.float32)
        base = cluster.ordinary_z(base)

        hybrid = base + beta * learned

        local_scores = (
            cluster.candidate_scores_from_clusters(
                question,
                hybrid,
                threshold,
            )
        )
        predictions[
            question["indices"]
        ] = local_scores

    return predictions


def standard_metrics(dataset, scores):
    question_ids = [
        str(row["question_uid"])
        for row in dataset["rows"]
    ]

    return cluster.ranking_metrics(
        dataset["labels"],
        scores,
        question_ids,
        dataset["rm_scores"],
    )


def add_delta(metrics, raw):
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


def pass_probability(labels, k):
    n = len(labels)
    correct = int(np.sum(labels == 1))

    total = math.comb(n, k)
    if n - correct < k:
        return 1.0

    return 1.0 - (
        math.comb(n - correct, k) / total
    )


def selector_probability(labels, scores, k):
    n = len(labels)
    denominator = math.comb(n, k)

    # mergesort 保证分数相同时采用稳定顺序。
    order = np.argsort(
        -scores,
        kind="mergesort",
    )
    sorted_labels = labels[order]

    numerator = 0

    for rank, label in enumerate(sorted_labels):
        if label != 1:
            continue

        candidates_below = n - rank - 1

        if candidates_below >= k - 1:
            numerator += math.comb(
                candidates_below,
                k - 1,
            )

    return numerator / denominator


def budget_metrics(dataset, scores):
    results = {}
    questions = dataset["questions"]

    for k in K_VALUES:
        eligible = [
            question
            for question in questions
            if len(question["labels"]) >= k
        ]

        pass_values = []
        selector_values = []

        for question in eligible:
            indices = question["indices"]
            labels = dataset["labels"][indices]
            question_scores = scores[indices]

            pass_values.append(
                pass_probability(labels, k)
            )
            selector_values.append(
                selector_probability(
                    labels,
                    question_scores,
                    k,
                )
            )

        pass_at_k = float(np.mean(pass_values))
        selection_at_k = float(
            np.mean(selector_values)
        )

        results[f"k{k}"] = {
            "eligible_questions": len(eligible),
            "total_questions": len(questions),
            "coverage": (
                len(eligible) / len(questions)
            ),
            "pass_at_k": pass_at_k,
            "selection_at_k": selection_at_k,
            "oracle_gap": (
                pass_at_k - selection_at_k
            ),
        }

    return results


def evaluate_scores(dataset, scores, raw_metrics):
    metrics = standard_metrics(dataset, scores)
    return {
        "ranking": add_delta(
            metrics,
            raw_metrics,
        ),
        "budget": budget_metrics(
            dataset,
            scores,
        ),
    }


def run_domain(domain_name, spec):
    print()
    print("=" * 78)
    print(
        f"{domain_name}: "
        f"{spec['model']} / {spec['layer']}"
    )

    train = load_dataset(
        f"{domain_name}_TRAIN",
        spec,
        *spec["train"],
    )
    pilot = load_dataset(
        f"{domain_name}_PILOT",
        spec,
        *spec["pilot"],
    )
    tests = {
        name: load_dataset(
            name,
            spec,
            *split_spec,
        )
        for name, split_spec
        in spec["tests"].items()
    }

    all_sets = {
        "PILOT": pilot,
        **tests,
    }

    raw_metrics = {
        name: standard_metrics(
            dataset,
            dataset["rm_scores"],
        )
        for name, dataset in all_sets.items()
    }

    baseline_predictions = {
        name: {
            "raw_rm": dataset["rm_scores"],
            "majority": predict_rule(
                dataset,
                "majority",
            ),
            "weighted_tau4": predict_rule(
                dataset,
                "weighted_tau4",
            ),
        }
        for name, dataset in all_sets.items()
    }

    result = {
        "model": spec["model"],
        "layer": spec["layer"],
        "feature_names": cluster.FEATURE_NAMES,
        "baselines": {},
        "ablations": {},
    }

    for name, dataset in all_sets.items():
        result["baselines"][name] = {}

        for method, scores in (
            baseline_predictions[name].items()
        ):
            result["baselines"][name][method] = (
                evaluate_scores(
                    dataset,
                    scores,
                    raw_metrics[name],
                )
            )

    for ablation_name, feature_indices in (
        ABLATIONS.items()
    ):
        print()
        print("消融：", ablation_name)

        selected, models, grid = (
            cluster.choose_configuration(
                train,
                pilot,
                feature_indices,
            )
        )

        print(
            "  Pilot选择："
            f"reg={selected['regularization']}, "
            f"beta={selected['beta']}, "
            f"threshold={selected['threshold']}, "
            f"Top1={selected['top1']:.6f}, "
            f"Pair={selected['pair_macro_strict']:.6f}, "
            f"Damage={selected['damage_rate']:.6f}"
        )

        evaluations = {}

        for name, dataset in all_sets.items():
            scores = predict_learned(
                dataset,
                feature_indices,
                models,
                selected["beta"],
                selected["threshold"],
            )

            evaluations[name] = evaluate_scores(
                dataset,
                scores,
                raw_metrics[name],
            )

            ranking = evaluations[name]["ranking"]
            k4 = evaluations[name]["budget"]["k4"]
            k8 = evaluations[name]["budget"]["k8"]

            print(
                f"  {name}: "
                f"Top1={ranking['raw_top1']:.6f}"
                f"->{ranking['top1']:.6f} "
                f"({ranking['top1_delta']:+.6f}), "
                f"Pair="
                f"{ranking['raw_pair_macro_strict']:.6f}"
                f"->{ranking['pair_macro_strict']:.6f} "
                f"({ranking['pair_delta']:+.6f}), "
                f"Damage={ranking['damage_rate']:.6f}"
            )
            print(
                f"    Best@4={k4['selection_at_k']:.6f}, "
                f"Pass@4={k4['pass_at_k']:.6f}, "
                f"Best@8={k8['selection_at_k']:.6f}, "
                f"Pass@8={k8['pass_at_k']:.6f}, "
                f"K8覆盖={k8['coverage']:.6f}"
            )

        result["ablations"][ablation_name] = {
            "feature_indices": feature_indices,
            "features": [
                cluster.FEATURE_NAMES[index]
                for index in feature_indices
            ],
            "selected": selected,
            "evaluations": evaluations,
            "grid": grid,
        }

    return result


def test_macro(results):
    test_entries = []

    for domain_result in results.values():
        test_names = [
            name
            for name in domain_result[
                "baselines"
            ]
            if name != "PILOT"
        ]

        for name in test_names:
            entry = {
                "raw_rm": domain_result[
                    "baselines"
                ][name]["raw_rm"],
                "weighted_tau4": domain_result[
                    "baselines"
                ][name]["weighted_tau4"],
            }

            for ablation_name, ablation in (
                domain_result["ablations"].items()
            ):
                entry[ablation_name] = (
                    ablation["evaluations"][name]
                )

            test_entries.append(entry)

    methods = [
        "raw_rm",
        "weighted_tau4",
        *ABLATIONS.keys(),
    ]
    summary = {}

    for method in methods:
        rankings = [
            entry[method]["ranking"]
            for entry in test_entries
        ]

        item = {
            "macro_top1": float(np.mean([
                value["top1"]
                for value in rankings
            ])),
            "macro_pair_macro_strict": float(
                np.mean([
                    value["pair_macro_strict"]
                    for value in rankings
                ])
            ),
            "macro_damage_rate": float(
                np.mean([
                    value["damage_rate"]
                    for value in rankings
                ])
            ),
            "budget": {},
        }

        for k in K_VALUES:
            key = f"k{k}"
            budget_rows = [
                entry[method]["budget"][key]
                for entry in test_entries
            ]

            item["budget"][key] = {
                "macro_pass_at_k": float(
                    np.mean([
                        value["pass_at_k"]
                        for value in budget_rows
                    ])
                ),
                "macro_selection_at_k": float(
                    np.mean([
                        value["selection_at_k"]
                        for value in budget_rows
                    ])
                ),
                "macro_oracle_gap": float(
                    np.mean([
                        value["oracle_gap"]
                        for value in budget_rows
                    ])
                ),
                "mean_coverage": float(
                    np.mean([
                        value["coverage"]
                        for value in budget_rows
                    ])
                ),
            }

        summary[method] = item

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

        for k in K_VALUES:
            key = f"k{k}"
            item["budget"][key][
                "selection_delta_vs_raw"
            ] = (
                item["budget"][key][
                    "macro_selection_at_k"
                ]
                - raw["budget"][key][
                    "macro_selection_at_k"
                ]
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

    macro = test_macro(results)

    output = {
        "version": (
            "answer_cluster_generator_full_v1"
        ),
        "scope": (
            "full train/pilot and current full "
            "ID/OOD evaluations"
        ),
        "budget_metric_definition": {
            "pass_at_k": (
                "exact probability that a uniformly "
                "sampled size-k subset contains at "
                "least one correct candidate"
            ),
            "selection_at_k": (
                "exact expected correctness of the "
                "highest-scored candidate in a "
                "uniformly sampled size-k subset"
            ),
            "insufficient_candidates": (
                "excluded for that k and reported "
                "through coverage"
            ),
        },
        "k_values": K_VALUES,
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
    print("=" * 78)
    print("===== 全量三数据集宏平均 =====")

    for method, item in macro.items():
        k4 = item["budget"]["k4"]
        k8 = item["budget"]["k8"]

        print(
            f"{method}: "
            f"Top1={item['macro_top1']:.6f} "
            f"({item['macro_top1_delta']:+.6f}), "
            f"Pair="
            f"{item['macro_pair_macro_strict']:.6f} "
            f"({item['macro_pair_delta']:+.6f}), "
            f"Damage={item['macro_damage_rate']:.6f}"
        )
        print(
            f"  Best@4="
            f"{k4['macro_selection_at_k']:.6f} "
            f"({k4['selection_delta_vs_raw']:+.6f}), "
            f"Pass@4={k4['macro_pass_at_k']:.6f}; "
            f"Best@8="
            f"{k8['macro_selection_at_k']:.6f} "
            f"({k8['selection_delta_vs_raw']:+.6f}), "
            f"Pass@8={k8['macro_pass_at_k']:.6f}, "
            f"K8覆盖={k8['mean_coverage']:.6f}"
        )

    print()
    print("结果：", OUTPUT)
    print("耗时秒：", output["elapsed_seconds"])


if __name__ == "__main__":
    main()
