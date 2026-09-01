from pathlib import Path
import json
import sys
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import explore_answer_cluster_evidence_smoke as cluster
import eval_answer_cluster_generator_full as evaluation


OUTPUT = (
    ROOT
    / "data/manifests/"
    "answer_cluster_rm_support_bootstrap_v1.json"
)

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260901
K_VALUES = [4, 8]
FEATURE_INDICES = list(range(8))


def pair_metric(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return float(np.mean(
        positive[:, None] > negative[None, :]
    ))


def build_question_records(
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
        method_choice = int(np.argmax(method))

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
            "raw_top1": raw_correct,
            "method_top1": method_correct,
            "pair_delta": (
                pair_metric(labels, method)
                - pair_metric(labels, raw)
            ),
            "raw_pair": pair_metric(
                labels,
                raw,
            ),
            "method_pair": pair_metric(
                labels,
                method,
            ),
            "raw_correct": raw_correct,
            "damage": int(
                raw_correct == 1
                and method_correct == 0
            ),
            "raw_wrong": int(
                raw_correct == 0
            ),
            "correction": int(
                raw_correct == 0
                and method_correct == 1
            ),
        }

        for k in K_VALUES:
            if len(labels) >= k:
                record[f"pass_{k}"] = (
                    evaluation.pass_probability(
                        labels,
                        k,
                    )
                )
                record[f"raw_best_{k}"] = (
                    evaluation.selector_probability(
                        labels,
                        raw,
                        k,
                    )
                )
                record[f"method_best_{k}"] = (
                    evaluation.selector_probability(
                        labels,
                        method,
                        k,
                    )
                )
                record[f"best_delta_{k}"] = (
                    record[f"method_best_{k}"]
                    - record[f"raw_best_{k}"]
                )
            else:
                record[f"pass_{k}"] = np.nan
                record[f"raw_best_{k}"] = np.nan
                record[f"method_best_{k}"] = np.nan
                record[f"best_delta_{k}"] = np.nan

        records.append(record)

    return records


def records_to_arrays(records):
    keys = list(records[0])
    return {
        key: np.asarray(
            [record[key] for record in records],
            dtype=np.float64,
        )
        for key in keys
    }


def point_metrics(arrays):
    raw_correct = np.sum(
        arrays["raw_correct"]
    )
    raw_wrong = np.sum(
        arrays["raw_wrong"]
    )

    result = {
        "questions": len(
            arrays["top1_delta"]
        ),
        "raw_top1": float(np.mean(
            arrays["raw_top1"]
        )),
        "method_top1": float(np.mean(
            arrays["method_top1"]
        )),
        "top1_delta": float(np.mean(
            arrays["top1_delta"]
        )),
        "raw_pair": float(np.mean(
            arrays["raw_pair"]
        )),
        "method_pair": float(np.mean(
            arrays["method_pair"]
        )),
        "pair_delta": float(np.mean(
            arrays["pair_delta"]
        )),
        "damage_rate": float(
            np.sum(arrays["damage"])
            / max(raw_correct, 1)
        ),
        "correction_rate": float(
            np.sum(arrays["correction"])
            / max(raw_wrong, 1)
        ),
        "budget": {},
    }

    for k in K_VALUES:
        valid = ~np.isnan(
            arrays[f"pass_{k}"]
        )

        result["budget"][f"k{k}"] = {
            "eligible_questions": int(
                np.sum(valid)
            ),
            "coverage": float(
                np.mean(valid)
            ),
            "pass_at_k": float(np.mean(
                arrays[f"pass_{k}"][valid]
            )),
            "raw_best_at_k": float(np.mean(
                arrays[f"raw_best_{k}"][valid]
            )),
            "method_best_at_k": float(
                np.mean(
                    arrays[
                        f"method_best_{k}"
                    ][valid]
                )
            ),
            "best_at_k_delta": float(
                np.mean(
                    arrays[
                        f"best_delta_{k}"
                    ][valid]
                )
            ),
        }

    return result


def bootstrap_metrics(
    arrays,
    samples,
    rng,
):
    question_count = len(
        arrays["top1_delta"]
    )

    sampled_indices = rng.integers(
        0,
        question_count,
        size=(samples, question_count),
    )

    raw_correct = np.sum(
        arrays["raw_correct"][
            sampled_indices
        ],
        axis=1,
    )
    raw_wrong = np.sum(
        arrays["raw_wrong"][
            sampled_indices
        ],
        axis=1,
    )

    result = {
        "top1_delta": np.mean(
            arrays["top1_delta"][
                sampled_indices
            ],
            axis=1,
        ),
        "pair_delta": np.mean(
            arrays["pair_delta"][
                sampled_indices
            ],
            axis=1,
        ),
        "damage_rate": (
            np.sum(
                arrays["damage"][
                    sampled_indices
                ],
                axis=1,
            )
            / np.maximum(raw_correct, 1)
        ),
        "correction_rate": (
            np.sum(
                arrays["correction"][
                    sampled_indices
                ],
                axis=1,
            )
            / np.maximum(raw_wrong, 1)
        ),
    }

    for k in K_VALUES:
        values = arrays[
            f"best_delta_{k}"
        ][sampled_indices]
        pass_values = arrays[
            f"pass_{k}"
        ][sampled_indices]

        result[f"best_delta_{k}"] = (
            np.nanmean(values, axis=1)
        )
        result[f"pass_{k}"] = (
            np.nanmean(
                pass_values,
                axis=1,
            )
        )

    return result


def interval(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def summarize_bootstrap(
    point,
    bootstrap,
):
    summary = {
        "top1_delta": {
            "point": point["top1_delta"],
            "ci95": interval(
                bootstrap["top1_delta"]
            ),
        },
        "pair_delta": {
            "point": point["pair_delta"],
            "ci95": interval(
                bootstrap["pair_delta"]
            ),
        },
        "damage_rate": {
            "point": point["damage_rate"],
            "ci95": interval(
                bootstrap["damage_rate"]
            ),
        },
        "correction_rate": {
            "point": point[
                "correction_rate"
            ],
            "ci95": interval(
                bootstrap[
                    "correction_rate"
                ]
            ),
        },
        "budget": {},
    }

    for k in K_VALUES:
        summary["budget"][f"k{k}"] = {
            "best_at_k_delta": {
                "point": point["budget"][
                    f"k{k}"
                ]["best_at_k_delta"],
                "ci95": interval(
                    bootstrap[
                        f"best_delta_{k}"
                    ]
                ),
            },
            "pass_at_k": {
                "point": point["budget"][
                    f"k{k}"
                ]["pass_at_k"],
                "ci95": interval(
                    bootstrap[f"pass_{k}"]
                ),
            },
            "coverage": point["budget"][
                f"k{k}"
            ]["coverage"],
        }

    return summary


def run_domain(domain_name, spec):
    print()
    print("=" * 76)
    print(domain_name)

    train = evaluation.load_dataset(
        f"{domain_name}_TRAIN",
        spec,
        *spec["train"],
    )
    pilot = evaluation.load_dataset(
        f"{domain_name}_PILOT",
        spec,
        *spec["pilot"],
    )
    tests = {
        name: evaluation.load_dataset(
            name,
            spec,
            *split_spec,
        )
        for name, split_spec
        in spec["tests"].items()
    }

    selected, models, _ = (
        cluster.choose_configuration(
            train,
            pilot,
            FEATURE_INDICES,
        )
    )

    print(
        "选择："
        f"reg={selected['regularization']}, "
        f"beta={selected['beta']}, "
        f"threshold={selected['threshold']}"
    )

    result = {
        "selected": {
            key: selected[key]
            for key in [
                "regularization",
                "beta",
                "threshold",
                "top1",
                "pair_macro_strict",
                "damage_rate",
            ]
        },
        "tests": {},
    }

    raw_arrays = {}

    for name, dataset in tests.items():
        method_scores = evaluation.predict_learned(
            dataset,
            FEATURE_INDICES,
            models,
            selected["beta"],
            selected["threshold"],
        )

        records = build_question_records(
            dataset,
            dataset["rm_scores"],
            method_scores,
        )
        arrays = records_to_arrays(records)
        raw_arrays[name] = arrays

        point = point_metrics(arrays)

        print(
            f"{name}: "
            f"ΔTop1={point['top1_delta']:+.6f}, "
            f"ΔPair={point['pair_delta']:+.6f}, "
            f"ΔBest@4="
            f"{point['budget']['k4']['best_at_k_delta']:+.6f}, "
            f"ΔBest@8="
            f"{point['budget']['k8']['best_at_k_delta']:+.6f}"
        )

        result["tests"][name] = {
            "point": point,
        }

    return result, raw_arrays


def main():
    started = time.time()
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    results = {}
    all_arrays = {}

    for domain_name, spec in (
        evaluation.DOMAINS.items()
    ):
        result, arrays = run_domain(
            domain_name,
            spec,
        )
        results[domain_name] = result
        all_arrays.update(arrays)

    dataset_bootstrap = {}
    dataset_points = {}

    for name, arrays in all_arrays.items():
        point = point_metrics(arrays)
        boot = bootstrap_metrics(
            arrays,
            BOOTSTRAP_SAMPLES,
            rng,
        )

        dataset_points[name] = point
        dataset_bootstrap[name] = boot

        summary = summarize_bootstrap(
            point,
            boot,
        )

        for domain_result in results.values():
            if name in domain_result["tests"]:
                domain_result["tests"][name][
                    "bootstrap"
                ] = summary

    macro_point = {
        "top1_delta": float(np.mean([
            value["top1_delta"]
            for value in dataset_points.values()
        ])),
        "pair_delta": float(np.mean([
            value["pair_delta"]
            for value in dataset_points.values()
        ])),
        "damage_rate": float(np.mean([
            value["damage_rate"]
            for value in dataset_points.values()
        ])),
        "correction_rate": float(np.mean([
            value["correction_rate"]
            for value in dataset_points.values()
        ])),
        "budget": {},
    }

    macro_bootstrap = {
        key: np.mean(
            np.stack([
                value[key]
                for value
                in dataset_bootstrap.values()
            ]),
            axis=0,
        )
        for key in [
            "top1_delta",
            "pair_delta",
            "damage_rate",
            "correction_rate",
            "best_delta_4",
            "best_delta_8",
            "pass_4",
            "pass_8",
        ]
    }

    for k in K_VALUES:
        macro_point["budget"][f"k{k}"] = {
            "best_at_k_delta": float(
                np.mean([
                    value["budget"][f"k{k}"][
                        "best_at_k_delta"
                    ]
                    for value
                    in dataset_points.values()
                ])
            ),
            "pass_at_k": float(
                np.mean([
                    value["budget"][f"k{k}"][
                        "pass_at_k"
                    ]
                    for value
                    in dataset_points.values()
                ])
            ),
            "coverage": float(
                np.mean([
                    value["budget"][f"k{k}"][
                        "coverage"
                    ]
                    for value
                    in dataset_points.values()
                ])
            ),
        }

    macro_summary = summarize_bootstrap(
        macro_point,
        macro_bootstrap,
    )

    raw_best_4 = float(np.mean([
        value["budget"]["k4"][
            "raw_best_at_k"
        ]
        for value in dataset_points.values()
    ]))
    new_best_4 = raw_best_4 + (
        macro_point["budget"]["k4"][
            "best_at_k_delta"
        ]
    )
    pass_4 = macro_point["budget"]["k4"][
        "pass_at_k"
    ]

    raw_best_8 = float(np.mean([
        value["budget"]["k8"][
            "raw_best_at_k"
        ]
        for value in dataset_points.values()
    ]))
    new_best_8 = raw_best_8 + (
        macro_point["budget"]["k8"][
            "best_at_k_delta"
        ]
    )
    pass_8 = macro_point["budget"]["k8"][
        "pass_at_k"
    ]

    oracle_gap_closed = {
        "k4": (
            (new_best_4 - raw_best_4)
            / max(pass_4 - raw_best_4, 1e-12)
        ),
        "k8": (
            (new_best_8 - raw_best_8)
            / max(pass_8 - raw_best_8, 1e-12)
        ),
    }

    output = {
        "version": (
            "answer_cluster_rm_support_bootstrap_v1"
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_resampling_unit": "question",
        "results": results,
        "macro": macro_summary,
        "oracle_gap_closed": oracle_gap_closed,
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
    print("===== 配对 Bootstrap 宏平均 =====")

    for key in [
        "top1_delta",
        "pair_delta",
        "damage_rate",
    ]:
        item = macro_summary[key]
        print(
            f"{key}: "
            f"{item['point']:+.6f}, "
            f"95%CI=[{item['ci95'][0]:+.6f}, "
            f"{item['ci95'][1]:+.6f}]"
        )

    for k in K_VALUES:
        item = macro_summary["budget"][
            f"k{k}"
        ]["best_at_k_delta"]
        print(
            f"Best@{k} delta: "
            f"{item['point']:+.6f}, "
            f"95%CI=[{item['ci95'][0]:+.6f}, "
            f"{item['ci95'][1]:+.6f}]"
        )

    print(
        "Oracle Gap Closed: "
        f"K4={oracle_gap_closed['k4']:.3%}, "
        f"K8={oracle_gap_closed['k8']:.3%}"
    )
    print("结果：", OUTPUT)
    print("耗时秒：", output["elapsed_seconds"])


if __name__ == "__main__":
    main()
