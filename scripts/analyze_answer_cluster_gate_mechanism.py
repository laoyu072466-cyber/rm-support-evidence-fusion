from pathlib import Path
from collections import Counter
import json
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answer_cluster_generator_full as evaluation
import explore_answer_cluster_evidence_smoke as cluster


OUTPUT = (
    ROOT / "data/manifests/"
    "answer_cluster_gate_mechanism_v1.json"
)

FEATURE_INDICES = list(range(8))

PARAMETERS = {
    "GSM8K": {
        "regularization": 0.1,
        "beta": 2.0,
        "threshold": 0.25,
    },
    "MATH": {
        "regularization": 0.001,
        "beta": 2.0,
        "threshold": 0.25,
    },
}


def describe(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def representative(question, cluster_index):
    members = np.asarray(
        question["clusters"][
            cluster_index
        ]["members"],
        dtype=np.int64,
    )
    return int(
        members[
            np.argmax(
                question["rm"][members]
            )
        ]
    )


def support(question, cluster_index):
    return len(
        question["clusters"][
            cluster_index
        ]["members"]
    )


def majority_cluster(question):
    return max(
        range(len(question["clusters"])),
        key=lambda index: (
            support(question, index),
            float(np.max(
                question["rm"][
                    question["clusters"][
                        index
                    ]["members"]
                ]
            )),
        ),
    )


def correctness(
    question,
    dataset,
    cluster_index,
):
    local = representative(
        question,
        cluster_index,
    )
    global_index = int(
        question["indices"][local]
    )
    return bool(
        dataset["labels"][global_index] == 1
    )


def example_record(
    dataset,
    question,
    values,
):
    global_index = int(
        question["indices"][0]
    )
    row = dataset["rows"][global_index]

    return {
        "question_uid": str(
            row["question_uid"]
        ),
        "problem_id": row.get("problem_id"),
        **values,
    }


def analyze_dataset(
    dataset,
    models,
    beta,
    threshold,
):
    records = []
    examples = {
        "majority_damages_raw": [],
        "majority_wrong_despite_correct_pool": [],
        "gate_prevented_damage": [],
        "gate_blocked_correction": [],
        "gate_accepted_correction": [],
        "gate_accepted_damage": [],
    }

    for question in dataset["questions"]:
        learned = cluster.model_cluster_scores(
            question,
            FEATURE_INDICES,
            models,
        )
        learned = cluster.ordinary_z(
            learned
        )

        base_scores = np.asarray([
            item["rm_max"]
            for item in question["clusters"]
        ], dtype=np.float32)
        base_scores = cluster.ordinary_z(
            base_scores
        )

        hybrid = (
            base_scores
            + beta * learned
        )

        raw_cluster = int(
            question["raw_cluster"]
        )
        proposal_cluster = int(
            np.argmax(hybrid)
        )
        majority = majority_cluster(
            question
        )

        if proposal_cluster == raw_cluster:
            advantage = 0.0
            accepted = False
            gated_cluster = raw_cluster
        else:
            advantage = float(
                hybrid[proposal_cluster]
                - hybrid[raw_cluster]
            )
            accepted = (
                advantage > threshold
            )
            gated_cluster = (
                proposal_cluster
                if accepted
                else raw_cluster
            )

        raw_ok = correctness(
            question,
            dataset,
            raw_cluster,
        )
        majority_ok = correctness(
            question,
            dataset,
            majority,
        )
        proposal_ok = correctness(
            question,
            dataset,
            proposal_cluster,
        )
        gated_ok = correctness(
            question,
            dataset,
            gated_cluster,
        )

        local_labels = dataset["labels"][
            question["indices"]
        ]
        oracle_has_correct = bool(
            np.any(local_labels == 1)
        )

        correct_cluster_supports = []

        for index in range(
            len(question["clusters"])
        ):
            if correctness(
                question,
                dataset,
                index,
            ):
                correct_cluster_supports.append(
                    support(question, index)
                )

        max_correct_support = (
            max(correct_cluster_supports)
            if correct_cluster_supports
            else 0
        )

        values = {
            "raw_ok": raw_ok,
            "majority_ok": majority_ok,
            "proposal_ok": proposal_ok,
            "gated_ok": gated_ok,
            "raw_cluster": raw_cluster,
            "majority_cluster": majority,
            "proposal_cluster": (
                proposal_cluster
            ),
            "gated_cluster": gated_cluster,
            "proposal_changed": (
                proposal_cluster
                != raw_cluster
            ),
            "gate_accepted": accepted,
            "advantage": advantage,
            "raw_support": support(
                question,
                raw_cluster,
            ),
            "majority_support": support(
                question,
                majority,
            ),
            "proposal_support": support(
                question,
                proposal_cluster,
            ),
            "max_correct_support": (
                max_correct_support
            ),
            "oracle_has_correct": (
                oracle_has_correct
            ),
        }
        records.append(values)

        detail = example_record(
            dataset,
            question,
            {
                key: value
                for key, value
                in values.items()
                if key not in {
                    "raw_cluster",
                    "majority_cluster",
                    "proposal_cluster",
                    "gated_cluster",
                }
            },
        )

        if (
            raw_ok
            and not majority_ok
            and len(
                examples[
                    "majority_damages_raw"
                ]
            ) < 5
        ):
            examples[
                "majority_damages_raw"
            ].append(detail)

        if (
            not majority_ok
            and oracle_has_correct
            and len(
                examples[
                    "majority_wrong_despite_correct_pool"
                ]
            ) < 5
        ):
            examples[
                "majority_wrong_despite_correct_pool"
            ].append(detail)

        if (
            values["proposal_changed"]
            and not accepted
            and raw_ok
            and not proposal_ok
            and len(
                examples[
                    "gate_prevented_damage"
                ]
            ) < 5
        ):
            examples[
                "gate_prevented_damage"
            ].append(detail)

        if (
            values["proposal_changed"]
            and not accepted
            and not raw_ok
            and proposal_ok
            and len(
                examples[
                    "gate_blocked_correction"
                ]
            ) < 5
        ):
            examples[
                "gate_blocked_correction"
            ].append(detail)

        if (
            accepted
            and not raw_ok
            and proposal_ok
            and len(
                examples[
                    "gate_accepted_correction"
                ]
            ) < 5
        ):
            examples[
                "gate_accepted_correction"
            ].append(detail)

        if (
            accepted
            and raw_ok
            and not proposal_ok
            and len(
                examples[
                    "gate_accepted_damage"
                ]
            ) < 5
        ):
            examples[
                "gate_accepted_damage"
            ].append(detail)

    questions = len(records)

    def count(predicate):
        return sum(
            bool(predicate(item))
            for item in records
        )

    raw_correct = count(
        lambda item: item["raw_ok"]
    )
    majority_correct = count(
        lambda item: item["majority_ok"]
    )
    ungated_correct = count(
        lambda item: item["proposal_ok"]
    )
    gated_correct = count(
        lambda item: item["gated_ok"]
    )

    changed = [
        item for item in records
        if item["proposal_changed"]
    ]
    accepted = [
        item for item in changed
        if item["gate_accepted"]
    ]
    rejected = [
        item for item in changed
        if not item["gate_accepted"]
    ]

    majority_wrong_with_correct_pool = count(
        lambda item: (
            not item["majority_ok"]
            and item["oracle_has_correct"]
        )
    )
    majority_wrong_without_correct_pool = count(
        lambda item: (
            not item["majority_ok"]
            and not item["oracle_has_correct"]
        )
    )
    wrong_plurality_over_correct = count(
        lambda item: (
            not item["majority_ok"]
            and item["oracle_has_correct"]
            and item["majority_support"]
            > item["max_correct_support"]
        )
    )
    wrong_support_tie = count(
        lambda item: (
            not item["majority_ok"]
            and item["oracle_has_correct"]
            and item["majority_support"]
            == item["max_correct_support"]
        )
    )

    accepted_corrections = sum(
        (not item["raw_ok"])
        and item["proposal_ok"]
        for item in accepted
    )
    accepted_damages = sum(
        item["raw_ok"]
        and not item["proposal_ok"]
        for item in accepted
    )
    accepted_neutral = (
        len(accepted)
        - accepted_corrections
        - accepted_damages
    )

    prevented_damages = sum(
        item["raw_ok"]
        and not item["proposal_ok"]
        for item in rejected
    )
    blocked_corrections = sum(
        (not item["raw_ok"])
        and item["proposal_ok"]
        for item in rejected
    )
    rejected_neutral = (
        len(rejected)
        - prevented_damages
        - blocked_corrections
    )

    majority_damages = count(
        lambda item: (
            item["raw_ok"]
            and not item["majority_ok"]
        )
    )
    majority_corrections = count(
        lambda item: (
            not item["raw_ok"]
            and item["majority_ok"]
        )
    )

    gate_gain_vs_ungated = (
        gated_correct - ungated_correct
    )
    expected_gate_gain = (
        prevented_damages
        - blocked_corrections
    )

    if gate_gain_vs_ungated != expected_gate_gain:
        raise RuntimeError(
            "Gate counterfactual accounting failed"
        )

    summary = {
        "questions": questions,
        "top1": {
            "raw_rm": raw_correct / questions,
            "majority": (
                majority_correct / questions
            ),
            "ungated_hybrid": (
                ungated_correct / questions
            ),
            "frozen_gate": (
                gated_correct / questions
            ),
        },
        "delta_vs_raw": {
            "majority": (
                majority_correct
                - raw_correct
            ) / questions,
            "ungated_hybrid": (
                ungated_correct
                - raw_correct
            ) / questions,
            "frozen_gate": (
                gated_correct
                - raw_correct
            ) / questions,
        },
        "majority_failure": {
            "corrections_of_raw": (
                majority_corrections
            ),
            "damages_of_raw": majority_damages,
            "net_corrected": (
                majority_corrections
                - majority_damages
            ),
            "wrong_with_correct_candidate_pool": (
                majority_wrong_with_correct_pool
            ),
            "wrong_without_correct_candidate": (
                majority_wrong_without_correct_pool
            ),
            "wrong_plurality_over_correct_cluster": (
                wrong_plurality_over_correct
            ),
            "wrong_support_tie_with_correct_cluster": (
                wrong_support_tie
            ),
        },
        "gate": {
            "proposal_switches": len(changed),
            "proposal_switch_rate": (
                len(changed) / questions
            ),
            "accepted_switches": len(accepted),
            "accepted_rate_of_proposals": (
                len(accepted)
                / max(len(changed), 1)
            ),
            "rejected_switches": len(rejected),
            "accepted_corrections": (
                accepted_corrections
            ),
            "accepted_damages": (
                accepted_damages
            ),
            "accepted_neutral": (
                accepted_neutral
            ),
            "prevented_damages": (
                prevented_damages
            ),
            "blocked_corrections": (
                blocked_corrections
            ),
            "rejected_neutral": (
                rejected_neutral
            ),
            "net_questions_saved_by_gate": (
                gate_gain_vs_ungated
            ),
            "top1_gain_vs_ungated": (
                gate_gain_vs_ungated
                / questions
            ),
            "accepted_advantage": describe([
                item["advantage"]
                for item in accepted
            ]),
            "rejected_advantage": describe([
                item["advantage"]
                for item in rejected
            ]),
        },
        "examples": examples,
    }

    return summary


def main():
    started = time.time()

    print(
        "===== Majority 失败与 Gate "
        "机制审计 ====="
    )
    print(
        "配置使用冻结参数；"
        "测试标签仅用于事后评价。"
    )

    output = {
        "version": (
            "answer_cluster_gate_mechanism_v1"
        ),
        "feature_indices": FEATURE_INDICES,
        "parameters": PARAMETERS,
        "datasets": {},
    }

    macro_values = {
        "raw_rm": [],
        "majority": [],
        "ungated_hybrid": [],
        "frozen_gate": [],
    }
    gate_totals = Counter()

    for domain_name, domain_spec in (
        evaluation.DOMAINS.items()
    ):
        print()
        print("=" * 76)
        print(domain_name)

        parameters = PARAMETERS[domain_name]

        train = evaluation.load_dataset(
            f"{domain_name}_TRAIN_GATE_AUDIT",
            domain_spec,
            *domain_spec["train"],
        )

        models = [
            cluster.fit_cluster_model(
                train,
                FEATURE_INDICES,
                parameters["regularization"],
                seed,
            )
            for seed in cluster.SEEDS
        ]

        for test_name, test_spec in (
            domain_spec["tests"].items()
        ):
            dataset = evaluation.load_dataset(
                test_name,
                domain_spec,
                *test_spec,
            )

            summary = analyze_dataset(
                dataset,
                models,
                parameters["beta"],
                parameters["threshold"],
            )
            output["datasets"][test_name] = (
                summary
            )

            for method, value in (
                summary["top1"].items()
            ):
                macro_values[method].append(
                    value
                )

            for key in [
                "proposal_switches",
                "accepted_switches",
                "rejected_switches",
                "accepted_corrections",
                "accepted_damages",
                "prevented_damages",
                "blocked_corrections",
                "net_questions_saved_by_gate",
            ]:
                gate_totals[key] += (
                    summary["gate"][key]
                )

            print()
            print(test_name)
            print(json.dumps(
                {
                    "top1": summary["top1"],
                    "majority_failure": (
                        summary[
                            "majority_failure"
                        ]
                    ),
                    "gate": summary["gate"],
                },
                ensure_ascii=False,
                indent=2,
            ))

    output["test_macro"] = {
        method: float(np.mean(values))
        for method, values
        in macro_values.items()
    }
    output["gate_totals"] = dict(
        gate_totals
    )
    output["elapsed_seconds"] = round(
        time.time() - started,
        3,
    )

    print()
    print("=" * 76)
    print("===== 三测试集宏平均 =====")
    print(json.dumps(
        output["test_macro"],
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("===== Gate 总计 =====")
    print(json.dumps(
        output["gate_totals"],
        ensure_ascii=False,
        indent=2,
    ))

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("结果：", OUTPUT)
    print(
        "耗时秒：",
        output["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
