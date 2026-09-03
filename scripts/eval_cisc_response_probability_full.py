from pathlib import Path
from collections import defaultdict
import json
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_answer_cluster_consensus as base


CACHE_ROOT = (
    ROOT / "data/cache/generator_cluster_features_v1"
)
OUTPUT = (
    ROOT / "data/manifests/"
    "cisc_response_probability_full_v1.json"
)

TEMPERATURES = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00,
]

CACHE_SPECS = {
    "GSM8K_PILOT": (
        CACHE_ROOT / "Qwen2-1.5B/gsm_pilot"
    ),
    "GSM8K_ID": (
        CACHE_ROOT / "Qwen2-1.5B/gsm_id_test"
    ),
    "SVAMP_OOD": (
        CACHE_ROOT / "Qwen2-1.5B/svamp_ood"
    ),
    "MATH_PILOT": (
        CACHE_ROOT / "Qwen2-7B/math_pilot"
    ),
    "MATH_ID": (
        CACHE_ROOT / "Qwen2-7B/math_id_test"
    ),
}

DOMAINS = {
    "GSM8K": {
        "pilot": "GSM8K_PILOT",
        "tests": [
            "GSM8K_ID",
            "SVAMP_OOD",
        ],
    },
    "MATH": {
        "pilot": "MATH_PILOT",
        "tests": [
            "MATH_ID",
        ],
    },
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_confidence(name, dataset):
    cache_base = CACHE_SPECS[name]

    nll = np.load(
        str(cache_base) + ".token_nll_f32.npy",
        mmap_mode="r",
    )
    metadata = read_jsonl(
        Path(str(cache_base) + ".metadata.jsonl")
    )

    if nll.ndim != 2 or nll.shape[1] != 5:
        raise RuntimeError(
            f"{name}: unexpected NLL shape {nll.shape}"
        )

    if not (
        len(nll)
        == len(metadata)
        == len(dataset["rows"])
    ):
        raise RuntimeError(
            f"{name}: candidate lengths differ"
        )

    for index, (row, meta) in enumerate(
        zip(dataset["rows"], metadata)
    ):
        if (
            str(row["question_uid"])
            != str(meta["question_uid"])
        ):
            raise RuntimeError(
                f"{name}: UID misalignment at {index}"
            )

        if int(row["candidate_index"]) != int(
            meta["candidate_index"]
        ):
            raise RuntimeError(
                f"{name}: candidate misalignment "
                f"at {index}"
            )

        if int(row["label"]) != int(meta["label"]):
            raise RuntimeError(
                f"{name}: label misalignment "
                f"at {index}"
            )

    mean_nll = np.asarray(
        nll[:, 0],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(mean_nll)):
        raise RuntimeError(
            f"{name}: non-finite mean NLL"
        )

    return np.exp(-mean_nll)


def selected_answer(question, selected):
    for answer, members in (
        question["clusters"].items()
    ):
        if selected in members:
            return answer

    raise RuntimeError("selected member not found")


def rm_representative(question, answer):
    members = question["clusters"][answer]
    return members[
        int(np.argmax(
            question["scores"][members]
        ))
    ]


def choose_count_sc(question):
    # 标准 Self-Consistency：
    # 每个采样候选贡献一票。
    best_answer = max(
        question["clusters"],
        key=lambda answer: (
            len(question["clusters"][answer]),
            max(
                question["scores"][
                    question["clusters"][answer]
                ]
            ),
        ),
    )
    return rm_representative(
        question,
        best_answer,
    )


def choose_cisc(question, confidence, temperature):
    local_confidence = confidence[
        question["indices"]
    ]

    # 官方 CISC:
    # softmax(confidence / T)。
    # 公共分母不影响答案簇之间的比较。
    shifted = (
        local_confidence
        - np.max(local_confidence)
    ) / temperature
    weights = np.exp(shifted)

    masses = {
        answer: float(
            np.sum(weights[members])
        )
        for answer, members
        in question["clusters"].items()
    }

    # 只有质量完全相同时才使用 RM 打破平局。
    best_answer = max(
        question["clusters"],
        key=lambda answer: (
            masses[answer],
            max(
                question["scores"][
                    question["clusters"][answer]
                ]
            ),
        ),
    )

    return rm_representative(
        question,
        best_answer,
    )


def choose_max_response(question, confidence):
    local_confidence = confidence[
        question["indices"]
    ]
    return int(np.argmax(local_confidence))


def evaluate(
    dataset,
    confidence,
    method,
    temperature=None,
):
    raw_correct = []
    selected_correct = []

    switches = 0
    corrections = 0
    damages = 0
    raw_correct_count = 0
    raw_wrong_count = 0

    for uid in dataset["groups"]:
        question = base.cluster_question(
            dataset,
            uid,
        )

        raw_local = question["raw_local"]
        raw_answer = question["raw_answer"]
        raw_ok = bool(
            question["labels"][raw_local] == 1
        )

        if method == "raw_rm":
            selected = raw_local
        elif method == "project_majority":
            selected = base.choose_majority(
                question
            )
        elif method == "count_sc":
            selected = choose_count_sc(
                question
            )
        elif method == "max_response":
            selected = choose_max_response(
                question,
                confidence,
            )
        elif method == "cisc_response":
            selected = choose_cisc(
                question,
                confidence,
                temperature,
            )
        else:
            raise ValueError(method)

        chosen_answer = selected_answer(
            question,
            selected,
        )
        selected_ok = bool(
            question["labels"][selected] == 1
        )

        raw_correct.append(raw_ok)
        selected_correct.append(selected_ok)

        if chosen_answer != raw_answer:
            switches += 1

        if raw_ok:
            raw_correct_count += 1
            if not selected_ok:
                damages += 1
        else:
            raw_wrong_count += 1
            if selected_ok:
                corrections += 1

    raw_top1 = float(np.mean(raw_correct))
    selected_top1 = float(
        np.mean(selected_correct)
    )

    return {
        "questions": len(raw_correct),
        "raw_top1": raw_top1,
        "selected_top1": selected_top1,
        "top1_delta": (
            selected_top1 - raw_top1
        ),
        "correct_questions": int(
            np.sum(selected_correct)
        ),
        "answer_switches": switches,
        "switch_rate": (
            switches / len(raw_correct)
        ),
        "corrections": corrections,
        "damages": damages,
        "net_corrected": (
            corrections - damages
        ),
        "correction_rate": (
            corrections
            / max(raw_wrong_count, 1)
        ),
        "damage_rate": (
            damages
            / max(raw_correct_count, 1)
        ),
    }


def choose_temperature(dataset, confidence):
    grid = {}

    for temperature in TEMPERATURES:
        metrics = evaluate(
            dataset,
            confidence,
            "cisc_response",
            temperature,
        )
        grid[str(temperature)] = metrics

        print(
            f"T={temperature:.2f} | "
            f"Top1={metrics['selected_top1']:.6f} | "
            f"Delta={metrics['top1_delta']:+.6f} | "
            f"Damage={metrics['damage_rate']:.6f}"
        )

    # 首先最大化 Pilot Top-1。
    # 完全同分时选择更大的温度，
    # 即更接近普通多数投票的保守配置。
    selected_temperature = max(
        TEMPERATURES,
        key=lambda value: (
            grid[str(value)]["selected_top1"],
            value,
        ),
    )

    return selected_temperature, grid


def main():
    started = time.time()

    print(
        "===== Generator-Likelihood "
        "Weighted Consensus ====="
    )
    print(
        "confidence = exp(-mean token NLL)"
    )
    print(
        "Pilot 选择温度；测试标签不参与配置。"
    )

    output = {
        "version": (
            "cisc_response_probability_full_v1"
        ),
        "method_display_name": (
            "Generator-Likelihood "
            "Weighted Consensus"
        ),
        "related_method": (
            "CISC Response Probability"
        ),
        "confidence_definition": (
            "exp(-token_nll_f32[:, 0])"
        ),
        "temperature_grid": TEMPERATURES,
        "temperature_selection": (
            "maximize Pilot full-candidate Top1; "
            "prefer larger T on exact ties"
        ),
        "cluster_parser": (
            "audit_answer_cluster_consensus"
        ),
        "cluster_representative": (
            "highest base-RM candidate inside "
            "the selected answer cluster"
        ),
        "domains": {},
    }

    all_test_results = defaultdict(list)

    for domain_name, spec in DOMAINS.items():
        print()
        print("=" * 76)
        print(domain_name)

        pilot_name = spec["pilot"]
        pilot = base.prepare_dataset(
            pilot_name,
            base.DATASETS[pilot_name],
        )
        pilot_confidence = load_confidence(
            pilot_name,
            pilot,
        )

        print(
            f"Pilot: {pilot_name}, "
            f"questions={pilot['audit']['questions']}, "
            f"parse={pilot['audit']['parse_coverage']:.6f}"
        )

        selected_temperature, grid = (
            choose_temperature(
                pilot,
                pilot_confidence,
            )
        )

        print(
            "选择温度：",
            selected_temperature,
        )

        pilot_methods = {
            "raw_rm": evaluate(
                pilot,
                pilot_confidence,
                "raw_rm",
            ),
            "project_majority": evaluate(
                pilot,
                pilot_confidence,
                "project_majority",
            ),
            "count_sc": evaluate(
                pilot,
                pilot_confidence,
                "count_sc",
            ),
            "max_response": evaluate(
                pilot,
                pilot_confidence,
                "max_response",
            ),
            "cisc_response": evaluate(
                pilot,
                pilot_confidence,
                "cisc_response",
                selected_temperature,
            ),
        }

        domain_result = {
            "selected_temperature": (
                selected_temperature
            ),
            "pilot_temperature_grid": grid,
            "pilot": {
                "name": pilot_name,
                "audit": pilot["audit"],
                "methods": pilot_methods,
            },
            "tests": {},
        }

        for test_name in spec["tests"]:
            dataset = base.prepare_dataset(
                test_name,
                base.DATASETS[test_name],
            )
            confidence = load_confidence(
                test_name,
                dataset,
            )

            methods = {
                "raw_rm": evaluate(
                    dataset,
                    confidence,
                    "raw_rm",
                ),
                "project_majority": evaluate(
                    dataset,
                    confidence,
                    "project_majority",
                ),
                "count_sc": evaluate(
                    dataset,
                    confidence,
                    "count_sc",
                ),
                "max_response": evaluate(
                    dataset,
                    confidence,
                    "max_response",
                ),
                "cisc_response": evaluate(
                    dataset,
                    confidence,
                    "cisc_response",
                    selected_temperature,
                ),
            }

            domain_result["tests"][test_name] = {
                "audit": dataset["audit"],
                "methods": methods,
            }

            for method, metrics in methods.items():
                all_test_results[method].append(
                    metrics
                )

            print()
            print(test_name)

            for method, metrics in methods.items():
                print(
                    f"  {method}: "
                    f"Top1={metrics['selected_top1']:.6f} "
                    f"({metrics['top1_delta']:+.6f}), "
                    f"Damage={metrics['damage_rate']:.6f}, "
                    f"Net={metrics['net_corrected']:+d}"
                )

        output["domains"][domain_name] = (
            domain_result
        )

    macro = {}

    print()
    print("=" * 76)
    print("===== 三测试集宏平均 =====")

    for method, values in all_test_results.items():
        summary = {
            "top1": float(np.mean([
                item["selected_top1"]
                for item in values
            ])),
            "top1_delta": float(np.mean([
                item["top1_delta"]
                for item in values
            ])),
            "damage_rate": float(np.mean([
                item["damage_rate"]
                for item in values
            ])),
            "switch_rate": float(np.mean([
                item["switch_rate"]
                for item in values
            ])),
        }
        macro[method] = summary
        print(
            f"{method}: "
            f"Top1={summary['top1']:.6f}, "
            f"Delta={summary['top1_delta']:+.6f}, "
            f"Damage={summary['damage_rate']:.6f}"
        )

    output["test_macro"] = macro
    output["elapsed_seconds"] = round(
        time.time() - started,
        3,
    )

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
