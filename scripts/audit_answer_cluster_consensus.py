from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data/processed/prototype_v2"
SCORE_ROOT = (
    ROOT / "data/cache/trajectory_features_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/layer_28"
)

DATASETS = {
    "GSM8K_PILOT": (
        DATA_ROOT / "gsm_pilot_validation.jsonl",
        SCORE_ROOT / "gsm_pilot.scores_f32.npy",
        "pilot",
        "gsm",
    ),
    "MATH_PILOT": (
        DATA_ROOT / "math_pilot_validation.jsonl",
        SCORE_ROOT / "math_pilot.scores_f32.npy",
        "pilot",
        "math",
    ),
    "GSM8K_ID": (
        DATA_ROOT / "gsm_id_test_mixed.jsonl",
        SCORE_ROOT / "gsm_id_test.scores_f32.npy",
        "test",
        "gsm",
    ),
    "MATH_ID": (
        DATA_ROOT / "math_id_test_mixed.jsonl",
        SCORE_ROOT / "math_id_test.scores_f32.npy",
        "test",
        "math",
    ),
    "SVAMP_OOD": (
        DATA_ROOT / "svamp_ood_mixed.jsonl",
        SCORE_ROOT / "svamp_ood.scores_f32.npy",
        "test",
        "gsm",
    ),
}

TAUS = [0.25, 0.50, 1.0, 2.0, 4.0]
THRESHOLDS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_boxed(text):
    markers = [
        r"\boxed{",
        r"\fbox{",
    ]
    results = []

    for marker in markers:
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break

            content_start = index + len(marker)
            depth = 1
            cursor = content_start

            while cursor < len(text) and depth:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1

            if depth == 0:
                results.append(
                    text[content_start:cursor - 1]
                )

            start = index + 1

    return results[-1] if results else None


def normalize_answer(value, family):
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace("\u2212", "-")
    value = value.replace(r"\left", "")
    value = value.replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac")
    value = value.replace(r"\tfrac", r"\frac")
    value = value.replace("$", "")

    value = re.sub(
        r"\\(?:text|mathrm)\{([^{}]*)\}",
        r"\1",
        value,
    )

    # 处理简单数值分数。
    for _ in range(4):
        updated = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            r"(\1)/(\2)",
            value,
        )
        if updated == value:
            break
        value = updated

    value = value.strip()
    value = re.sub(
        r"^[=:]\s*",
        "",
        value,
    )
    value = re.sub(
        r"[\s。.!]+$",
        "",
        value,
    )
    value = re.sub(r"\s+", "", value)
    value = value.lower()

    # 只对纯数值答案做精确有理数规范化。
    numeric = value.replace(",", "")
    numeric = numeric.replace("(", "").replace(")", "")

    try:
        if re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?",
            numeric,
        ):
            number = Fraction(numeric)
            return f"number:{number.numerator}/{number.denominator}"
    except Exception:
        pass

    if family == "gsm":
        value = value.replace(",", "")

    return "expr:" + value if value else None


def extract_answer(text, family):
    matches = re.findall(
        r"####\s*([^\n\r]+)",
        text,
    )
    if matches:
        return normalize_answer(
            matches[-1],
            family,
        ), "hash"

    boxed = last_boxed(text)
    if boxed is not None:
        return normalize_answer(
            boxed,
            family,
        ), "boxed"

    final_patterns = [
        r"(?i)final answer(?: is|:)?\s*([^\n]+)$",
        r"(?i)the answer(?: is|:)?\s*([^\n]+)$",
        r"(?i)therefore[,\s]+([^\n]+)$",
    ]

    for pattern in final_patterns:
        match = re.search(pattern, text.strip())
        if match:
            answer = normalize_answer(
                match.group(1),
                family,
            )
            if answer:
                return answer, "final_phrase"

    # 只作为覆盖率诊断使用的弱回退。
    last_line = text.strip().splitlines()[-1]
    numbers = re.findall(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
        r"(?:/[-+]?\d+(?:\.\d+)?)?",
        last_line,
    )
    if numbers:
        return normalize_answer(
            numbers[-1],
            family,
        ), "last_number"

    return None, "unparsed"


def canonical_solution(text):
    text = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + math.log(
        float(np.exp(values - maximum).sum())
    )


def prepare_dataset(name, spec):
    data_path, score_path, role, family = spec

    rows = read_jsonl(data_path)
    scores = np.asarray(
        np.load(score_path),
        dtype=np.float32,
    )

    if len(rows) != len(scores):
        raise RuntimeError(
            f"{name} 行数和 RM 分数不一致："
            f"{len(rows)} != {len(scores)}"
        )

    groups = defaultdict(list)
    parse_methods = Counter()
    parsed = []
    cluster_labels = defaultdict(list)
    unparsed_examples = []

    for index, row in enumerate(rows):
        answer, method = extract_answer(
            str(row["solution_text"]),
            family,
        )
        parsed.append(answer)
        parse_methods[method] += 1

        uid = str(row["question_uid"])
        groups[uid].append(index)

        if answer is not None:
            cluster_labels[(uid, answer)].append(
                int(row["label"])
            )
        elif len(unparsed_examples) < 10:
            unparsed_examples.append({
                "uid": uid,
                "candidate_index": row.get(
                    "candidate_index"
                ),
                "preview": str(
                    row["solution_text"]
                )[-500:],
            })

    pure_clusters = 0
    mixed_clusters = 0

    for labels in cluster_labels.values():
        if len(set(labels)) == 1:
            pure_clusters += 1
        else:
            mixed_clusters += 1

    return {
        "name": name,
        "role": role,
        "family": family,
        "rows": rows,
        "scores": scores,
        "groups": dict(groups),
        "answers": parsed,
        "audit": {
            "candidates": len(rows),
            "questions": len(groups),
            "parse_coverage": float(
                np.mean([
                    value is not None
                    for value in parsed
                ])
            ),
            "parse_methods": dict(parse_methods),
            "answer_clusters": len(cluster_labels),
            "pure_clusters": pure_clusters,
            "mixed_clusters": mixed_clusters,
            "cluster_purity": float(
                pure_clusters
                / max(
                    pure_clusters + mixed_clusters,
                    1,
                )
            ),
            "unparsed_examples": unparsed_examples,
        },
    }


def cluster_question(dataset, uid):
    indices = np.asarray(
        dataset["groups"][uid],
        dtype=np.int64,
    )
    scores = dataset["scores"][indices]
    labels = np.asarray([
        int(dataset["rows"][index]["label"])
        for index in indices
    ])

    clusters = defaultdict(list)

    for local_index, global_index in enumerate(indices):
        answer = dataset["answers"][global_index]

        if answer is None:
            # 未解析候选不能互相形成伪共识。
            answer = f"__unparsed_{global_index}"

        clusters[answer].append(local_index)

    raw_local = int(np.argmax(scores))
    raw_answer = next(
        answer
        for answer, members in clusters.items()
        if raw_local in members
    )

    support = {}
    for answer, members in clusters.items():
        unique_solutions = {
            canonical_solution(
                dataset["rows"][
                    int(indices[member])
                ]["solution_text"]
            )
            for member in members
        }
        support[answer] = len(unique_solutions)

    return {
        "indices": indices,
        "scores": scores,
        "labels": labels,
        "clusters": dict(clusters),
        "support": support,
        "raw_local": raw_local,
        "raw_answer": raw_answer,
    }


def choose_majority(question):
    best_answer = max(
        question["clusters"],
        key=lambda answer: (
            question["support"][answer],
            max(
                question["scores"][
                    question["clusters"][answer]
                ]
            ),
        ),
    )
    members = question["clusters"][best_answer]
    return members[
        int(np.argmax(
            question["scores"][members]
        ))
    ]


def choose_weighted(
    question,
    tau,
    threshold,
):
    masses = {}

    for answer, members in question["clusters"].items():
        masses[answer] = logsumexp(
            question["scores"][members] / tau
        )

    best_answer = max(masses, key=masses.get)
    raw_answer = question["raw_answer"]

    if (
        best_answer != raw_answer
        and masses[best_answer] - masses[raw_answer]
        <= threshold
    ):
        best_answer = raw_answer

    members = question["clusters"][best_answer]
    selected = members[
        int(np.argmax(
            question["scores"][members]
        ))
    ]

    return selected


def evaluate(dataset, method, tau=None, threshold=None):
    raw_correct = []
    selected_correct = []
    switches = 0
    damage = 0
    correction = 0
    raw_correct_count = 0
    raw_wrong_count = 0

    for uid in dataset["groups"]:
        question = cluster_question(dataset, uid)

        raw_local = question["raw_local"]
        raw_ok = bool(
            question["labels"][raw_local] == 1
        )

        if method == "raw":
            selected = raw_local
        elif method == "majority":
            selected = choose_majority(question)
        elif method == "weighted":
            selected = choose_weighted(
                question,
                tau,
                threshold,
            )
        else:
            raise ValueError(method)

        selected_ok = bool(
            question["labels"][selected] == 1
        )

        raw_correct.append(raw_ok)
        selected_correct.append(selected_ok)

        if selected != raw_local:
            switches += 1

        if raw_ok:
            raw_correct_count += 1
            if not selected_ok:
                damage += 1
        else:
            raw_wrong_count += 1
            if selected_ok:
                correction += 1

    questions = len(raw_correct)

    return {
        "questions": questions,
        "raw_top1": float(np.mean(raw_correct)),
        "selected_top1": float(
            np.mean(selected_correct)
        ),
        "top1_delta": float(
            np.mean(selected_correct)
            - np.mean(raw_correct)
        ),
        "damage_rate": float(
            damage / max(raw_correct_count, 1)
        ),
        "correction_rate": float(
            correction / max(raw_wrong_count, 1)
        ),
        "switch_rate": float(
            switches / questions
        ),
        "net_corrected_questions": int(
            sum(selected_correct)
            - sum(raw_correct)
        ),
    }


def main():
    datasets = {
        name: prepare_dataset(name, spec)
        for name, spec in DATASETS.items()
    }

    print("===== 答案解析审计 =====")
    for name, dataset in datasets.items():
        audit = dataset["audit"]
        print(
            f"{name}: "
            f"候选={audit['candidates']}, "
            f"问题={audit['questions']}, "
            f"覆盖率={audit['parse_coverage']:.6f}, "
            f"簇纯度={audit['cluster_purity']:.6f}, "
            f"混合簇={audit['mixed_clusters']}, "
            f"方法={audit['parse_methods']}"
        )

    pilot_names = ["GSM8K_PILOT", "MATH_PILOT"]
    test_names = [
        "GSM8K_ID",
        "MATH_ID",
        "SVAMP_OOD",
    ]

    print("\n===== Pilot 简单多数投票 =====")
    majority_pilot = {}
    for name in pilot_names:
        majority_pilot[name] = evaluate(
            datasets[name],
            "majority",
        )
        print(name, majority_pilot[name])

    search = []

    print("\n===== Pilot 奖励加权共识搜索 =====")
    for tau in TAUS:
        for threshold in THRESHOLDS:
            per_dataset = {
                name: evaluate(
                    datasets[name],
                    "weighted",
                    tau=tau,
                    threshold=threshold,
                )
                for name in pilot_names
            }

            macro = {
                "top1": float(np.mean([
                    value["selected_top1"]
                    for value in per_dataset.values()
                ])),
                "damage": float(np.mean([
                    value["damage_rate"]
                    for value in per_dataset.values()
                ])),
                "switch": float(np.mean([
                    value["switch_rate"]
                    for value in per_dataset.values()
                ])),
            }

            search.append({
                "tau": tau,
                "threshold": threshold,
                "datasets": per_dataset,
                "macro": macro,
            })

    best_top1 = max(
        row["macro"]["top1"]
        for row in search
    )

    band = [
        row for row in search
        if row["macro"]["top1"]
        >= best_top1 - 0.005
    ]

    selected = min(
        band,
        key=lambda row: (
            row["macro"]["damage"],
            -row["macro"]["top1"],
            row["macro"]["switch"],
        ),
    )

    print(json.dumps(
        {
            "tau": selected["tau"],
            "threshold": selected["threshold"],
            "macro": selected["macro"],
            "datasets": selected["datasets"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    print("\n===== 当前测试集 =====")

    test_result = {}
    for name in test_names:
        raw = evaluate(datasets[name], "raw")
        majority = evaluate(
            datasets[name],
            "majority",
        )
        weighted = evaluate(
            datasets[name],
            "weighted",
            tau=selected["tau"],
            threshold=selected["threshold"],
        )

        test_result[name] = {
            "raw": raw,
            "majority": majority,
            "weighted_consensus": weighted,
        }

        print(
            f"{name}: "
            f"Raw={raw['selected_top1']:.6f} | "
            f"Majority={majority['selected_top1']:.6f} "
            f"({majority['top1_delta']:+.6f}) | "
            f"Weighted={weighted['selected_top1']:.6f} "
            f"({weighted['top1_delta']:+.6f}), "
            f"Damage={weighted['damage_rate']:.6f}, "
            f"Correction={weighted['correction_rate']:.6f}, "
            f"Switch={weighted['switch_rate']:.6f}"
        )

    weighted_macro = {
        "top1": float(np.mean([
            test_result[name][
                "weighted_consensus"
            ]["selected_top1"]
            for name in test_names
        ])),
        "top1_delta": float(np.mean([
            test_result[name][
                "weighted_consensus"
            ]["top1_delta"]
            for name in test_names
        ])),
        "damage": float(np.mean([
            test_result[name][
                "weighted_consensus"
            ]["damage_rate"]
            for name in test_names
        ])),
    }

    print("Weighted Macro:", weighted_macro)

    result = {
        "version": "answer_cluster_consensus_v1",
        "selection_uses_gold_answer": False,
        "audit": {
            name: dataset["audit"]
            for name, dataset in datasets.items()
        },
        "pilot_majority": majority_pilot,
        "selected_configuration": {
            "tau": selected["tau"],
            "threshold": selected["threshold"],
            "pilot": selected,
        },
        "test": test_result,
        "weighted_test_macro": weighted_macro,
    }

    output = (
        ROOT / "data/manifests/"
        "answer_cluster_consensus_v1.json"
    )
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("结果：", output)


if __name__ == "__main__":
    main()
