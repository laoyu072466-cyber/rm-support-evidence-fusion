from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import re
import statistics
import unicodedata

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
IMPORTED = (
    ROOT / "data/imported/"
    "SGLDSV_CANDIDATE_DATASETS_20260813"
)
V1 = ROOT / "data/processed/prototype_v1"
OUT = ROOT / "data/processed/prototype_v2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260829
DISCOVERY_RATIO = 0.10


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", "", text)


def uid(problem):
    return hashlib.sha256(
        normalize(problem).encode("utf-8")
    ).hexdigest()[:16]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean(rows):
    seen = set()
    output = []
    duplicates = 0

    for row in rows:
        qid = uid(row["problem"])
        key = (
            qid,
            normalize(row.get("solution_text", "")),
            int(row["label"]),
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        output.append(dict(row))

    return output, duplicates


def group(rows):
    result = defaultdict(list)
    for row in rows:
        result[uid(row["problem"])].append(row)
    return result


def flatten(groups, role, source):
    output = []

    for qid in sorted(groups):
        for index, row in enumerate(groups[qid]):
            new = dict(row)
            new["question_uid"] = qid
            new["candidate_index"] = index
            new["data_role"] = role
            new["source_dataset"] = source
            output.append(new)

    return output


def relabel(rows, role):
    output = []
    groups = group(rows)

    for qid in sorted(groups):
        for index, row in enumerate(groups[qid]):
            new = dict(row)
            new["question_uid"] = qid
            new["candidate_index"] = index
            new["data_role"] = role
            output.append(new)

    return output


def assert_mixed(rows, name):
    groups = group(rows)
    bad = 0

    for items in groups.values():
        labels = {int(row["label"]) for row in items}
        if labels != {0, 1}:
            bad += 1

    if bad:
        raise RuntimeError(
            f"{name} 有 {bad} 个问题不是正负混合"
        )


def summary(rows):
    groups = group(rows)
    counts = [len(items) for items in groups.values()]
    labels = Counter(int(row["label"]) for row in rows)

    return {
        "questions": len(groups),
        "candidates": len(rows),
        "positive": labels[1],
        "negative": labels[0],
        "candidate_min": min(counts),
        "candidate_median": statistics.median(counts),
        "candidate_max": max(counts),
    }


# ---------- GSM8K ----------

gsm_source, gsm_duplicates = clean(
    read_jsonl(IMPORTED / "gsm8k_gen/train.jsonl")
)
gsm_groups = group(gsm_source)

ordered = sorted(
    gsm_groups,
    key=lambda qid: hashlib.sha256(
        f"{SEED}|gsm|{qid}".encode()
    ).hexdigest(),
)

n_discovery = round(len(ordered) * DISCOVERY_RATIO)
gsm_discovery_ids = set(ordered[:n_discovery])

gsm_train_groups = {
    qid: rows for qid, rows in gsm_groups.items()
    if qid not in gsm_discovery_ids
}
gsm_discovery_groups = {
    qid: rows for qid, rows in gsm_groups.items()
    if qid in gsm_discovery_ids
}

gsm_val, gsm_val_duplicates = clean(
    read_jsonl(IMPORTED / "gsm8k_gen/val.jsonl")
)

gsm_test = read_jsonl(V1 / "gsm8k_ood_mixed.jsonl")

gsm_train = flatten(
    gsm_train_groups, "train", "GSM8K/Qwen2-1.5B"
)
gsm_discovery = flatten(
    gsm_discovery_groups,
    "layer_discovery",
    "GSM8K/Qwen2-1.5B",
)
gsm_pilot = relabel(gsm_val, "pilot_validation")
gsm_test = relabel(gsm_test, "id_test_mixed")

# ---------- MATH：直接使用已经清洗的 v1 ----------

math_train = read_jsonl(V1 / "math_train.jsonl")
math_discovery = read_jsonl(
    V1 / "math_layer_discovery.jsonl"
)
math_pilot = read_jsonl(
    V1 / "math_pilot_validation.jsonl"
)
math_test = read_jsonl(
    V1 / "math_id_test_mixed.jsonl"
)

# ---------- SVAMP 简单 OOD ----------

svamp = read_jsonl(V1 / "svamp_ood_mixed.jsonl")

# ---------- RewardBench 2 Math 困难 OOD ----------

rb_path = (
    ROOT / "data/raw/reward-bench-2/data/"
    "test-00000-of-00001.parquet"
)

rb_rows = pq.read_table(rb_path).to_pylist()
rb_math = []

for row in rb_rows:
    if str(row["subset"]).lower() != "math":
        continue

    qid = uid(row["prompt"])
    completions = []

    chosen = row["chosen"]
    rejected = row["rejected"]

    if not isinstance(chosen, list):
        chosen = [chosen]
    if not isinstance(rejected, list):
        rejected = [rejected]

    for text in chosen:
        completions.append((text, 1))
    for text in rejected:
        completions.append((text, 0))

    for index, (text, label) in enumerate(completions):
        rb_math.append({
            "question_uid": qid,
            "problem_id": str(row["id"]),
            "problem": row["prompt"],
            "solution_text": str(text),
            "solution_steps": [
                part for part in str(text).split("\n\n")
                if part.strip()
            ],
            "label": label,
            "candidate_index": index,
            "data_role": "hard_ood",
            "source_dataset": "RewardBench2/Math",
        })

# ---------- ProcessBench 困难机制测试 ----------

process_rows = []

for subset in ("olympiadbench", "omnimath"):
    path = ROOT / f"data/raw/ProcessBench/{subset}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))

    candidate_counter = defaultdict(int)

    for row in rows:
        qid = uid(row["problem"])
        index = candidate_counter[qid]
        candidate_counter[qid] += 1

        steps = row["steps"]
        if not isinstance(steps, list):
            steps = [str(steps)]

        process_rows.append({
            "question_uid": qid,
            "problem_id": row["id"],
            "problem": row["problem"],
            "solution_steps": steps,
            "solution_text": "\n\n".join(steps),
            "label": int(bool(row["final_answer_correct"])),
            "first_error_label": row["label"],
            "generator": row["generator"],
            "candidate_index": index,
            "data_role": "mechanism_ood",
            "source_dataset": f"ProcessBench/{subset}",
        })

# ---------- 完整性检查 ----------

ranking_sets = {
    "gsm_train": gsm_train,
    "gsm_layer_discovery": gsm_discovery,
    "gsm_pilot_validation": gsm_pilot,
    "gsm_id_test_mixed": gsm_test,
    "math_train": math_train,
    "math_layer_discovery": math_discovery,
    "math_pilot_validation": math_pilot,
    "math_id_test_mixed": math_test,
    "svamp_ood_mixed": svamp,
    "rewardbench2_math_hard_ood": rb_math,
}

for name, rows in ranking_sets.items():
    assert_mixed(rows, name)

# 联合训练文件；实际训练时使用数据集平衡采样
joint_train = gsm_train + math_train
joint_discovery = gsm_discovery + math_discovery
joint_pilot = gsm_pilot + math_pilot

roles = {
    **ranking_sets,
    "joint_train": joint_train,
    "joint_layer_discovery": joint_discovery,
    "joint_pilot_validation": joint_pilot,
    "processbench_hard_mechanism": process_rows,
}

hashes = {}

for name, rows in roles.items():
    path = OUT / f"{name}.jsonl"
    write_jsonl(path, rows)
    hashes[path.name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

manifest = {
    "version": "prototype_v2",
    "seed": SEED,
    "experiments": {
        "gsm_only": {
            "train": "gsm_train.jsonl",
            "discovery": "gsm_layer_discovery.jsonl",
            "pilot": "gsm_pilot_validation.jsonl",
            "id_test": "gsm_id_test_mixed.jsonl",
        },
        "math_only": {
            "train": "math_train.jsonl",
            "discovery": "math_layer_discovery.jsonl",
            "pilot": "math_pilot_validation.jsonl",
            "id_test": "math_id_test_mixed.jsonl",
        },
        "joint": {
            "train": "joint_train.jsonl",
            "discovery": "joint_layer_discovery.jsonl",
            "pilot": "joint_pilot_validation.jsonl",
            "dataset_balanced_sampling": "50% GSM8K + 50% MATH",
            "id_tests": [
                "gsm_id_test_mixed.jsonl",
                "math_id_test_mixed.jsonl",
            ],
        },
    },
    "ood": {
        "simple": "svamp_ood_mixed.jsonl",
        "hard_ranking": "rewardbench2_math_hard_ood.jsonl",
        "hard_mechanism": "processbench_hard_mechanism.jsonl",
    },
    "gsm_duplicates_removed": {
        "train": gsm_duplicates,
        "pilot": gsm_val_duplicates,
    },
    "roles": {
        name: (
            summary(rows)
            if name != "processbench_hard_mechanism"
            else {
                "samples": len(rows),
                "unique_questions": len(group(rows)),
                "correct": sum(
                    int(row["label"]) for row in rows
                ),
                "incorrect": sum(
                    1 - int(row["label"]) for row in rows
                ),
            }
        )
        for name, rows in roles.items()
    },
    "sha256": hashes,
}

config = {
    "version": "prototype_v2",
    "training_modes": [
        "gsm_only",
        "math_only",
        "joint",
    ],
    "joint_dataset_sampling": {
        "GSM8K": 0.5,
        "MATH": 0.5,
    },
    "joint_selection_metric": (
        "mean(PairMacro_GSM8K, PairMacro_MATH)"
    ),
    "simple_ood": "SVAMP",
    "hard_ranking_ood": "RewardBench2 Math",
    "mechanism_ood": (
        "ProcessBench OlympiadBench + Omni-MATH"
    ),
}

(ROOT / "data/manifests/prototype_v2_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

(ROOT / "configs/prototype_v2_data.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("\nPrototype v2 完成：", OUT)
