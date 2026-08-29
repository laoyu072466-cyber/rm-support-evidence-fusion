from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import statistics
import unicodedata
import re

ROOT = Path("/root/autodl-tmp/rm_traj_project")
BASE = (
    ROOT / "data/imported/"
    "SGLDSV_CANDIDATE_DATASETS_20260813"
)
OUT = ROOT / "data/processed/prototype_v1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260829
DISCOVERY_RATIO = 0.10

SOURCE_FILES = {
    "math_train_source": BASE / "math_gen7b/train.jsonl",
    "math_pilot": BASE / "math_gen7b/val.jsonl",
    "math_test": BASE / "math_gen7b/test.jsonl",
    "gsm8k_ood": BASE / "gsm8k_gen/test.jsonl",
    "svamp_ood": BASE / "svamp_gen/test.jsonl",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", "", text)


def question_uid(problem):
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


def clean_candidates(rows):
    seen = set()
    cleaned = []
    duplicates = 0

    for row in rows:
        uid = question_uid(row["problem"])
        solution = row.get("solution_text", "")
        label = int(row["label"])

        key = (
            uid,
            normalize(solution),
            label,
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        cleaned.append(dict(row))

    return cleaned, duplicates


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[question_uid(row["problem"])].append(row)
    return groups


def assert_mixed(groups, name):
    bad = []

    for uid, rows in groups.items():
        labels = {int(row["label"]) for row in rows}
        if labels != {0, 1}:
            bad.append((uid, sorted(labels)))

    if bad:
        raise RuntimeError(
            f"{name} 中有 {len(bad)} 个问题不是正负混合"
        )


def flatten(groups, role, source_dataset):
    output = []

    for uid in sorted(groups):
        rows = groups[uid]

        for candidate_index, row in enumerate(rows):
            new_row = dict(row)
            new_row["question_uid"] = uid
            new_row["candidate_index"] = candidate_index
            new_row["data_role"] = role
            new_row["source_dataset"] = source_dataset
            output.append(new_row)

    return output


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows):
    groups = group_rows(rows)
    counts = [len(items) for items in groups.values()]
    labels = Counter(int(row["label"]) for row in rows)

    return {
        "questions": len(groups),
        "candidates": len(rows),
        "positive_candidates": labels[1],
        "negative_candidates": labels[0],
        "candidate_min": min(counts),
        "candidate_median": statistics.median(counts),
        "candidate_max": max(counts),
    }


# 清洗 MATH 训练来源
math_train_rows, math_train_dups = clean_candidates(
    read_jsonl(SOURCE_FILES["math_train_source"])
)
math_train_groups = group_rows(math_train_rows)
assert_mixed(math_train_groups, "math_train_source")

# 使用稳定哈希选出 10% Layer Discovery
ordered_uids = sorted(
    math_train_groups,
    key=lambda uid: hashlib.sha256(
        f"{SEED}|{uid}".encode()
    ).hexdigest(),
)

discovery_count = round(
    len(ordered_uids) * DISCOVERY_RATIO
)
discovery_uids = set(ordered_uids[:discovery_count])

train_groups = {
    uid: rows
    for uid, rows in math_train_groups.items()
    if uid not in discovery_uids
}
discovery_groups = {
    uid: rows
    for uid, rows in math_train_groups.items()
    if uid in discovery_uids
}

# 清洗其他集合
pilot_rows, pilot_dups = clean_candidates(
    read_jsonl(SOURCE_FILES["math_pilot"])
)
test_rows, test_dups = clean_candidates(
    read_jsonl(SOURCE_FILES["math_test"])
)
gsm_rows, gsm_dups = clean_candidates(
    read_jsonl(SOURCE_FILES["gsm8k_ood"])
)
svamp_rows, svamp_dups = clean_candidates(
    read_jsonl(SOURCE_FILES["svamp_ood"])
)

pilot_groups = group_rows(pilot_rows)
test_groups = group_rows(test_rows)
gsm_groups = group_rows(gsm_rows)
svamp_groups = group_rows(svamp_rows)

for name, groups in [
    ("math_pilot", pilot_groups),
    ("math_test", test_groups),
    ("gsm8k_ood", gsm_groups),
    ("svamp_ood", svamp_groups),
]:
    assert_mixed(groups, name)

# 确认 MATH 的四个角色互不重叠
math_sets = {
    "train": set(train_groups),
    "layer_discovery": set(discovery_groups),
    "pilot_validation": set(pilot_groups),
    "id_test": set(test_groups),
}

names = list(math_sets)

for i, left in enumerate(names):
    for right in names[i + 1:]:
        overlap = math_sets[left] & math_sets[right]
        if overlap:
            raise RuntimeError(
                f"{left} 与 {right} 重叠 {len(overlap)} 题"
            )

# 从 OOD 中排除与全部 MATH 完全相同的问题
all_math_uids = set().union(*math_sets.values())

gsm_overlap = set(gsm_groups) & all_math_uids
for uid in gsm_overlap:
    del gsm_groups[uid]

svamp_overlap = (
    set(svamp_groups)
    & (all_math_uids | set(gsm_groups))
)
for uid in svamp_overlap:
    del svamp_groups[uid]

roles = {
    "math_train": flatten(
        train_groups, "train", "MATH/Qwen2-7B"
    ),
    "math_layer_discovery": flatten(
        discovery_groups,
        "layer_discovery",
        "MATH/Qwen2-7B",
    ),
    "math_pilot_validation": flatten(
        pilot_groups,
        "pilot_validation",
        "MATH/Qwen2-7B",
    ),
    "math_id_test_mixed": flatten(
        test_groups,
        "id_test_mixed",
        "MATH-500/Qwen2-7B",
    ),
    "gsm8k_ood_mixed": flatten(
        gsm_groups,
        "ood_mixed",
        "GSM8K/Qwen2-1.5B",
    ),
    "svamp_ood_mixed": flatten(
        svamp_groups,
        "ood_mixed",
        "SVAMP/Qwen2-1.5B",
    ),
}

hashes = {}

for name, rows in roles.items():
    path = OUT / f"{name}.jsonl"
    write_jsonl(path, rows)
    hashes[path.name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

summary = {
    "version": "prototype_v1",
    "seed": SEED,
    "purpose": "engineering feasibility, not final main experiment",
    "layer_discovery_source": (
        "deterministic 10% question-level split "
        "from prepared MATH training data"
    ),
    "ignored_file": (
        "math_gen7b/raw_train_probe.jsonl "
        "because it overlaps train/validation"
    ),
    "duplicates_removed": {
        "math_train": math_train_dups,
        "math_pilot": pilot_dups,
        "math_test": test_dups,
        "gsm8k_ood": gsm_dups,
        "svamp_ood": svamp_dups,
    },
    "cross_dataset_exact_overlap_removed": {
        "gsm8k": len(gsm_overlap),
        "svamp": len(svamp_overlap),
    },
    "roles": {
        name: summarize(rows)
        for name, rows in roles.items()
    },
    "sha256": hashes,
}

config = {
    "version": "prototype_v1",
    "seed": SEED,
    "variable_candidates_per_question": True,
    "candidate_range_expected": [4, 16],
    "training_distribution": "MATH/Qwen2-7B",
    "layer_discovery_ratio": DISCOVERY_RATIO,
    "pilot_distribution": "MATH/Qwen2-7B",
    "id_test_distribution": "MATH-500/Qwen2-7B",
    "ood_distributions": [
        "GSM8K/Qwen2-1.5B",
        "SVAMP/Qwen2-1.5B",
    ],
}

(ROOT / "data/manifests/prototype_v1_manifest.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

(ROOT / "configs/prototype_data.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n原型数据准备完成：", OUT)
