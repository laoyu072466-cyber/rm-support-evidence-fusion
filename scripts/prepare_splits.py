from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DEDUP = ROOT / "data/interim/dedup"
SPLIT_DIR = ROOT / "data/splits"
MANIFEST_DIR = ROOT / "data/manifests"
CONFIG_DIR = ROOT / "configs"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260829
FUZZY_REMOVE_THRESHOLD = 96.0

RATIOS = {
    "train": 0.65,
    "layer_discovery": 0.10,
    "pilot_validation": 0.10,
    "id_test": 0.15,
}


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    for token in (
        r"\left", r"\right", r"\,", r"\!", r"\;",
        r"\:", r"\quad", r"\qquad"
    ):
        text = text.replace(token, "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[\s`$]+", "", text)
    text = re.sub(r'[“”"\'‘’]', "", text)
    text = re.sub(r"[，。；：]+", "", text)
    return text


def problem_id(problem):
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


def primary_domain(row):
    domain = row.get("domain", "unknown")
    if isinstance(domain, list):
        domain = domain[0] if domain else "unknown"
    parts = [x.strip() for x in str(domain).split("->")]
    return parts[1] if len(parts) >= 2 else parts[0]


def difficulty_bucket(row):
    try:
        return str(int(round(float(row.get("difficulty", 0)))))
    except (TypeError, ValueError):
        return "unknown"


def stable_key(split_name, stratum):
    text = f"{SEED}|{stratum}|{split_name}"
    return hashlib.sha256(text.encode()).hexdigest()


def allocate_counts(n, stratum):
    raw = {name: n * ratio for name, ratio in RATIOS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = n - sum(counts.values())

    order = sorted(
        RATIOS,
        key=lambda name: (
            -(raw[name] - counts[name]),
            stable_key(name, stratum),
        ),
    )

    for name in order[:remaining]:
        counts[name] += 1

    return counts


omni = read_jsonl(RAW / "Omni-MATH/test.jsonl")
exact = read_jsonl(DEDUP / "exact_matches.jsonl")
fuzzy = read_jsonl(DEDUP / "fuzzy_candidates.jsonl")
internal = read_jsonl(DEDUP / "internal_duplicates.jsonl")

excluded_reasons = defaultdict(list)

for row in exact:
    excluded_reasons[row["omni_index"]].append("external_exact")

for row in fuzzy:
    if float(row["score"]) >= FUZZY_REMOVE_THRESHOLD:
        excluded_reasons[row["omni_index"]].append(
            f'external_fuzzy_{row["score"]}'
        )

for row in internal:
    excluded_reasons[row["omni_index"]].append("internal_duplicate")

clean = []
excluded = []

for index, row in enumerate(omni):
    pid = problem_id(row["problem"])

    if index in excluded_reasons:
        excluded.append({
            "original_index": index,
            "problem_id": pid,
            "reasons": excluded_reasons[index],
            "problem": row["problem"],
        })
        continue

    new_row = dict(row)
    new_row["problem_id"] = pid
    new_row["original_index"] = index
    new_row["stratify_domain"] = primary_domain(row)
    new_row["stratify_difficulty"] = difficulty_bucket(row)
    clean.append(new_row)

# 按“领域 × 难度”分组
groups = defaultdict(list)

for row in clean:
    stratum = (
        f'{row["stratify_domain"]}|'
        f'{row["stratify_difficulty"]}'
    )
    groups[stratum].append(row)

splits = {name: [] for name in RATIOS}

for stratum in sorted(groups):
    rows = groups[stratum]

    rows.sort(
        key=lambda row: hashlib.sha256(
            f'{SEED}|{row["problem_id"]}'.encode()
        ).hexdigest()
    )

    counts = allocate_counts(len(rows), stratum)
    position = 0

    for split_name in RATIOS:
        count = counts[split_name]
        selected = rows[position:position + count]
        position += count

        for row in selected:
            row["split"] = split_name
            splits[split_name].append(row)

# 完整性检查
all_ids = [
    row["problem_id"]
    for rows in splits.values()
    for row in rows
]

assert len(all_ids) == len(set(all_ids)), "发现问题跨 split 重复"
assert len(all_ids) == len(clean), "划分后题目数量不一致"

for split_name, rows in splits.items():
    rows.sort(key=lambda row: row["problem_id"])
    write_jsonl(
        SPLIT_DIR / f"omni_{split_name}.jsonl",
        rows,
    )

write_jsonl(
    ROOT / "data/interim/dedup/excluded_omni_records.jsonl",
    excluded,
)

file_hashes = {}

for split_name in RATIOS:
    path = SPLIT_DIR / f"omni_{split_name}.jsonl"
    file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "seed": SEED,
    "original_total": len(omni),
    "external_exact_records": len(exact),
    "fuzzy_threshold": FUZZY_REMOVE_THRESHOLD,
    "fuzzy_records_removed": sum(
        float(row["score"]) >= FUZZY_REMOVE_THRESHOLD
        for row in fuzzy
    ),
    "internal_duplicate_records": len(internal),
    "unique_excluded_records": len(excluded),
    "remaining_unique_questions": len(clean),
    "splits": {
        name: {
            "questions": len(rows),
            "ratio": round(len(rows) / len(clean), 6),
            "candidates_at_k8": len(rows) * 8,
        }
        for name, rows in splits.items()
    },
    "sha256": file_hashes,
}

config = {
    "dataset": "AI-ModelScope/Omni-MATH",
    "input": "data/raw/Omni-MATH/test.jsonl",
    "seed": SEED,
    "fuzzy_remove_threshold": FUZZY_REMOVE_THRESHOLD,
    "split_ratios": RATIOS,
    "stratification": ["primary_domain", "rounded_difficulty"],
    "candidate_count_per_question": 8,
}

(MANIFEST_DIR / "omni_split_manifest.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

(CONFIG_DIR / "data_split.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n数据划分完成，原始数据没有被修改。")
