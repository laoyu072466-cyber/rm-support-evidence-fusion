from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import statistics
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "data/imported/"
    "SGLDSV_CANDIDATE_DATASETS_20260813"
)

FILES = {
    "gsm_train": BASE / "gsm8k_gen/train.jsonl",
    "gsm_val": BASE / "gsm8k_gen/val.jsonl",
    "gsm_test": BASE / "gsm8k_gen/test.jsonl",
    "math_train": BASE / "math_gen7b/train.jsonl",
    "math_val": BASE / "math_gen7b/val.jsonl",
    "math_test": BASE / "math_gen7b/test.jsonl",
    "math_probe_raw": BASE / "math_gen7b/raw_train_probe.jsonl",
    "svamp_test": BASE / "svamp_gen/test.jsonl",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[`$“”\"'‘’]", "", text)
    return text


def percentile(values, ratio):
    if not values:
        return None
    values = sorted(values)
    index = round((len(values) - 1) * ratio)
    return values[index]


all_problem_sets = {}
summaries = {}

for name, path in FILES.items():
    print("\n" + "=" * 72)
    print(name, "->", path.relative_to(BASE))

    if not path.exists():
        print("文件不存在")
        continue

    rows = 0
    invalid = 0
    first = None
    groups = defaultdict(lambda: {"count": 0, "labels": Counter()})
    record_hashes = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except Exception:
                invalid += 1
                continue

            rows += 1
            if first is None:
                first = row

            if not isinstance(row, dict):
                continue

            problem = row.get("problem", row.get("question", ""))
            problem_key = normalize(problem)
            groups[problem_key]["count"] += 1

            if "label" in row:
                groups[problem_key]["labels"][str(row["label"])] += 1

            response = row.get(
                "solution_text",
                row.get("response", row.get("completion", "")),
            )

            record_hash = hashlib.sha256(
                (
                    problem_key
                    + "\n"
                    + normalize(response)
                    + "\n"
                    + str(row.get("label"))
                ).encode("utf-8")
            ).hexdigest()
            record_hashes[record_hash] += 1

    counts = [group["count"] for group in groups.values()]
    labelled_groups = [
        group for group in groups.values()
        if group["labels"]
    ]

    mixed = sum(
        group["labels"].get("0", 0) > 0
        and group["labels"].get("1", 0) > 0
        for group in labelled_groups
    )
    all_correct = sum(
        group["labels"].get("1", 0) > 0
        and group["labels"].get("0", 0) == 0
        for group in labelled_groups
    )
    all_wrong = sum(
        group["labels"].get("0", 0) > 0
        and group["labels"].get("1", 0) == 0
        for group in labelled_groups
    )

    duplicate_rows = sum(
        count - 1 for count in record_hashes.values()
        if count > 1
    )

    print("有效候选行：", rows)
    print("无效行：", invalid)
    print("唯一问题：", len(groups))
    print("完全重复候选行：", duplicate_rows)

    if counts:
        print(
            "每题候选数：",
            f"min={min(counts)},",
            f"p50={statistics.median(counts)},",
            f"p90={percentile(counts, 0.90)},",
            f"max={max(counts)}",
        )

    if labelled_groups:
        print("混合正负问题：", mixed)
        print("全正确问题：", all_correct)
        print("全错误问题：", all_wrong)

    if isinstance(first, dict):
        print("字段：", sorted(first.keys()))

    all_problem_sets[name] = set(groups.keys())
    summaries[name] = {
        "candidate_rows": rows,
        "unique_problems": len(groups),
        "duplicate_candidate_rows": duplicate_rows,
        "mixed_problems": mixed,
        "all_correct_problems": all_correct,
        "all_wrong_problems": all_wrong,
        "candidate_min": min(counts) if counts else None,
        "candidate_median": statistics.median(counts) if counts else None,
        "candidate_p90": percentile(counts, 0.90),
        "candidate_max": max(counts) if counts else None,
    }

print("\n" + "=" * 72)
print("跨文件问题重叠：")

pairs = [
    ("gsm_train", "gsm_val"),
    ("gsm_train", "gsm_test"),
    ("gsm_val", "gsm_test"),
    ("math_train", "math_val"),
    ("math_train", "math_test"),
    ("math_val", "math_test"),
    ("math_probe_raw", "math_train"),
    ("math_probe_raw", "math_val"),
    ("math_probe_raw", "math_test"),
    ("gsm_test", "svamp_test"),
    ("math_test", "svamp_test"),
]

for left, right in pairs:
    if left in all_problem_sets and right in all_problem_sets:
        overlap = all_problem_sets[left] & all_problem_sets[right]
        print(f"{left} ∩ {right}: {len(overlap)}")

output = ROOT / "data/manifests/imported_candidates_summary.json"
output.write_text(
    json.dumps(summaries, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n汇总保存到：", output)
