from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import unicodedata

import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

ROOT = Path("/root/autodl-tmp/rm_traj_project")
RAW = ROOT / "data/raw"
OUT = ROOT / "data/interim/dedup"
OUT.mkdir(parents=True, exist_ok=True)

FUZZY_THRESHOLD = 92.0


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


def short_id(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 读取 Omni-MATH
omni = []
with (RAW / "Omni-MATH/test.jsonl").open("r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        if line.strip():
            row = json.loads(line)
            row["_original_index"] = index
            omni.append(row)

# 读取所有外部测试问题
external = []

for path in sorted((RAW / "ProcessBench").glob("*.json")):
    if path.name == "dataset_infos.json":
        continue
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        external.append({
            "source": f"ProcessBench/{path.stem}",
            "id": row.get("id"),
            "problem": row["problem"],
        })

olympiad_path = (
    RAW / "OlympiadBench/OlympiadBench/"
    "OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet"
)

for row in pq.read_table(
    olympiad_path, columns=["id", "question"]
).to_pylist():
    external.append({
        "source": "OlympiadBench/OE_TO_maths_en_COMP",
        "id": row["id"],
        "problem": row["question"],
    })

external_norms = [normalize(row["problem"]) for row in external]
exact_map = defaultdict(list)

for index, norm in enumerate(external_norms):
    exact_map[norm].append(index)

exact_matches = []
fuzzy_candidates = []
internal_duplicates = []
seen_omni = {}

print(f"Omni-MATH 题数：{len(omni)}")
print(f"外部测试问题数：{len(external)}")
print("开始进行规范化精确匹配和模糊匹配……", flush=True)

for index, row in enumerate(omni):
    problem = row["problem"]
    norm = normalize(problem)
    pid = short_id(problem)

    if norm in seen_omni:
        internal_duplicates.append({
            "omni_index": index,
            "duplicate_of": seen_omni[norm],
            "problem_id": pid,
            "problem": problem,
        })
    else:
        seen_omni[norm] = index

    if norm in exact_map:
        ext = external[exact_map[norm][0]]
        exact_matches.append({
            "omni_index": index,
            "problem_id": pid,
            "reason": "normalized_exact",
            "external_source": ext["source"],
            "external_id": ext["id"],
            "omni_problem": problem,
            "external_problem": ext["problem"],
        })
    else:
        match = process.extractOne(
            norm,
            external_norms,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if match is not None:
            _, score, external_index = match
            ext = external[external_index]
            fuzzy_candidates.append({
                "omni_index": index,
                "problem_id": pid,
                "score": round(float(score), 2),
                "external_source": ext["source"],
                "external_id": ext["id"],
                "omni_problem": problem,
                "external_problem": ext["problem"],
            })

    if (index + 1) % 500 == 0:
        print(f"已检查 {index + 1}/{len(omni)}", flush=True)

exact_indices = {row["omni_index"] for row in exact_matches}

summary = {
    "omni_total": len(omni),
    "external_total": len(external),
    "exact_cross_dataset_matches": len(exact_matches),
    "fuzzy_candidates_threshold": FUZZY_THRESHOLD,
    "fuzzy_candidates": len(fuzzy_candidates),
    "internal_duplicates": len(internal_duplicates),
    "remaining_after_exact_removal": len(omni) - len(exact_indices),
}

write_jsonl(OUT / "exact_matches.jsonl", exact_matches)
write_jsonl(OUT / "fuzzy_candidates.jsonl", fuzzy_candidates)
write_jsonl(OUT / "internal_duplicates.jsonl", internal_duplicates)

with (OUT / "dedup_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n查重结果：")
print(json.dumps(summary, ensure_ascii=False, indent=2))

print("\n分数最高的模糊匹配候选：")
for row in sorted(
    fuzzy_candidates, key=lambda x: x["score"], reverse=True
)[:10]:
    print(
        f'{row["score"]:>6.2f} | '
        f'{row["external_source"]} | '
        f'Omni #{row["omni_index"]}'
    )

print(f"\n详细报告目录：{OUT}")
