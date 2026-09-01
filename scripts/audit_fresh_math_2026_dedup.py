from pathlib import Path
from collections import defaultdict
import hashlib
import html
import json
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


ROOT = Path("/root/autodl-tmp/rm_traj_project")
QUESTIONS_PATH = (
    ROOT / "data/external/fresh_math_2026/questions.jsonl"
)
OUTPUT_PATH = (
    ROOT / "data/manifests/fresh_math_2026_dedup_audit.json"
)

REFERENCE_PATHS = sorted(
    (ROOT / "data/processed/prototype_v2").glob("*.jsonl")
)

REFERENCE_PATHS += [
    ROOT / "data/raw/Omni-MATH/test.jsonl",
    ROOT / "data/raw/ProcessBench/olympiadbench.json",
    ROOT / "data/raw/ProcessBench/omnimath.json",
]

REFERENCE_PATHS += sorted(
    (
        ROOT
        / "data/raw/OlympiadBench/OlympiadBench"
    ).rglob("*.parquet")
)

TEXT_KEYS = [
    "problem",
    "question",
    "prompt",
    "problem_text",
    "instruction",
]

HIGH_RISK_RATIO = 92.0
HIGH_RISK_TOKEN_SET = 95.0
REVIEW_RATIO = 84.0
TOP_MATCHES = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_normalize(text):
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text).lower()

    for command in [
        r"\left",
        r"\right",
        r"\displaystyle",
        r"\textstyle",
    ]:
        text = text.replace(command, "")

    text = re.sub(
        r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}",
        r"\1",
        text,
    )
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    return text


def token_normalize(text):
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(
        r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}",
        r" \1 ",
        text,
    )
    tokens = re.findall(
        r"[a-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]",
        text,
    )
    return " ".join(tokens)


def extract_text(record):
    for key in TEXT_KEYS:
        value = record.get(key)

        if isinstance(value, str) and len(value.strip()) >= 20:
            return value.strip(), key

        if isinstance(value, list):
            parts = [
                str(item).strip()
                for item in value
                if isinstance(item, str)
                and item.strip()
            ]
            joined = "\n".join(parts)
            if len(joined) >= 20:
                return joined, key

    return None, None


def iter_nested_records(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from iter_nested_records(child)

    elif isinstance(value, list):
        for child in value:
            yield from iter_nested_records(child)


def iter_records(path):
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if line.strip():
                    yield index, json.loads(line)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            content = json.load(file)

        for index, record in enumerate(
            iter_nested_records(content)
        ):
            yield index, record

    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
        for index, record in enumerate(
            frame.to_dict(orient="records")
        ):
            yield index, record

    else:
        raise RuntimeError(f"不支持的文件：{path}")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"为避免覆盖审计结果，已停止：{OUTPUT_PATH}"
        )

    new_questions = read_jsonl(QUESTIONS_PATH)
    print("新盲测题目：", len(new_questions))
    print("密封标签未读取。")

    references = {}
    source_stats = {}

    for path in REFERENCE_PATHS:
        if not path.exists():
            continue

        relative = str(path.relative_to(ROOT))
        record_count = 0
        extracted_count = 0

        for record_index, record in iter_records(path):
            record_count += 1

            if not isinstance(record, dict):
                continue

            text, text_key = extract_text(record)
            if text is None:
                continue

            compact = compact_normalize(text)
            if len(compact) < 20:
                continue

            extracted_count += 1

            if compact not in references:
                references[compact] = {
                    "problem": text,
                    "token_text": token_normalize(text),
                    "occurrences": [],
                }

            if len(
                references[compact]["occurrences"]
            ) < 8:
                references[compact]["occurrences"].append({
                    "file": relative,
                    "record_index": record_index,
                    "text_key": text_key,
                })

        source_stats[relative] = {
            "records_seen": record_count,
            "questions_extracted": extracted_count,
        }

        print(
            f"{relative}: records={record_count}, "
            f"questions={extracted_count}"
        )

    choices = list(references)
    print()
    print("唯一参考问题：", len(choices))

    exact_matches = []
    high_risk_matches = []
    review_matches = []
    per_question = []

    for question in new_questions:
        query_text = question["problem"]
        query_compact = compact_normalize(query_text)
        query_tokens = token_normalize(query_text)

        exact = query_compact in references
        top = process.extract(
            query_compact,
            choices,
            scorer=fuzz.ratio,
            limit=TOP_MATCHES,
        )

        matches = []

        for matched_compact, ratio, _ in top:
            reference = references[matched_compact]
            token_set = fuzz.token_set_ratio(
                query_tokens,
                reference["token_text"],
            )
            length_ratio = (
                min(
                    len(query_compact),
                    len(matched_compact),
                )
                / max(
                    len(query_compact),
                    len(matched_compact),
                )
            )

            match = {
                "ratio": round(float(ratio), 4),
                "token_set_ratio": round(
                    float(token_set),
                    4,
                ),
                "length_ratio": round(
                    float(length_ratio),
                    4,
                ),
                "matched_problem_preview": (
                    reference["problem"][:500]
                ),
                "occurrences": reference[
                    "occurrences"
                ],
            }
            matches.append(match)

        best = matches[0]
        high_risk = (
            exact
            or (
                best["ratio"] >= HIGH_RISK_RATIO
                and best["token_set_ratio"]
                >= HIGH_RISK_TOKEN_SET
            )
        )
        review = (
            high_risk
            or best["ratio"] >= REVIEW_RATIO
        )

        result = {
            "question_uid": question["question_uid"],
            "source_dataset": question[
                "source_dataset"
            ],
            "problem_id": question["problem_id"],
            "exact_match": exact,
            "high_risk_near_duplicate": high_risk,
            "manual_review": review,
            "top_matches": matches,
        }
        per_question.append(result)

        if exact:
            exact_matches.append(result)
        if high_risk:
            high_risk_matches.append(result)
        if review:
            review_matches.append(result)

    ranked = sorted(
        per_question,
        key=lambda item: (
            item["top_matches"][0]["ratio"],
            item["top_matches"][0][
                "token_set_ratio"
            ],
        ),
        reverse=True,
    )

    output = {
        "version": "fresh_math_2026_dedup_audit_v1",
        "labels_read": False,
        "new_question_file": str(
            QUESTIONS_PATH.relative_to(ROOT)
        ),
        "new_question_sha256": sha256_file(
            QUESTIONS_PATH
        ),
        "new_questions": len(new_questions),
        "reference_files": len(source_stats),
        "reference_occurrences": sum(
            item["questions_extracted"]
            for item in source_stats.values()
        ),
        "reference_unique_questions": len(references),
        "thresholds": {
            "exact": "normalized character equality",
            "high_risk_ratio": HIGH_RISK_RATIO,
            "high_risk_token_set_ratio": (
                HIGH_RISK_TOKEN_SET
            ),
            "manual_review_ratio": REVIEW_RATIO,
        },
        "summary": {
            "exact_matches": len(exact_matches),
            "high_risk_near_duplicates": len(
                high_risk_matches
            ),
            "manual_review": len(review_matches),
            "decision": (
                "pass"
                if not high_risk_matches
                else "manual_review_required"
            ),
        },
        "source_statistics": source_stats,
        "manual_review_items": review_matches,
        "all_questions_ranked": ranked,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== 查重结论 =====")
    print(
        json.dumps(
            output["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("===== 相似度最高的 10 道题 =====")
    for item in ranked[:10]:
        best = item["top_matches"][0]
        occurrence = best["occurrences"][0]
        print({
            "dataset": item["source_dataset"],
            "problem_id": item["problem_id"],
            "ratio": best["ratio"],
            "token_set": best["token_set_ratio"],
            "reference": occurrence["file"],
            "reference_index": occurrence[
                "record_index"
            ],
        })

    print()
    print("结果：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
