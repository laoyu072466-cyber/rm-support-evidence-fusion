from pathlib import Path
import hashlib
import json
import re
import statistics

from transformers import AutoTokenizer


PROJECT = Path("/root/autodl-tmp/rm_traj_project")
DATA_ROOT = PROJECT / "data/processed/prototype_v2"
MODEL_PATH = (
    PROJECT
    / "models/reward/Skywork-Reward-V2-Qwen3-1.7B"
)
CONFIG_PATH = PROJECT / "configs/trajectory_chunking.json"

CACHE_ROOT = (
    PROJECT
    / "data/cache/trajectory_chunks_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
)
MANIFEST_PATH = (
    PROJECT
    / "data/manifests/trajectory_endpoint_mapping.json"
)

FILES = [
    "gsm_layer_discovery.jsonl",
    "math_layer_discovery.jsonl",
    "gsm_train.jsonl",
    "math_train.jsonl",
    "gsm_pilot_validation.jsonl",
    "math_pilot_validation.jsonl",
    "gsm_id_test_mixed.jsonl",
    "math_id_test_mixed.jsonl",
    "svamp_ood_mixed.jsonl",
]


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

single_cap = int(
    config["single_line_policy"]["max_tokens"]
)
general_cap = int(
    config["general_policy"]["max_tokens"]
)
sentence_trigger = int(
    config["general_policy"]["sentence_split_if_over_tokens"]
)

sentence_boundary = re.compile(
    config["sentence_boundary_pattern"]
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True,
)

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def percentile(values, q):
    values = sorted(values)
    index = round((len(values) - 1) * q)
    return values[index]


def describe(values):
    return {
        "min": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
    }


def trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def nonempty_line_spans(text):
    spans = []

    for match in re.finditer(r"[^\r\n]+", text):
        start, end = trim_span(
            text,
            match.start(),
            match.end(),
        )
        if start < end:
            spans.append((start, end))

    return spans


def span_tokenization(text, start, end):
    return tokenizer(
        text[start:end],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )


def sentence_spans(text, start, end):
    segment = text[start:end]
    spans = []
    cursor = 0

    for match in sentence_boundary.finditer(segment):
        piece_start, piece_end = trim_span(
            text,
            start + cursor,
            start + match.start(),
        )
        if piece_start < piece_end:
            spans.append((piece_start, piece_end))

        cursor = match.end()

    piece_start, piece_end = trim_span(
        text,
        start + cursor,
        end,
    )
    if piece_start < piece_end:
        spans.append((piece_start, piece_end))

    return spans or [(start, end)]


def split_span_by_token_cap(text, start, end, cap):
    encoded = span_tokenization(text, start, end)
    offsets = encoded["offset_mapping"]
    token_count = len(encoded["input_ids"])

    if token_count <= cap:
        return [end]

    endpoints = []

    for stop in range(cap, token_count, cap):
        local_end = offsets[stop - 1][1]
        endpoints.append(start + local_end)

    endpoints.append(end)

    return sorted(set(endpoints))


def build_char_endpoints(solution_text):
    line_spans = nonempty_line_spans(solution_text)

    if not line_spans:
        raise ValueError("solution_text 没有非空内容")

    is_single_line = len(line_spans) == 1
    cap = single_cap if is_single_line else general_cap

    endpoints = []

    for start, end in line_spans:
        line_tokens = len(
            span_tokenization(
                solution_text,
                start,
                end,
            )["input_ids"]
        )

        if is_single_line or line_tokens > sentence_trigger:
            pieces = sentence_spans(
                solution_text,
                start,
                end,
            )
        else:
            pieces = [(start, end)]

        for piece_start, piece_end in pieces:
            endpoints.extend(
                split_span_by_token_cap(
                    solution_text,
                    piece_start,
                    piece_end,
                    cap,
                )
            )

    endpoints = sorted(set(endpoints))

    if endpoints[-1] != line_spans[-1][1]:
        raise ValueError("最终端点不等于回答最后一个内容字符")

    return endpoints, line_spans[0][0]


def map_endpoints_to_tokens(problem, solution_text, char_ends):
    conversation = [
        {
            "role": "user",
            "content": str(problem),
        },
        {
            "role": "assistant",
            "content": solution_text,
        },
    ]

    rendered = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )

    direct_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    rendered_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if direct_ids != rendered_ids:
        raise ValueError(
            "chat template 两种 tokenization 结果不一致"
        )

    assistant_start = rendered.rfind(solution_text)
    if assistant_start < 0:
        raise ValueError(
            "无法在 chat template 中定位 solution_text"
        )

    token_positions = []
    search_from = 0

    for relative_end in char_ends:
        target_char = assistant_start + relative_end - 1
        found = None

        for position in range(search_from, len(offsets)):
            token_start, token_end = offsets[position]

            if token_start <= target_char < token_end:
                found = position
                break

            if token_start > target_char:
                break

        if found is None:
            raise ValueError(
                f"字符端点无法映射到 token：{relative_end}"
            )

        token_positions.append(found)
        search_from = found

    if token_positions != sorted(set(token_positions)):
        raise ValueError("端点映射出现重复或逆序 token")

    return {
        "input_ids": direct_ids,
        "token_positions": token_positions,
        "assistant_start": assistant_start,
        "offsets": offsets,
    }


def record_sha256(row):
    payload = {
        "question_uid": row.get("question_uid"),
        "candidate_index": row.get("candidate_index"),
        "problem": row.get("problem"),
        "solution_text": row.get("solution_text"),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = {
    "version": "trajectory_endpoint_mapping_v1",
    "tokenizer": str(MODEL_PATH),
    "chunk_config": str(CONFIG_PATH),
    "files": {},
}

for filename in FILES:
    source_path = DATA_ROOT / filename
    output_path = CACHE_ROOT / filename

    candidate_count = 0
    endpoint_counts = []
    input_lengths = []
    actual_chunk_token_lengths = []
    terminal_gap_tokens = []

    with (
        source_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            solution_text = str(row["solution_text"])

            char_ends, first_content_char = (
                build_char_endpoints(solution_text)
            )

            mapping = map_endpoints_to_tokens(
                row["problem"],
                solution_text,
                char_ends,
            )

            input_ids = mapping["input_ids"]
            token_positions = mapping["token_positions"]
            offsets = mapping["offsets"]

            if len(input_ids) > int(
                config["model_sequence_max_tokens"]
            ):
                raise ValueError(
                    f"{filename}:{line_number} 超过序列上限"
                )

            rendered_first_char = (
                mapping["assistant_start"]
                + first_content_char
            )

            first_content_token = None
            for position, (start, end) in enumerate(offsets):
                if start <= rendered_first_char < end:
                    first_content_token = position
                    break

            if first_content_token is None:
                raise ValueError("无法定位回答首 token")

            previous = first_content_token - 1
            chunk_lengths = []

            for position in token_positions:
                chunk_lengths.append(position - previous)
                previous = position

            cache_record = {
                "question_uid": row.get("question_uid"),
                "candidate_index": row.get("candidate_index"),
                "source_dataset": row.get("source_dataset"),
                "label": int(row["label"]),
                "record_sha256": record_sha256(row),
                "input_length": len(input_ids),
                "chunk_count": len(token_positions),
                "chunk_char_ends": char_ends,
                "chunk_token_positions": token_positions,
                "final_content_token_position": (
                    token_positions[-1]
                ),
                "terminal_token_position": len(input_ids) - 1,
            }

            output.write(
                json.dumps(
                    cache_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            candidate_count += 1
            endpoint_counts.append(len(token_positions))
            input_lengths.append(len(input_ids))
            actual_chunk_token_lengths.extend(chunk_lengths)
            terminal_gap_tokens.append(
                len(input_ids) - 1 - token_positions[-1]
            )

    file_report = {
        "source": str(source_path),
        "cache": str(output_path),
        "candidates": candidate_count,
        "total_endpoints": sum(endpoint_counts),
        "chunk_count": describe(endpoint_counts),
        "input_length": describe(input_lengths),
        "actual_chunk_token_length": describe(
            actual_chunk_token_lengths
        ),
        "terminal_gap_tokens": describe(
            terminal_gap_tokens
        ),
        "cache_sha256": file_sha256(output_path),
    }

    manifest["files"][filename] = file_report

    print()
    print("=" * 72)
    print(json.dumps(
        file_report,
        ensure_ascii=False,
        indent=2,
    ))

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("端点映射完成：", MANIFEST_PATH)
