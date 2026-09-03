from pathlib import Path
from collections import defaultdict
import hashlib
import json
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data/processed/prototype_v2"
MODEL_PATH = (
    PROJECT
    / "models/reward/Skywork-Reward-V2-Qwen3-1.7B"
)
MAPPING_ROOT = (
    PROJECT
    / "data/cache/trajectory_chunks_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
)
OUTPUT_ROOT = (
    PROJECT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
SELECTION_PATH = (
    PROJECT
    / "data/manifests/selected_hidden_layer_1p7b.json"
)
BASELINE_PATH = (
    PROJECT / "outputs/b0_pilot_qwen3_1p7b.json"
)
MANIFEST_PATH = (
    PROJECT
    / "data/manifests/trajectory_features_layer28_1p7b.json"
)

FILES = {
    "gsm_train": {
        "filename": "gsm_train.jsonl",
        "dataset": "GSM8K",
        "role": "train",
    },
    "math_train": {
        "filename": "math_train.jsonl",
        "dataset": "MATH",
        "role": "train",
    },
    "gsm_pilot": {
        "filename": "gsm_pilot_validation.jsonl",
        "dataset": "GSM8K",
        "role": "pilot_validation",
    },
    "math_pilot": {
        "filename": "math_pilot_validation.jsonl",
        "dataset": "MATH",
        "role": "pilot_validation",
    },
}

BATCH_SIZE = 16
MAX_LENGTH = 2048
DEVICE = "cuda"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def ranking_metrics(metadata, scores, labels):
    groups = defaultdict(list)

    for index, row in enumerate(metadata):
        groups[row["question_uid"]].append(index)

    top1_values = []
    pair_values = []
    ties = 0
    total_pairs = 0

    for indices in groups.values():
        best_index = max(
            indices,
            key=lambda i: float(scores[i]),
        )
        top1_values.append(
            float(labels[best_index] == 1)
        )

        positives = [
            float(scores[i])
            for i in indices
            if labels[i] == 1
        ]
        negatives = [
            float(scores[i])
            for i in indices
            if labels[i] == 0
        ]

        comparisons = []

        for positive in positives:
            for negative in negatives:
                comparisons.append(
                    float(positive > negative)
                )
                ties += int(positive == negative)
                total_pairs += 1

        if comparisons:
            pair_values.append(
                sum(comparisons) / len(comparisons)
            )

    return {
        "questions": len(groups),
        "candidates": len(metadata),
        "top1": float(np.mean(top1_values)),
        "pair_macro_strict": float(
            np.mean(pair_values)
        ),
        "pair_tie_rate": (
            ties / total_pairs
            if total_pairs else 0.0
        ),
        "positive_score_mean": float(
            np.mean(scores[labels == 1])
        ),
        "negative_score_mean": float(
            np.mean(scores[labels == 0])
        ),
    }


selection = json.loads(
    SELECTION_PATH.read_text(encoding="utf-8")
)
selected_layer = int(selection["selected_layer"])

if selected_layer != 28:
    raise RuntimeError(
        f"预期选择第28层，实际为第{selected_layer}层"
    )

baseline = json.loads(
    BASELINE_PATH.read_text(encoding="utf-8")
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

print("加载 tokenizer……")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("加载 1.7B 奖励模型……")
model = (
    AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    .to(DEVICE)
    .eval()
)

model.config.use_cache = False

hidden_size = int(model.config.hidden_size)

manifest = {
    "version": "trajectory_features_v1",
    "model": str(MODEL_PATH),
    "selected_layer": selected_layer,
    "hidden_size": hidden_size,
    "feature_dtype": "float16",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "scope": "train_and_pilot_only",
    "final_test_used": False,
    "files": {},
}

for file_key, file_config in FILES.items():
    filename = file_config["filename"]
    dataset = file_config["dataset"]
    role = file_config["role"]

    print()
    print("=" * 72)
    print("准备：", filename)

    data_rows = read_jsonl(DATA_ROOT / filename)
    mapping_rows = read_jsonl(MAPPING_ROOT / filename)

    if len(data_rows) != len(mapping_rows):
        raise RuntimeError(
            f"{filename} 数据与映射行数不一致"
        )

    candidate_count = len(data_rows)
    labels = np.empty(candidate_count, dtype=np.int8)
    scores = np.empty(candidate_count, dtype=np.float32)
    endpoint_offsets = np.empty(
        (candidate_count, 2),
        dtype=np.int64,
    )

    metadata = []
    items = []
    endpoint_cursor = 0

    for record_index, (row, mapping) in enumerate(
        zip(data_rows, mapping_rows)
    ):
        if record_sha256(row) != mapping["record_sha256"]:
            raise RuntimeError(
                f"{filename}:{record_index} 哈希不一致"
            )

        conversation = [
            {
                "role": "user",
                "content": str(row["problem"]),
            },
            {
                "role": "assistant",
                "content": str(row["solution_text"]),
            },
        ]

        input_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )

        if len(input_ids) != mapping["input_length"]:
            raise RuntimeError(
                f"{filename}:{record_index} 输入长度不一致"
            )

        if len(input_ids) > MAX_LENGTH:
            raise RuntimeError(
                f"{filename}:{record_index} 超过长度上限"
            )

        positions = [
            int(value)
            for value in mapping[
                "chunk_token_positions"
            ]
        ]

        if len(positions) != mapping["chunk_count"]:
            raise RuntimeError("chunk 数量不一致")

        if positions != sorted(set(positions)):
            raise RuntimeError("chunk token 位置异常")

        endpoint_start = endpoint_cursor
        endpoint_end = endpoint_start + len(positions)
        endpoint_cursor = endpoint_end

        endpoint_offsets[record_index] = [
            endpoint_start,
            endpoint_end,
        ]

        label = int(row["label"])
        labels[record_index] = label

        metadata.append({
            "record_index": record_index,
            "question_uid": row.get("question_uid"),
            "candidate_index": row.get("candidate_index"),
            "source_dataset": row.get("source_dataset"),
            "dataset": dataset,
            "role": role,
            "label": label,
            "input_length": len(input_ids),
            "response_char_length": len(
                str(row["solution_text"])
            ),
            "response_token_length": len(
                tokenizer(
                    str(row["solution_text"]),
                    add_special_tokens=False,
                )["input_ids"]
            ),
            "chunk_count": len(positions),
            "endpoint_start": endpoint_start,
            "endpoint_end": endpoint_end,
        })

        items.append({
            "record_index": record_index,
            "input_ids": input_ids,
            "positions": positions,
            "endpoint_start": endpoint_start,
            "endpoint_end": endpoint_end,
        })

    total_endpoints = endpoint_cursor
    items.sort(key=lambda item: len(item["input_ids"]))

    feature_path = (
        OUTPUT_ROOT / f"{file_key}.chunk_states_f16.npy"
    )
    partial_path = (
        OUTPUT_ROOT
        / f"{file_key}.chunk_states_f16.partial.npy"
    )
    score_path = (
        OUTPUT_ROOT / f"{file_key}.scores_f32.npy"
    )
    label_path = (
        OUTPUT_ROOT / f"{file_key}.labels_i8.npy"
    )
    offset_path = (
        OUTPUT_ROOT / f"{file_key}.offsets_i64.npy"
    )
    metadata_path = (
        OUTPUT_ROOT / f"{file_key}.metadata.jsonl"
    )

    for path in [
        feature_path,
        partial_path,
        score_path,
        label_path,
        offset_path,
        metadata_path,
    ]:
        if path.exists():
            raise FileExistsError(
                f"为避免覆盖已有结果，已停止：{path}"
            )

    feature_memmap = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float16,
        shape=(total_endpoints, hidden_size),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.time()
    processed = 0

    for batch_start in range(
        0,
        candidate_count,
        BATCH_SIZE,
    ):
        batch_items = items[
            batch_start:batch_start + BATCH_SIZE
        ]

        padded = tokenizer.pad(
            {
                "input_ids": [
                    item["input_ids"]
                    for item in batch_items
                ]
            },
            padding=True,
            return_tensors="pt",
        )
        padded = {
            key: value.to(DEVICE)
            for key, value in padded.items()
        }

        with torch.inference_mode():
            outputs = model(
                **padded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        selected_hidden = outputs.hidden_states[
            selected_layer
        ]

        gathered = []

        for local_index, item in enumerate(batch_items):
            positions = torch.tensor(
                item["positions"],
                device=DEVICE,
                dtype=torch.long,
            )
            gathered.append(
                selected_hidden[local_index].index_select(
                    0,
                    positions,
                )
            )

        gathered_tensor = torch.cat(gathered, dim=0)
        gathered_numpy = (
            gathered_tensor
            .to(dtype=torch.float16)
            .cpu()
            .numpy()
        )
        batch_scores = (
            outputs.logits
            .squeeze(-1)
            .float()
            .cpu()
            .numpy()
        )

        local_cursor = 0

        for local_index, item in enumerate(batch_items):
            count = len(item["positions"])
            start = item["endpoint_start"]
            end = item["endpoint_end"]

            feature_memmap[start:end] = (
                gathered_numpy[
                    local_cursor:local_cursor + count
                ]
            )
            scores[item["record_index"]] = (
                batch_scores[local_index]
            )
            local_cursor += count

        if local_cursor != len(gathered_numpy):
            raise RuntimeError("batch 端点写入数量异常")

        processed += len(batch_items)

        if (
            processed == candidate_count
            or processed % 640 < BATCH_SIZE
        ):
            elapsed = time.time() - started
            print(
                f"{filename}: "
                f"{processed}/{candidate_count}, "
                f"{processed / elapsed:.2f} 候选/秒",
                flush=True,
            )

        del (
            outputs,
            selected_hidden,
            gathered,
            gathered_tensor,
            gathered_numpy,
            padded,
        )

    elapsed = time.time() - started

    feature_memmap.flush()
    del feature_memmap
    partial_path.replace(feature_path)

    np.save(score_path, scores)
    np.save(label_path, labels)
    np.save(offset_path, endpoint_offsets)

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    metrics = ranking_metrics(
        metadata,
        scores,
        labels,
    )

    pilot_validation = None

    if role == "pilot_validation":
        expected = baseline["datasets"][dataset]

        top1_difference = (
            metrics["top1"] - expected["top1"]
        )
        pair_difference = (
            metrics["pair_macro_strict"]
            - expected["pair_macro_strict"]
        )

        pilot_validation = {
            "expected_top1": expected["top1"],
            "actual_top1": metrics["top1"],
            "top1_difference": top1_difference,
            "expected_pair_macro_strict": (
                expected["pair_macro_strict"]
            ),
            "actual_pair_macro_strict": (
                metrics["pair_macro_strict"]
            ),
            "pair_difference": pair_difference,
            "passed": (
                abs(top1_difference) <= 0.006
                and abs(pair_difference) <= 0.003
            ),
        }

    report = {
        "file": filename,
        "dataset": dataset,
        "role": role,
        "candidates": candidate_count,
        "total_endpoints": total_endpoints,
        "feature_shape": [
            total_endpoints,
            hidden_size,
        ],
        "feature_file_gb": round(
            feature_path.stat().st_size / 1024**3,
            4,
        ),
        "elapsed_seconds": round(elapsed, 3),
        "candidates_per_second": round(
            candidate_count / elapsed,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            3,
        ),
        "baseline_metrics": metrics,
        "pilot_baseline_reproduction": (
            pilot_validation
        ),
        "files": {
            "features": str(feature_path),
            "scores": str(score_path),
            "labels": str(label_path),
            "offsets": str(offset_path),
            "metadata": str(metadata_path),
        },
        "sha256": {
            "features": file_sha256(feature_path),
            "scores": file_sha256(score_path),
            "labels": file_sha256(label_path),
            "offsets": file_sha256(offset_path),
            "metadata": file_sha256(metadata_path),
        },
    }

    manifest["files"][file_key] = report

    print(json.dumps(
        report,
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
print("===== 第28层轨迹特征提取完成 =====")
print("结果清单：", MANIFEST_PATH)
