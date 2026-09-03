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
    / "data/cache/layer_discovery_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
)
MANIFEST_PATH = (
    PROJECT
    / "data/manifests/layer_discovery_features_1p7b.json"
)

FILES = [
    "gsm_layer_discovery.jsonl",
    "math_layer_discovery.jsonl",
]

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

    for indices in groups.values():
        best_index = max(
            indices,
            key=lambda i: float(scores[i]),
        )
        top1_values.append(int(labels[best_index] == 1))

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

        comparisons = [
            float(positive > negative)
            for positive in positives
            for negative in negatives
        ]

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
        "positive_score_mean": float(
            np.mean(scores[labels == 1])
        ),
        "negative_score_mean": float(
            np.mean(scores[labels == 0])
        ),
    }


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

num_layers = int(model.config.num_hidden_layers)
hidden_size = int(model.config.hidden_size)

if num_layers != 28 or hidden_size != 2048:
    raise RuntimeError(
        f"模型结构异常：layers={num_layers}, "
        f"hidden={hidden_size}"
    )

manifest = {
    "version": "layer_discovery_features_v1",
    "model": str(MODEL_PATH),
    "scope": "layer_discovery_only",
    "representation": (
        "last content chunk hidden state h_T"
    ),
    "cached_layers": list(range(1, num_layers + 1)),
    "dtype": "float16",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "files": {},
}

for filename in FILES:
    print()
    print("=" * 72)
    print("准备：", filename)

    data_rows = read_jsonl(DATA_ROOT / filename)
    mapping_rows = read_jsonl(MAPPING_ROOT / filename)

    if len(data_rows) != len(mapping_rows):
        raise RuntimeError("数据行与端点映射行数不一致")

    candidate_count = len(data_rows)
    items = []
    metadata = []
    labels = np.empty(candidate_count, dtype=np.int8)

    for record_index, (row, mapping) in enumerate(
        zip(data_rows, mapping_rows)
    ):
        if record_sha256(row) != mapping["record_sha256"]:
            raise RuntimeError(
                f"{filename}:{record_index} 内容哈希不一致"
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
                f"{filename}:{record_index} 超过序列上限"
            )

        final_position = int(
            mapping["final_content_token_position"]
        )

        if final_position >= len(input_ids):
            raise RuntimeError("最终内容端点越界")

        label = int(row["label"])
        labels[record_index] = label

        metadata.append({
            "record_index": record_index,
            "question_uid": row.get("question_uid"),
            "candidate_index": row.get("candidate_index"),
            "source_dataset": row.get("source_dataset"),
            "label": label,
            "input_length": len(input_ids),
            "final_content_token_position": final_position,
        })

        items.append({
            "record_index": record_index,
            "input_ids": input_ids,
            "final_position": final_position,
        })

    items.sort(key=lambda item: len(item["input_ids"]))

    stem = filename.removesuffix(".jsonl")

    final_feature_path = (
        OUTPUT_ROOT / f"{stem}.terminal_states_f16.npy"
    )
    partial_feature_path = (
        OUTPUT_ROOT
        / f"{stem}.terminal_states_f16.partial.npy"
    )
    score_path = OUTPUT_ROOT / f"{stem}.scores_f32.npy"
    label_path = OUTPUT_ROOT / f"{stem}.labels_i8.npy"
    metadata_path = OUTPUT_ROOT / f"{stem}.metadata.jsonl"

    for path in [
        final_feature_path,
        partial_feature_path,
        score_path,
        label_path,
        metadata_path,
    ]:
        if path.exists():
            raise FileExistsError(
                f"输出已存在，为防止覆盖已停止：{path}"
            )

    features = np.lib.format.open_memmap(
        partial_feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            num_layers,
            candidate_count,
            hidden_size,
        ),
    )
    scores = np.empty(candidate_count, dtype=np.float32)

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

        final_positions = torch.tensor(
            [
                item["final_position"]
                for item in batch_items
            ],
            dtype=torch.long,
            device=DEVICE,
        )
        batch_indices = torch.arange(
            len(batch_items),
            device=DEVICE,
        )

        with torch.inference_mode():
            outputs = model(
                **padded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        if len(outputs.hidden_states) != num_layers + 1:
            raise RuntimeError("hidden_states 层数异常")

        selected_layers = torch.stack(
            [
                outputs.hidden_states[layer_index][
                    batch_indices,
                    final_positions,
                    :,
                ]
                for layer_index in range(1, num_layers + 1)
            ],
            dim=0,
        )

        selected_numpy = (
            selected_layers
            .to(dtype=torch.float16)
            .cpu()
            .numpy()
        )
        score_numpy = (
            outputs.logits
            .squeeze(-1)
            .float()
            .cpu()
            .numpy()
        )

        for local_index, item in enumerate(batch_items):
            record_index = item["record_index"]
            features[:, record_index, :] = (
                selected_numpy[:, local_index, :]
            )
            scores[record_index] = score_numpy[local_index]

        processed += len(batch_items)

        if (
            processed == candidate_count
            or processed % 320 < BATCH_SIZE
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
            selected_layers,
            selected_numpy,
            padded,
        )

    elapsed = time.time() - started

    features.flush()
    del features

    partial_feature_path.replace(final_feature_path)

    np.save(score_path, scores)
    np.save(label_path, labels)

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    metrics = ranking_metrics(
        metadata,
        scores,
        labels,
    )

    report = {
        "source_file": filename,
        "candidates": candidate_count,
        "feature_shape": [
            num_layers,
            candidate_count,
            hidden_size,
        ],
        "feature_file_gb": round(
            final_feature_path.stat().st_size / 1024**3,
            4,
        ),
        "elapsed_seconds": round(elapsed, 3),
        "candidates_per_second": round(
            candidate_count / elapsed,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
        "baseline_discovery_metrics": metrics,
        "files": {
            "features": str(final_feature_path),
            "scores": str(score_path),
            "labels": str(label_path),
            "metadata": str(metadata_path),
        },
        "sha256": {
            "features": file_sha256(
                final_feature_path
            ),
            "scores": file_sha256(score_path),
            "labels": file_sha256(label_path),
            "metadata": file_sha256(metadata_path),
        },
    }

    manifest["files"][filename] = report

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
print("===== 选层特征提取完成 =====")
print("结果清单：", MANIFEST_PATH)
