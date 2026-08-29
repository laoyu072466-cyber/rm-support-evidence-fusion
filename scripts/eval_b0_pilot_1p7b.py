from pathlib import Path
from collections import defaultdict
import json
import math
import statistics
import time

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path("/root/autodl-tmp/rm_traj_project")
DATA = ROOT / "data/processed/prototype_v2"
MODEL_PATH = (
    ROOT / "models/reward/"
    "Skywork-Reward-V2-Qwen3-1.7B"
)
CACHE = (
    ROOT / "data/cache/b0_pilot/"
    "Skywork-Reward-V2-Qwen3-1.7B"
)
CACHE.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "GSM8K": DATA / "gsm_pilot_validation.jsonl",
    "MATH": DATA / "math_pilot_validation.jsonl",
}

BATCH_SIZE = 32
MAX_LENGTH = 16384


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary.replace(path)


def compute_metrics(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["question_uid"]].append(row)

    top1 = []
    top1_tie_aware = []
    pair_macro_strict = []
    pair_macro_half_ties = []

    total_pairs = 0
    total_ties = 0

    for candidates in groups.values():
        best_score = max(
            row["reward_score"] for row in candidates
        )
        best = [
            row for row in candidates
            if row["reward_score"] == best_score
        ]

        best.sort(
            key=lambda row: row.get("candidate_index", 0)
        )

        top1.append(int(best[0]["label"]))
        top1_tie_aware.append(
            sum(int(row["label"]) for row in best)
            / len(best)
        )

        positives = [
            row["reward_score"]
            for row in candidates
            if int(row["label"]) == 1
        ]
        negatives = [
            row["reward_score"]
            for row in candidates
            if int(row["label"]) == 0
        ]

        wins = 0
        ties = 0
        pair_count = len(positives) * len(negatives)

        for positive in positives:
            for negative in negatives:
                if positive > negative:
                    wins += 1
                elif positive == negative:
                    ties += 1

        pair_macro_strict.append(
            wins / pair_count
        )
        pair_macro_half_ties.append(
            (wins + 0.5 * ties) / pair_count
        )

        total_pairs += pair_count
        total_ties += ties

    positive_scores = [
        row["reward_score"] for row in rows
        if int(row["label"]) == 1
    ]
    negative_scores = [
        row["reward_score"] for row in rows
        if int(row["label"]) == 0
    ]

    return {
        "questions": len(groups),
        "candidates": len(rows),
        "top1": statistics.mean(top1),
        "top1_tie_aware": statistics.mean(
            top1_tie_aware
        ),
        "pair_macro_strict": statistics.mean(
            pair_macro_strict
        ),
        "pair_macro_half_ties": statistics.mean(
            pair_macro_half_ties
        ),
        "pair_tie_rate": (
            total_ties / total_pairs
            if total_pairs else 0.0
        ),
        "positive_score_mean": statistics.mean(
            positive_scores
        ),
        "negative_score_mean": statistics.mean(
            negative_scores
        ),
    }


print("加载 tokenizer……", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("加载奖励模型……", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    num_labels=1,
    local_files_only=True,
)
model.eval()

all_metrics = {}
overall_started = time.time()

for dataset_name, data_path in DATASETS.items():
    rows = read_jsonl(data_path)
    formatted = []

    print(
        f"\n开始评分 {dataset_name}："
        f"{len(rows)} 个候选",
        flush=True,
    )

    for row in rows:
        conversation = [
            {
                "role": "user",
                "content": row["problem"],
            },
            {
                "role": "assistant",
                "content": row["solution_text"],
            },
        ]

        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
        )

        if (
            tokenizer.bos_token is not None
            and text.startswith(tokenizer.bos_token)
        ):
            text = text[len(tokenizer.bos_token):]

        formatted.append(text)

    # 按文本长度排序，减少批次内补齐浪费
    order = sorted(
        range(len(rows)),
        key=lambda index: len(formatted[index]),
    )

    scores = [None] * len(rows)
    token_lengths = [None] * len(rows)
    started = time.time()

    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        texts = [formatted[index] for index in indices]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            batch_scores = (
                model(**inputs)
                .logits[:, 0]
                .float()
                .cpu()
                .tolist()
            )

        lengths = (
            inputs["attention_mask"]
            .sum(dim=1)
            .cpu()
            .tolist()
        )

        for index, score, length in zip(
            indices, batch_scores, lengths
        ):
            if not math.isfinite(score):
                raise RuntimeError("发现非有限 reward score")
            scores[index] = score
            token_lengths[index] = int(length)

        completed = min(start + BATCH_SIZE, len(order))

        if (
            completed == len(order)
            or completed % 640 == 0
        ):
            print(
                f"{dataset_name}: "
                f"{completed}/{len(order)}",
                flush=True,
            )

    for row, score, token_length in zip(
        rows, scores, token_lengths
    ):
        row["reward_score"] = score
        row["rm_input_tokens"] = token_length
        row["reward_model"] = str(MODEL_PATH)

    elapsed = time.time() - started
    metrics = compute_metrics(rows)
    metrics["elapsed_seconds"] = round(elapsed, 3)
    metrics["candidates_per_second"] = round(
        len(rows) / elapsed, 3
    )

    all_metrics[dataset_name] = metrics

    write_jsonl(
        CACHE / f"{dataset_name.lower()}_scores.jsonl",
        rows,
    )

    print(
        json.dumps(
            {dataset_name: metrics},
            ensure_ascii=False,
            indent=2,
        )
    )

report = {
    "model": str(MODEL_PATH),
    "evaluation_scope": "pilot_validation_only",
    "batch_size": BATCH_SIZE,
    "datasets": all_metrics,
    "dataset_macro": {
        "top1": statistics.mean(
            value["top1"]
            for value in all_metrics.values()
        ),
        "pair_macro_strict": statistics.mean(
            value["pair_macro_strict"]
            for value in all_metrics.values()
        ),
    },
    "total_elapsed_seconds": round(
        time.time() - overall_started, 3
    ),
}

output = ROOT / "outputs/b0_pilot_qwen3_1p7b.json"
output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== B0 Pilot 最终结果 =====")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("\n结果文件：", output)
