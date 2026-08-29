from pathlib import Path
from collections import defaultdict
import json
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
    "Skywork-Reward-V2-Qwen3-4B"
)

FILES = {
    "GSM8K": DATA / "gsm_train.jsonl",
    "MATH": DATA / "math_train.jsonl",
    "RewardBench2-Math": (
        DATA / "rewardbench2_math_hard_ood.jsonl"
    ),
}


def find_pair(path):
    groups = defaultdict(dict)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["question_uid"]
            label = int(row["label"])

            if label not in groups[qid]:
                groups[qid][label] = row

            if 0 in groups[qid] and 1 in groups[qid]:
                return groups[qid][1], groups[qid][0]

    raise RuntimeError(f"{path} 中没有正负候选对")


pairs = {
    name: find_pair(path)
    for name, path in FILES.items()
}

print("加载 tokenizer……", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("加载 Skywork Reward V2 Qwen3-4B……", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    num_labels=1,
    local_files_only=True,
)
model.eval()

formatted = []
metadata = []

for dataset_name, (positive, negative) in pairs.items():
    for label_name, row in [
        ("positive", positive),
        ("negative", negative),
    ]:
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
        metadata.append({
            "dataset": dataset_name,
            "type": label_name,
            "question_uid": row["question_uid"],
        })

inputs = tokenizer(
    formatted,
    padding=True,
    truncation=True,
    max_length=16384,
    return_tensors="pt",
).to("cuda")

torch.cuda.reset_peak_memory_stats()
started = time.time()

with torch.inference_mode():
    scores = (
        model(**inputs)
        .logits[:, 0]
        .float()
        .cpu()
        .tolist()
    )

elapsed = time.time() - started

for index, score in enumerate(scores):
    metadata[index]["score"] = score
    metadata[index]["tokens"] = int(
        inputs["attention_mask"][index].sum()
    )

results = {}

for dataset_name in pairs:
    positive = next(
        row for row in metadata
        if row["dataset"] == dataset_name
        and row["type"] == "positive"
    )
    negative = next(
        row for row in metadata
        if row["dataset"] == dataset_name
        and row["type"] == "negative"
    )

    results[dataset_name] = {
        "positive_score": positive["score"],
        "negative_score": negative["score"],
        "margin": (
            positive["score"] - negative["score"]
        ),
        "positive_ranked_higher": (
            positive["score"] > negative["score"]
        ),
        "positive_tokens": positive["tokens"],
        "negative_tokens": negative["tokens"],
    }

report = {
    "model": str(MODEL_PATH),
    "inference_seconds": round(elapsed, 4),
    "peak_gpu_gb": round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3,
    ),
    "results": results,
}

output = ROOT / "outputs/reward_smoke_v2.json"
output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== 奖励模型三数据集测试 =====")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("\n输出：", output)
