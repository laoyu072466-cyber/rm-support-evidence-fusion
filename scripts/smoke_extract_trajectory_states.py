from pathlib import Path
import json

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT = Path("/root/autodl-tmp/rm_traj_project")
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

FILES = [
    "gsm_layer_discovery.jsonl",
    "math_layer_discovery.jsonl",
]

SAMPLES_PER_FILE = 4


def read_first(path, count):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == count:
                break
    return rows


print("加载 tokenizer……")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

items = []

for filename in FILES:
    data_rows = read_first(
        DATA_ROOT / filename,
        SAMPLES_PER_FILE,
    )
    mapping_rows = read_first(
        MAPPING_ROOT / filename,
        SAMPLES_PER_FILE,
    )

    if len(data_rows) != len(mapping_rows):
        raise RuntimeError("数据与端点映射数量不一致")

    for row, mapping in zip(data_rows, mapping_rows):
        if (
            row.get("question_uid")
            != mapping.get("question_uid")
        ):
            raise RuntimeError("question_uid 不一致")

        if (
            row.get("candidate_index")
            != mapping.get("candidate_index")
        ):
            raise RuntimeError("candidate_index 不一致")

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
            raise RuntimeError("模型输入长度与映射不一致")

        if (
            max(mapping["chunk_token_positions"])
            >= len(input_ids)
        ):
            raise RuntimeError("chunk token 位置越界")

        items.append({
            "dataset": filename,
            "question_uid": row.get("question_uid"),
            "candidate_index": row.get("candidate_index"),
            "label": int(row["label"]),
            "input_ids": input_ids,
            "chunk_positions": (
                mapping["chunk_token_positions"]
            ),
            "terminal_position": (
                mapping["terminal_token_position"]
            ),
        })

batch = tokenizer.pad(
    {
        "input_ids": [
            item["input_ids"]
            for item in items
        ]
    },
    padding=True,
    return_tensors="pt",
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("加载 1.7B 奖励模型……")
model = (
    AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)

batch = {
    key: value.to("cuda")
    for key, value in batch.items()
}

print("执行逐层 hidden-state 冒烟测试……")
with torch.inference_mode():
    outputs = model(
        **batch,
        output_hidden_states=True,
        return_dict=True,
    )

hidden_states = outputs.hidden_states
scores = outputs.logits.squeeze(-1)

print()
print("===== 模型输出结构 =====")
print("样本数：", len(items))
print("hidden_states 数量：", len(hidden_states))
print("embedding shape：", tuple(hidden_states[0].shape))
print("第 1 层 shape：", tuple(hidden_states[1].shape))
print("第 14 层 shape：", tuple(hidden_states[14].shape))
print("第 28 层 shape：", tuple(hidden_states[28].shape))
print(
    "显存峰值 GB：",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        3,
    ),
)

print()
print("===== 每条候选端点检查 =====")

for batch_index, item in enumerate(items):
    positions = torch.tensor(
        item["chunk_positions"],
        device="cuda",
        dtype=torch.long,
    )

    layer_1 = hidden_states[1][
        batch_index,
        positions,
        :,
    ]
    layer_14 = hidden_states[14][
        batch_index,
        positions,
        :,
    ]
    layer_28 = hidden_states[28][
        batch_index,
        positions,
        :,
    ]

    print({
        "dataset": item["dataset"],
        "question_uid": item["question_uid"],
        "candidate_index": item["candidate_index"],
        "label": item["label"],
        "input_tokens": len(item["input_ids"]),
        "chunks": len(positions),
        "score": round(
            float(scores[batch_index].item()),
            6,
        ),
        "layer_1_mean_norm": round(
            float(layer_1.float().norm(
                dim=-1
            ).mean().item()),
            4,
        ),
        "layer_14_mean_norm": round(
            float(layer_14.float().norm(
                dim=-1
            ).mean().item()),
            4,
        ),
        "layer_28_mean_norm": round(
            float(layer_28.float().norm(
                dim=-1
            ).mean().item()),
            4,
        ),
        "terminal_gap": (
            item["terminal_position"]
            - item["chunk_positions"][-1]
        ),
    })

expected_hidden_state_count = (
    model.config.num_hidden_layers + 1
)

if len(hidden_states) != expected_hidden_state_count:
    raise RuntimeError(
        "hidden_states 层数与配置不一致"
    )

print()
print("逐层隐藏状态冒烟测试通过。")
