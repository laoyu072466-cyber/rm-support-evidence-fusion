from pathlib import Path
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/generator/Qwen3-4B"
DATA_PATH = ROOT / "data/splits/omni_train.jsonl"
OUTPUT_PATH = ROOT / "data/candidates/smoke_k8.jsonl"

K = 8
MAX_NEW_TOKENS = 8192
SEED = 20260829

with DATA_PATH.open("r", encoding="utf-8") as f:
    sample = json.loads(next(f))

prompt = (
    sample["problem"]
    + "\n\nSolve the problem step by step. "
      "Put the final answer in \\boxed{}."
)

print("加载 tokenizer……", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("加载 Qwen3-4B……", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    local_files_only=True,
)
model.eval()

formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

inputs = tokenizer(
    formatted,
    return_tensors="pt",
).to("cuda")

torch.manual_seed(SEED)
torch.cuda.reset_peak_memory_stats()

print(
    f"开始生成：1 道题 × {K} 个候选，"
    f"上限 {MAX_NEW_TOKENS} token……",
    flush=True,
)

started = time.time()

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        num_return_sequences=K,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

elapsed = time.time() - started
prompt_length = inputs["input_ids"].shape[1]

eos_ids = model.generation_config.eos_token_id
if isinstance(eos_ids, int):
    eos_ids = {eos_ids}
else:
    eos_ids = set(eos_ids or [])

results = []
lengths = []

for candidate_index, sequence in enumerate(generated):
    token_ids = sequence[prompt_length:].tolist()

    finished_by_eos = False
    trimmed = []

    for token_id in token_ids:
        trimmed.append(token_id)
        if token_id in eos_ids:
            finished_by_eos = True
            break

    response = tokenizer.decode(
        trimmed,
        skip_special_tokens=True,
    )

    lengths.append(len(trimmed))

    results.append({
        "problem_id": sample["problem_id"],
        "original_index": sample["original_index"],
        "candidate_index": candidate_index,
        "seed": SEED,
        "generated_tokens": len(trimmed),
        "finished_by_eos": finished_by_eos,
        "has_think_end": "</think>" in response,
        "has_boxed_answer": "\\boxed" in response,
        "response": response,
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

total_tokens = sum(lengths)

print("\n===== K=8 测试汇总 =====")
print("候选数量：", len(results))
print("正常遇到 EOS：", sum(x["finished_by_eos"] for x in results))
print("包含 </think>：", sum(x["has_think_end"] for x in results))
print("包含 boxed 答案：", sum(x["has_boxed_answer"] for x in results))
print("最短 token：", min(lengths))
print("中位 token：", statistics.median(lengths))
print("最长 token：", max(lengths))
print("总生成 token：", total_tokens)
print("总耗时：", round(elapsed, 2), "秒")
print("总吞吐：", round(total_tokens / elapsed, 2), "token/秒")
print(
    "显存峰值：",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)
print("输出文件：", OUTPUT_PATH)

print("\n===== 每个候选 =====")
for row in results:
    print(
        f'候选 {row["candidate_index"]}: '
        f'{row["generated_tokens"]} token, '
        f'EOS={row["finished_by_eos"]}, '
        f'boxed={row["has_boxed_answer"]}'
    )
