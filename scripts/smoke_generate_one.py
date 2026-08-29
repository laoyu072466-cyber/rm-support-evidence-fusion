from pathlib import Path
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/root/autodl-tmp/rm_traj_project")
MODEL_PATH = ROOT / "models/generator/Qwen3-4B"
DATA_PATH = ROOT / "data/splits/omni_train.jsonl"
OUTPUT_PATH = ROOT / "data/candidates/smoke_one.json"

with DATA_PATH.open("r", encoding="utf-8") as f:
    sample = json.loads(next(f))

prompt = (
    sample["problem"]
    + "\n\nSolve the problem step by step. "
      "Put the final answer in \\boxed{}."
)

print("正在加载 tokenizer……", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("正在加载 Qwen3-4B 到显卡……", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    local_files_only=True,
)
model.eval()

messages = [{"role": "user", "content": prompt}]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

inputs = tokenizer(
    formatted,
    return_tensors="pt",
).to("cuda")

torch.manual_seed(20260829)
torch.cuda.reset_peak_memory_stats()

print("开始生成一个测试答案……", flush=True)
started = time.time()

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

elapsed = time.time() - started
new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_ids, skip_special_tokens=True)

result = {
    "problem_id": sample["problem_id"],
    "original_index": sample["original_index"],
    "seed": 20260829,
    "model": str(MODEL_PATH),
    "max_new_tokens": 512,
    "generated_tokens": int(new_ids.shape[0]),
    "elapsed_seconds": round(elapsed, 2),
    "response": response,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== 测试结果 =====")
print("问题 ID：", result["problem_id"])
print("生成 token：", result["generated_tokens"])
print("耗时：", result["elapsed_seconds"], "秒")
print(
    "显存峰值：",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)
print("\n===== 模型回答 =====")
print(response)
print("\n结果保存到：", OUTPUT_PATH)
