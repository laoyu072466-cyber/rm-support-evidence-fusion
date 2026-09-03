from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModel


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import score_full_candidates_multi_reward as base


base.CONFIG_PATH = (
    ROOT / "configs/"
    "internlm2_reward_full_scoring_v1.json"
)


def load_model(model_path, tokenizer):
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if not hasattr(model, "get_scores"):
        raise RuntimeError(
            "InternLM 模型缺少官方 get_scores 接口"
        )

    return model


def score_batch(
    model,
    tokenizer,
    rows,
    indices,
):
    conversations = []

    for index in indices:
        row = rows[int(index)]
        conversations.append([
            {
                "role": "user",
                "content": str(row["problem"]),
            },
            {
                "role": "assistant",
                "content": str(
                    row["solution_text"]
                ),
            },
        ])

    with torch.inference_mode():
        values = model.get_scores(
            tokenizer,
            conversations,
        )

    scores = np.asarray(
        values,
        dtype=np.float32,
    )

    if scores.ndim == 0:
        scores = scores.reshape(1)
    else:
        scores = scores.reshape(-1)

    if len(scores) != len(indices):
        raise RuntimeError(
            "InternLM 返回数量错误："
            f"{len(scores)} != {len(indices)}"
        )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            "InternLM 产生非有限奖励分数"
        )

    return scores


base.load_model = load_model
base.score_batch = score_batch


if __name__ == "__main__":
    base.main()
