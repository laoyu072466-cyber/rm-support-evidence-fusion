from pathlib import Path
from collections import defaultdict
import gc
import json
import shutil
import sys
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_sgldsv_current_rm as base


MODEL_PATH = (
    ROOT / "models/reward/"
    "Skywork-Reward-V2-Qwen3-1.7B"
)
DATA_PATH = ROOT / "data/processed/prototype_v2"
CACHE_PATH = (
    ROOT / "data/cache/sgldsv_full_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/block_21"
)
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "sgldsv_full_features_block21_1p7b.json"
)

DATASETS = [
    {
        "prefix": "gsm_train",
        "filename": "gsm_train.jsonl",
        "dataset": "GSM8K",
        "role": "train",
    },
    {
        "prefix": "gsm_pilot",
        "filename": "gsm_pilot_validation.jsonl",
        "dataset": "GSM8K",
        "role": "validation",
    },
    {
        "prefix": "gsm_id_test",
        "filename": "gsm_id_test_mixed.jsonl",
        "dataset": "GSM8K",
        "role": "id_test",
    },
    {
        "prefix": "math_train",
        "filename": "math_train.jsonl",
        "dataset": "MATH",
        "role": "train",
    },
    {
        "prefix": "math_pilot",
        "filename": "math_pilot_validation.jsonl",
        "dataset": "MATH",
        "role": "validation",
    },
    {
        "prefix": "math_id_test",
        "filename": "math_id_test_mixed.jsonl",
        "dataset": "MATH",
        "role": "id_test",
    },
    {
        "prefix": "svamp_ood",
        "filename": "svamp_ood_mixed.jsonl",
        "dataset": "SVAMP",
        "role": "ood_test",
    },
]


def count_questions(rows):
    groups = defaultdict(set)

    for row in rows:
        groups[str(row["question_uid"])].add(
            int(row["label"])
        )

    invalid = [
        uid
        for uid, labels in groups.items()
        if labels != {0, 1}
    ]

    if invalid:
        raise RuntimeError(
            f"发现 {len(invalid)} 个非正负混合问题"
        )

    return len(groups)


def cache_status(prefix):
    paths = base.cache_paths(prefix)
    exists = {
        name: path.exists()
        for name, path in paths.items()
    }

    if all(exists.values()):
        return "complete", paths

    if any(exists.values()):
        missing = [
            name
            for name, present in exists.items()
            if not present
        ]
        raise RuntimeError(
            f"{prefix} 存在不完整缓存，缺少：{missing}。"
            "为防止误覆盖，程序已停止。"
        )

    return "missing", paths


def summarize_existing(prefix, paths):
    features = np.load(
        paths["features"],
        mmap_mode="r",
    )
    offsets = np.load(
        paths["offsets"],
        mmap_mode="r",
    )
    labels = np.load(
        paths["labels"],
        mmap_mode="r",
    )
    scores = np.load(
        paths["scores"],
        mmap_mode="r",
    )

    if len(labels) != len(scores):
        raise RuntimeError(f"{prefix} 标签与分数数量不一致")

    if offsets.ndim == 1:
        candidate_count = len(offsets) - 1
    elif offsets.ndim == 2 and offsets.shape[1] == 2:
        candidate_count = len(offsets)
    else:
        raise RuntimeError(
            f"{prefix} offsets 结构异常：{offsets.shape}"
        )

    if candidate_count != len(labels):
        raise RuntimeError(
            f"{prefix} 候选数量不一致"
        )

    return {
        "prefix": prefix,
        "status": "reused_complete_cache",
        "candidates": int(candidate_count),
        "response_tokens": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "feature_gb": round(
            paths["features"].stat().st_size / 1024 ** 3,
            4,
        ),
        "files": {
            name: str(path)
            for name, path in paths.items()
        },
    }


def main():
    start = time.time()
    base.CACHE_PATH = CACHE_PATH
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    free_gb = shutil.disk_usage(
        ROOT / "data"
    ).free / 1024 ** 3

    print("===== SG-LDSV 全量特征提取 =====")
    print("冻结骨干：", MODEL_PATH)
    print("读取 Block：", base.BLOCK_NUMBER)
    print("当前可用磁盘 GB：", round(free_gb, 2))

    if free_gb < 80:
        raise RuntimeError(
            "可用磁盘不足 80GB，暂不启动全量提取"
        )

    statuses = {}
    pending = []

    for item in DATASETS:
        status, paths = cache_status(item["prefix"])
        statuses[item["prefix"]] = (status, paths)

        if status == "missing":
            pending.append(item)

    tokenizer = None
    backbone = None

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer.padding_side = "right"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not tokenizer.chat_template:
            tokenizer.chat_template = (
                MODEL_PATH / "chat_template.jinja"
            ).read_text(encoding="utf-8")

        print("加载冻结奖励模型……")
        backbone = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
            )
            .to(base.DEVICE)
            .eval()
        )
        backbone.config.use_cache = False
        backbone.config.pad_token_id = (
            tokenizer.pad_token_id
        )

    summaries = {}

    for item in DATASETS:
        prefix = item["prefix"]
        status, paths = statuses[prefix]

        print("\n" + "=" * 72)
        print(
            f"{item['dataset']} / {item['role']} "
            f"-> {item['filename']}"
        )

        if status == "complete":
            print("检测到完整缓存，直接复用。")
            summary = summarize_existing(
                prefix,
                paths,
            )
        else:
            rows = base.read_jsonl(
                DATA_PATH / item["filename"]
            )
            question_count = count_questions(rows)

            print("问题数：", question_count)
            print("候选数：", len(rows))

            summary = base.extract_cache(
                backbone,
                tokenizer,
                rows,
                prefix,
            )
            summary["questions"] = question_count

            del rows
            gc.collect()
            torch.cuda.empty_cache()

        summary["dataset"] = item["dataset"]
        summary["role"] = item["role"]
        summary["source_file"] = item["filename"]
        summaries[prefix] = summary

        print(json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ))

    if backbone is not None:
        del backbone
        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "version": "sgldsv_full_features_v1",
        "backbone": str(MODEL_PATH),
        "backbone_frozen": True,
        "block_number": base.BLOCK_NUMBER,
        "hidden_size": base.HIDDEN_SIZE,
        "feature_dtype": "float16",
        "response_tokens_only": True,
        "datasets": summaries,
        "total_feature_gb": round(
            sum(
                Path(value["files"]["features"])
                .stat().st_size
                for value in summaries.values()
            ) / 1024 ** 3,
            4,
        ),
        "elapsed_seconds": round(
            time.time() - start,
            3,
        ),
        "remaining_disk_gb": round(
            shutil.disk_usage(
                ROOT / "data"
            ).free / 1024 ** 3,
            3,
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("===== SG-LDSV 全量特征提取完成 =====")
    print(json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ))
    print("清单：", MANIFEST_PATH)


if __name__ == "__main__":
    main()
