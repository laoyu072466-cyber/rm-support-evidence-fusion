from pathlib import Path
import gc
import json
import random
import sys
import time

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_sgldsv_current_rm as base


CACHE_PATH = (
    ROOT / "data/cache/sgldsv_full_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/block_21"
)
OUTPUT_DIR = ROOT / "outputs/sgldsv_full"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints"
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "sgldsv_full_separate_training_1p7b.json"
)

EXPERIMENTS = {
    "GSM8K": {
        "train_prefix": "gsm_train",
        "validation_prefix": "gsm_pilot",
        "test_prefixes": [
            "gsm_id_test",
            "svamp_ood",
        ],
    },
    "MATH": {
        "train_prefix": "math_train",
        "validation_prefix": "math_pilot",
        "test_prefixes": [
            "math_id_test",
        ],
    },
}

SEEDS = [42, 123, 456]


def set_seed(seed):
    base.SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_paths(dataset, seed):
    name = dataset.lower()

    result_path = (
        OUTPUT_DIR
        / f"sgldsv_{name}_seed{seed}.json"
    )
    checkpoint_path = (
        CHECKPOINT_DIR
        / f"sgldsv_{name}_seed{seed}.pt"
    )

    return result_path, checkpoint_path


def main():
    overall_start = time.time()

    base.CACHE_PATH = CACHE_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===== SG-LDSV 正式分数据集训练 =====")
    print("缓存：", CACHE_PATH)
    print("数据集：GSM8K、MATH")
    print("随机种子：", SEEDS)
    print("选择指标：验证集 Pair-Macro Strict")
    print("最大 epoch：", base.MAX_EPOCHS)
    print("patience：", base.PATIENCE)
    print("每题正负对上限：", base.PAIR_CAP_PER_QUESTION)

    all_results = {}

    for dataset, config in EXPERIMENTS.items():
        print("\n" + "=" * 72)
        print("加载数据集：", dataset)

        train_cache = base.CachedDataset(
            config["train_prefix"]
        )
        validation_cache = base.CachedDataset(
            config["validation_prefix"]
        )

        print(
            "训练候选：",
            len(train_cache.labels),
        )
        print(
            "验证候选：",
            len(validation_cache.labels),
        )

        dataset_results = {}

        for seed in SEEDS:
            result_path, checkpoint_path = run_paths(
                dataset,
                seed,
            )

            result_exists = result_path.exists()
            checkpoint_exists = checkpoint_path.exists()

            if result_exists and checkpoint_exists:
                print("\n" + "-" * 72)
                print(
                    f"{dataset} seed={seed} "
                    "检测到完整结果，直接复用。"
                )
                saved = json.loads(
                    result_path.read_text(
                        encoding="utf-8"
                    )
                )
                dataset_results[str(seed)] = saved
                continue

            if result_exists != checkpoint_exists:
                raise RuntimeError(
                    f"{dataset} seed={seed} 只存在部分结果。"
                    "为防止覆盖，程序已停止。"
                )

            print("\n" + "-" * 72)
            print(
                f"开始训练 {dataset}，seed={seed}"
            )

            set_seed(seed)
            base.CHECKPOINT_PATH = checkpoint_path
            base.OUTPUT_PATH = result_path

            run_start = time.time()
            torch.cuda.reset_peak_memory_stats()

            training = base.train_smoke(
                train_cache,
                validation_cache,
            )

            run_result = {
                "version": "sgldsv_full_separate_v1",
                "dataset": dataset,
                "seed": seed,
                "training_mode": (
                    "dataset_specific_separate_head"
                ),
                "train_prefix": config["train_prefix"],
                "validation_prefix": (
                    config["validation_prefix"]
                ),
                "reserved_test_prefixes": (
                    config["test_prefixes"]
                ),
                "test_used_during_training": False,
                "backbone": (
                    "Skywork-Reward-V2-Qwen3-1.7B"
                ),
                "block_number": base.BLOCK_NUMBER,
                "architecture": (
                    "exact_SG-LDSV_token_level_adaptation"
                ),
                "training": training,
                "elapsed_seconds": round(
                    time.time() - run_start,
                    3,
                ),
                "peak_gpu_gb": round(
                    torch.cuda.max_memory_allocated()
                    / 1024 ** 3,
                    3,
                ),
                "checkpoint": str(checkpoint_path),
            }

            result_path.write_text(
                json.dumps(
                    run_result,
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

            dataset_results[str(seed)] = run_result

            print(
                f"{dataset} seed={seed} 完成：",
                result_path,
            )

            gc.collect()
            torch.cuda.empty_cache()

        all_results[dataset] = dataset_results

        del train_cache
        del validation_cache
        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "version": "sgldsv_full_separate_training_v1",
        "protocol": {
            "separate_heads": True,
            "datasets": ["GSM8K", "MATH"],
            "seeds": SEEDS,
            "validation_selection": (
                "pair_macro_strict"
            ),
            "max_epochs": base.MAX_EPOCHS,
            "early_stopping_patience": base.PATIENCE,
            "pair_cap_per_question": (
                base.PAIR_CAP_PER_QUESTION
            ),
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
            "gradient_clip": base.GRAD_CLIP,
            "test_used_during_training": False,
        },
        "runs": all_results,
        "total_elapsed_seconds": round(
            time.time() - overall_start,
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
    print("===== 六组正式训练全部完成 =====")
    print("训练清单：", MANIFEST_PATH)
    print(
        "总耗时秒：",
        manifest["total_elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
