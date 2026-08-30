from pathlib import Path
from collections import defaultdict
import copy
import gc
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

ROOT = Path("/root/autodl-tmp/rm_traj_project")
CACHE_PATH = (
    ROOT / "data/cache/sgldsv_full_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/block_21"
)
OUTPUT_DIR = ROOT / "outputs/sgldsv_rm_residual"
CHECKPOINT_DIR = ROOT / "outputs/checkpoints"
MANIFEST_PATH = (
    ROOT / "data/manifests/"
    "sgldsv_rm_residual_pilot_training.json"
)

DEVICE = torch.device("cuda")
SEEDS = [42, 123, 456]

FAMILIES = {
    "GSM8K": {
        "train": "gsm_train",
        "validation": "gsm_pilot",
    },
    "MATH": {
        "train": "math_train",
        "validation": "math_pilot",
    },
}

HIDDEN_SIZE = 2048
PROJECTION_SIZE = 128
TRANSITION_SIZE = 32
LOCAL_HIDDEN_SIZE = 96

PAIR_CAP = 256
PAIR_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 48
MAX_EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

CORRECTION_CAP = 0.5
RESIDUAL_PENALTY = 0.1
TOP1_DAMAGE_GUARD = 0.005


def paths(prefix):
    return {
        "features": CACHE_PATH / (
            f"{prefix}.response_states_f16.npy"
        ),
        "offsets": CACHE_PATH / (
            f"{prefix}.offsets_i64.npy"
        ),
        "scores": CACHE_PATH / (
            f"{prefix}.rm_scores_f32.npy"
        ),
        "labels": CACHE_PATH / (
            f"{prefix}.labels_i8.npy"
        ),
        "metadata": CACHE_PATH / (
            f"{prefix}.metadata.jsonl"
        ),
    }


class CachedResponses:
    def __init__(self, prefix):
        self.prefix = prefix
        current = paths(prefix)

        for path in current.values():
            if not path.exists():
                raise FileNotFoundError(path)

        self.features = np.load(
            current["features"],
            mmap_mode="r",
        )
        self.offsets = np.load(
            current["offsets"],
            mmap_mode="r",
        )
        self.rm_scores = np.load(
            current["scores"],
        ).astype(np.float32)
        self.labels = np.load(
            current["labels"],
        ).astype(np.int8)

        self.metadata = []
        with current["metadata"].open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if line.strip():
                    self.metadata.append(
                        json.loads(line)
                    )

        if not (
            len(self.labels)
            == len(self.rm_scores)
            == len(self.metadata)
        ):
            raise RuntimeError(
                f"{prefix} 候选数量不一致"
            )

        self.groups = defaultdict(list)
        for index, row in enumerate(self.metadata):
            self.groups[
                str(row["question_uid"])
            ].append(index)

        for uid, indices in self.groups.items():
            label_set = {
                int(self.labels[index])
                for index in indices
            }
            if label_set != {0, 1}:
                raise RuntimeError(
                    f"{prefix}:{uid} 不是正负混合问题"
                )

    def __len__(self):
        return len(self.labels)

    def offset_range(self, index):
        if self.offsets.ndim == 1:
            return (
                int(self.offsets[index]),
                int(self.offsets[index + 1]),
            )

        if (
            self.offsets.ndim == 2
            and self.offsets.shape[1] == 2
        ):
            return (
                int(self.offsets[index, 0]),
                int(self.offsets[index, 1]),
            )

        raise RuntimeError(
            f"offsets 结构异常：{self.offsets.shape}"
        )

    def make_batch(self, indices):
        sequences = []

        for index in indices:
            start, end = self.offset_range(int(index))
            array = np.asarray(
                self.features[start:end],
                dtype=np.float32,
            ).copy()

            if len(array) == 0:
                raise RuntimeError(
                    f"{self.prefix}:{index} token 为空"
                )

            sequences.append(
                torch.from_numpy(array)
            )

        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            device=DEVICE,
        )

        padded = pad_sequence(
            sequences,
            batch_first=True,
        ).to(
            device=DEVICE,
            dtype=torch.float32,
        )

        positions = torch.arange(
            padded.shape[1],
            device=DEVICE,
        )
        mask = (
            positions.unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        return padded, mask


class RMPreservingLDSV(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = nn.Linear(
            HIDDEN_SIZE,
            PROJECTION_SIZE,
            bias=False,
        )
        self.projection_norm = nn.LayerNorm(
            PROJECTION_SIZE,
            elementwise_affine=False,
        )
        self.transition = nn.Linear(
            PROJECTION_SIZE,
            TRANSITION_SIZE,
        )
        self.source_gate = nn.Linear(
            PROJECTION_SIZE,
            1,
        )
        self.local_norm = nn.LayerNorm(
            TRANSITION_SIZE * 2,
            elementwise_affine=False,
        )
        self.local_hidden = nn.Linear(
            TRANSITION_SIZE * 2,
            LOCAL_HIDDEN_SIZE,
        )
        self.local_output = nn.Linear(
            LOCAL_HIDDEN_SIZE,
            1,
        )

        # 初始残差严格为 0，初始模型等于原始 RM。
        nn.init.zeros_(self.local_output.weight)
        nn.init.zeros_(self.local_output.bias)

    def local_score(self, hidden_states, mask):
        projected = self.projection_norm(
            self.projection(hidden_states)
        )

        source = projected[:, :-1]
        differences = (
            projected[:, 1:]
            - projected[:, :-1]
        )
        transition_mask = (
            mask[:, 1:]
            & mask[:, :-1]
        )

        embeddings = F.relu(
            self.transition(differences)
        )
        weights = torch.sigmoid(
            self.source_gate(source)
        ).squeeze(-1)
        weights = (
            weights
            * transition_mask.to(weights.dtype)
        )

        denominator = (
            weights.sum(dim=1, keepdim=True)
            .clamp_min(1e-8)
        )

        mean = (
            embeddings
            * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator

        centered = (
            embeddings
            - mean.unsqueeze(1)
        )
        variance = (
            centered.square()
            * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator

        std = torch.sqrt(
            variance.clamp_min(0.0) + 1e-8
        )

        statistics = torch.cat(
            [mean, std],
            dim=-1,
        )
        local = self.local_output(
            F.relu(
                self.local_hidden(
                    self.local_norm(statistics)
                )
            )
        ).squeeze(-1)

        valid = transition_mask.any(dim=1)
        local = torch.where(
            valid,
            local,
            torch.zeros_like(local),
        )

        return local

    def forward(self, hidden_states, mask, raw_z):
        local = self.local_score(
            hidden_states,
            mask,
        )
        correction = (
            CORRECTION_CAP
            * torch.tanh(local)
        )
        final = raw_z + correction

        return final, correction, local


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pairs(cache, seed):
    rng = np.random.default_rng(seed)
    all_pairs = []

    for indices in cache.groups.values():
        positives = [
            index
            for index in indices
            if cache.labels[index] == 1
        ]
        negatives = [
            index
            for index in indices
            if cache.labels[index] == 0
        ]

        pairs = np.asarray([
            (positive, negative)
            for positive in positives
            for negative in negatives
        ], dtype=np.int64)

        if len(pairs) > PAIR_CAP:
            chosen = rng.choice(
                len(pairs),
                size=PAIR_CAP,
                replace=False,
            )
            pairs = pairs[chosen]

        all_pairs.append(pairs)

    return np.concatenate(all_pairs, axis=0)


def ranking_metrics(cache, scores):
    top1_values = []
    pair_values = []

    for indices in cache.groups.values():
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        labels = cache.labels[indices]
        local_scores = scores[indices]

        top_index = indices[
            int(np.argmax(local_scores))
        ]
        top1_values.append(
            float(cache.labels[top_index] == 1)
        )

        positive_scores = local_scores[
            labels == 1
        ]
        negative_scores = local_scores[
            labels == 0
        ]

        pair_values.append(float(
            (
                positive_scores[:, None]
                > negative_scores[None, :]
            ).mean()
        ))

    return {
        "questions": len(cache.groups),
        "candidates": len(cache),
        "top1": float(np.mean(top1_values)),
        "pair_macro_strict": float(
            np.mean(pair_values)
        ),
        "positive_score_mean": float(
            scores[cache.labels == 1].mean()
        ),
        "negative_score_mean": float(
            scores[cache.labels == 0].mean()
        ),
    }


def transition_metrics(cache, raw_scores, new_scores):
    raw_correct = 0
    raw_wrong = 0
    corrected = 0
    damaged = 0

    for indices in cache.groups.values():
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        raw_choice = indices[
            np.argmax(raw_scores[indices])
        ]
        new_choice = indices[
            np.argmax(new_scores[indices])
        ]

        raw_ok = cache.labels[raw_choice] == 1
        new_ok = cache.labels[new_choice] == 1

        if raw_ok:
            raw_correct += 1
            if not new_ok:
                damaged += 1
        else:
            raw_wrong += 1
            if new_ok:
                corrected += 1

    return {
        "correction_rate": (
            corrected / raw_wrong
            if raw_wrong else 0.0
        ),
        "damage_rate": (
            damaged / raw_correct
            if raw_correct else 0.0
        ),
        "corrected_questions": corrected,
        "damaged_questions": damaged,
    }


@torch.inference_mode()
def predict(model, cache, reward_mean, reward_std):
    scores = np.empty(
        len(cache),
        dtype=np.float32,
    )
    corrections = np.empty(
        len(cache),
        dtype=np.float32,
    )

    model.eval()

    for start in range(
        0,
        len(cache),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(cache),
        )
        indices = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        hidden, mask = cache.make_batch(indices)
        raw = torch.from_numpy(
            cache.rm_scores[indices]
        ).to(
            device=DEVICE,
            dtype=torch.float32,
        )
        raw_z = (
            raw - reward_mean
        ) / reward_std

        final, correction, _ = model(
            hidden,
            mask,
            raw_z,
        )

        scores[start:end] = (
            final.cpu().numpy()
        )
        corrections[start:end] = (
            correction.cpu().numpy()
        )

    return scores, corrections


def train_one(
    family,
    seed,
    train_cache,
    validation_cache,
):
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    reward_mean = float(
        train_cache.rm_scores.mean()
    )
    reward_std = float(
        train_cache.rm_scores.std()
    )

    if reward_std < 1e-8:
        raise RuntimeError("奖励标准差异常")

    pairs = build_pairs(
        train_cache,
        seed,
    )
    rng = np.random.default_rng(seed)

    model = RMPreservingLDSV().to(DEVICE)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if parameter_count != 272738:
        raise RuntimeError(
            f"参数量异常：{parameter_count}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    validation_raw_z = (
        validation_cache.rm_scores
        - reward_mean
    ) / reward_std
    raw_metrics = ranking_metrics(
        validation_cache,
        validation_raw_z,
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )
    best_epoch = 0
    best_pair = raw_metrics[
        "pair_macro_strict"
    ]
    best_metrics = raw_metrics
    best_transition = {
        "correction_rate": 0.0,
        "damage_rate": 0.0,
        "corrected_questions": 0,
        "damaged_questions": 0,
    }
    best_correction_abs_mean = 0.0
    patience_count = 0
    history = []

    print("\n" + "-" * 72)
    print(f"{family} seed={seed}")
    print("训练正负对：", len(pairs))
    print("可训练参数：", parameter_count)
    print("Pilot 原始 RM：")
    print(json.dumps(
        raw_metrics,
        ensure_ascii=False,
        indent=2,
    ))

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        losses = []

        for start in range(
            0,
            len(shuffled),
            PAIR_BATCH_SIZE,
        ):
            batch_pairs = shuffled[
                start:start + PAIR_BATCH_SIZE
            ]

            positive_indices = batch_pairs[:, 0]
            negative_indices = batch_pairs[:, 1]
            combined = np.concatenate([
                positive_indices,
                negative_indices,
            ])

            hidden, mask = train_cache.make_batch(
                combined
            )
            raw = torch.from_numpy(
                train_cache.rm_scores[combined]
            ).to(
                device=DEVICE,
                dtype=torch.float32,
            )
            raw_z = (
                raw - reward_mean
            ) / reward_std

            final, correction, _ = model(
                hidden,
                mask,
                raw_z,
            )

            count = len(batch_pairs)
            positive_scores = final[:count]
            negative_scores = final[count:]

            bt_loss = F.softplus(
                -(
                    positive_scores
                    - negative_scores
                )
            ).mean()

            penalty = correction.square().mean()
            loss = (
                bt_loss
                + RESIDUAL_PENALTY * penalty
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            optimizer.step()

            losses.append(float(loss.item()))

        validation_scores, corrections = predict(
            model,
            validation_cache,
            reward_mean,
            reward_std,
        )
        metrics = ranking_metrics(
            validation_cache,
            validation_scores,
        )
        transitions = transition_metrics(
            validation_cache,
            validation_raw_z,
            validation_scores,
        )
        correction_abs_mean = float(
            np.abs(corrections).mean()
        )

        eligible = (
            metrics["top1"]
            >= raw_metrics["top1"]
            - TOP1_DAMAGE_GUARD
        )
        improved = (
            eligible
            and metrics["pair_macro_strict"]
            > best_pair
        )

        if improved:
            best_pair = metrics[
                "pair_macro_strict"
            ]
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_metrics = metrics
            best_transition = transitions
            best_correction_abs_mean = (
                correction_abs_mean
            )
            patience_count = 0
        else:
            patience_count += 1

        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation": metrics,
            "transition": transitions,
            "correction_abs_mean": (
                correction_abs_mean
            ),
            "eligible": eligible,
            "selected": improved,
        })

        print(
            f"epoch {epoch}/{MAX_EPOCHS} | "
            f"loss={np.mean(losses):.6f} | "
            f"Top1={metrics['top1']:.6f} | "
            f"Pair={metrics['pair_macro_strict']:.6f} | "
            f"Damage={transitions['damage_rate']:.6f} | "
            f"|delta|={correction_abs_mean:.6f} | "
            f"selected={improved}"
        )

        if patience_count >= PATIENCE:
            print("触发 early stopping。")
            break

    model.load_state_dict(best_state)

    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            f"sgldsv_rm_residual_"
            f"{family.lower()}_seed{seed}.pt"
        )
    )
    torch.save({
        "model_state_dict": best_state,
        "family": family,
        "seed": seed,
        "best_epoch": best_epoch,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "correction_cap": CORRECTION_CAP,
        "residual_penalty": RESIDUAL_PENALTY,
    }, checkpoint_path)

    result = {
        "family": family,
        "seed": seed,
        "best_epoch": best_epoch,
        "fallback_to_raw_rm": best_epoch == 0,
        "raw_pilot": raw_metrics,
        "best_pilot": best_metrics,
        "best_transition": best_transition,
        "best_correction_abs_mean": (
            best_correction_abs_mean
        ),
        "history": history,
        "reward_normalization": {
            "mean": reward_mean,
            "std": reward_std,
            "scope": "train_only",
        },
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint_path),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        ),
    }

    result_path = (
        OUTPUT_DIR
        / (
            f"sgldsv_rm_residual_"
            f"{family.lower()}_seed{seed}.json"
        )
    )
    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.set_float32_matmul_precision("high")

    print("===== RM-Preserving SG-LDSV Residual =====")
    print("测试集不会在本脚本中加载。")
    print("修正上限：", CORRECTION_CAP)
    print("残差惩罚：", RESIDUAL_PENALTY)
    print("Top1 损伤保护：", TOP1_DAMAGE_GUARD)

    results = {}

    for family, config in FAMILIES.items():
        print("\n" + "=" * 72)
        print("加载：", family)

        train_cache = CachedResponses(
            config["train"]
        )
        validation_cache = CachedResponses(
            config["validation"]
        )

        results[family] = {}

        for seed in SEEDS:
            result_path = (
                OUTPUT_DIR
                / (
                    f"sgldsv_rm_residual_"
                    f"{family.lower()}_seed{seed}.json"
                )
            )
            checkpoint_path = (
                CHECKPOINT_DIR
                / (
                    f"sgldsv_rm_residual_"
                    f"{family.lower()}_seed{seed}.pt"
                )
            )

            if (
                result_path.exists()
                and checkpoint_path.exists()
            ):
                print(
                    f"{family} seed={seed} "
                    "检测到完整结果，直接复用。"
                )
                result = json.loads(
                    result_path.read_text(
                        encoding="utf-8"
                    )
                )
            elif (
                result_path.exists()
                or checkpoint_path.exists()
            ):
                raise RuntimeError(
                    f"{family} seed={seed} "
                    "存在不完整结果，停止覆盖"
                )
            else:
                result = train_one(
                    family,
                    seed,
                    train_cache,
                    validation_cache,
                )

            results[family][str(seed)] = result

        del train_cache
        del validation_cache
        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "version": "sgldsv_rm_residual_pilot_v1",
        "scope": (
            "exploratory_after_exact_transfer_failure"
        ),
        "tests_loaded": False,
        "architecture": {
            "endpoint": "frozen_original_RM_score",
            "local_branch": (
                "SG-LDSV gated local dynamics"
            ),
            "initial_correction": 0.0,
            "correction": (
                "0.5 * tanh(local_score)"
            ),
            "residual_penalty": RESIDUAL_PENALTY,
        },
        "selection": {
            "primary": "pilot_pair_macro_strict",
            "top1_damage_guard": (
                TOP1_DAMAGE_GUARD
            ),
            "epoch_zero_raw_fallback": True,
        },
        "results": results,
        "elapsed_seconds": round(
            time.time() - start,
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
    print("===== 残差版本 Pilot 训练完成 =====")

    for family in FAMILIES:
        for seed in SEEDS:
            result = results[family][str(seed)]
            raw = result["raw_pilot"]
            best = result["best_pilot"]

            print(
                f"{family} seed={seed} | "
                f"epoch={result['best_epoch']} | "
                f"Top1 "
                f"{raw['top1']:.6f}"
                f" -> {best['top1']:.6f} | "
                f"Pair "
                f"{raw['pair_macro_strict']:.6f}"
                f" -> "
                f"{best['pair_macro_strict']:.6f} | "
                f"fallback="
                f"{result['fallback_to_raw_rm']}"
            )

    print("清单：", MANIFEST_PATH)
    print("耗时秒：", manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
