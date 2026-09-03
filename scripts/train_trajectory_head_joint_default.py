from pathlib import Path
from collections import defaultdict
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
CACHE = (
    ROOT / "data/cache/trajectory_features_v1/"
    "Skywork-Reward-V2-Qwen3-1.7B/layer_28"
)
STATS_PATH = (
    ROOT / "data/manifests/trajectory_normalization_stats.json"
)
RUN_TAG = os.environ.get(
    "TRAJ_RUN_TAG",
    "joint_default_seed20260829",
)
OUTPUT_PATH = (
    ROOT / f"outputs/trajectory_head_{RUN_TAG}.json"
)
CHECKPOINT_PATH = (
    ROOT / "outputs/checkpoints"
    / f"trajectory_head_{RUN_TAG}.pt"
)

DEVICE = torch.device("cuda")
SEED = int(os.environ.get("TRAJ_SEED", "20260829"))
MAX_EPOCHS = int(
    os.environ.get("TRAJ_MAX_EPOCHS", "30")
)
PATIENCE = int(
    os.environ.get("TRAJ_PATIENCE", "5")
)
QUESTIONS_PER_DATASET_PER_BATCH = int(
    os.environ.get("TRAJ_QUESTIONS_PER_DATASET", "8")
)

GAMMA = float(os.environ.get("TRAJ_GAMMA", "0.9"))
LAMBDA_BT = float(
    os.environ.get("TRAJ_LAMBDA_BT", "0.5")
)
LAMBDA_CAL = float(
    os.environ.get("TRAJ_LAMBDA_CAL", "0.1")
)
LEARNING_RATE = float(
    os.environ.get("TRAJ_LEARNING_RATE", "0.001")
)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision("high")


class DatasetStore:
    def __init__(self, prefix, dataset):
        self.prefix = prefix
        self.dataset = dataset

        feature_path = CACHE / f"{prefix}.chunk_states_f16.npy"
        score_path = CACHE / f"{prefix}.scores_f32.npy"
        label_path = CACHE / f"{prefix}.labels_i8.npy"
        offset_path = CACHE / f"{prefix}.offsets_i64.npy"
        metadata_path = CACHE / f"{prefix}.metadata.jsonl"

        print(f"加载 {dataset}/{prefix} 到显卡……")
        feature_array = np.load(feature_path, mmap_mode="r")
        self.features = torch.from_numpy(
            np.array(feature_array, copy=True)
        ).to(DEVICE, dtype=torch.float16)

        self.scores = np.asarray(
            np.load(score_path), dtype=np.float32
        )
        self.labels = np.asarray(
            np.load(label_path), dtype=np.int8
        )
        self.offsets = np.asarray(
            np.load(offset_path), dtype=np.int64
        )

        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        candidate_count = len(self.scores)
        assert len(self.labels) == candidate_count
        assert len(self.metadata) == candidate_count

        if (
            self.offsets.ndim == 1
            and self.offsets.shape[0] == candidate_count + 1
        ):
            self.offset_format = "boundaries"
            final_offset = int(self.offsets[-1])
        elif (
            self.offsets.ndim == 2
            and self.offsets.shape == (candidate_count, 2)
        ):
            self.offset_format = "pairs"
            final_offset = int(self.offsets[-1, 1])
        else:
            raise RuntimeError(
                "无法识别 offsets 格式："
                f"shape={self.offsets.shape}, "
                f"candidates={candidate_count}"
            )

        assert self.features.shape[0] == final_offset

        self.response_lengths = np.asarray(
            [
                int(item["response_token_length"])
                for item in self.metadata
            ],
            dtype=np.int64,
        )

        groups = defaultdict(list)
        for index, item in enumerate(self.metadata):
            groups[item["question_uid"]].append(index)

        self.groups = dict(groups)
        self.question_uids = list(self.groups)

        for uid, indices in self.groups.items():
            labels = self.labels[indices]
            if not (np.any(labels == 1) and np.any(labels == 0)):
                raise RuntimeError(
                    f"{dataset} 问题 {uid} 不是正负混合问题"
                )

        print(
            f"  问题={len(self.question_uids)}, "
            f"候选={candidate_count}, "
            f"端点={self.features.shape[0]}"
        )


class TrajectoryCorrectionHead(nn.Module):
    def __init__(self, normalization):
        super().__init__()

        hidden_size = 2048
        gate_hidden_size = 32
        alpha_hidden_size = 16

        self.layer_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.prefix_probe = nn.Linear(hidden_size, 1)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, gate_hidden_size),
            nn.GELU(),
            nn.Linear(gate_hidden_size, 1),
        )

        self.alpha = nn.Sequential(
            nn.Linear(5, alpha_hidden_size),
            nn.GELU(),
            nn.Linear(alpha_hidden_size, 1),
        )

        alpha_initial = 0.9
        alpha_bias = math.log(
            alpha_initial / (1.0 - alpha_initial)
        )
        nn.init.zeros_(self.alpha[-1].weight)
        nn.init.constant_(self.alpha[-1].bias, alpha_bias)

        self.gamma = GAMMA

        for name in (
            "reward_mean",
            "reward_std",
            "log_t_mean",
            "log_t_std",
            "log_l_mean",
            "log_l_std",
        ):
            self.register_buffer(
                name,
                torch.tensor(
                    float(normalization[name]),
                    dtype=torch.float32,
                ),
            )

    def forward(
        self,
        hidden_states,
        chunk_counts,
        original_rewards,
        response_lengths,
    ):
        hidden_states = hidden_states.float()
        chunk_counts = chunk_counts.long()

        normalized_hidden = self.layer_norm(hidden_states)
        prefix_values = self.prefix_probe(
            normalized_hidden
        ).squeeze(-1)

        terminal_index = (chunk_counts - 1).unsqueeze(1)
        terminal_prefix = prefix_values.gather(
            1, terminal_index
        ).squeeze(1)

        candidate_count, max_chunks, _ = hidden_states.shape
        trajectory_reward = torch.zeros(
            candidate_count,
            device=hidden_states.device,
            dtype=torch.float32,
        )

        if max_chunks > 1:
            previous = normalized_hidden[:, :-1]
            current = normalized_hidden[:, 1:]
            evidence = (
                prefix_values[:, 1:]
                - prefix_values[:, :-1]
            )

            gate_input = torch.cat(
                [
                    previous,
                    current,
                    evidence.unsqueeze(-1),
                ],
                dim=-1,
            )
            gate_values = torch.sigmoid(
                self.gate(gate_input).squeeze(-1)
            )

            positions = torch.arange(
                max_chunks - 1,
                device=hidden_states.device,
            ).unsqueeze(0)

            transition_counts = chunk_counts - 1
            transition_mask = (
                positions < transition_counts.unsqueeze(1)
            )

            exponents = (
                transition_counts.unsqueeze(1)
                - 1
                - positions
            ).clamp_min(0)

            decay = torch.pow(
                torch.tensor(
                    self.gamma,
                    device=hidden_states.device,
                    dtype=torch.float32,
                ),
                exponents.float(),
            )

            weights = (
                decay
                * gate_values
                * transition_mask.float()
            )

            numerator = (weights * evidence).sum(dim=1)
            denominator = weights.sum(dim=1).clamp_min(1e-6)
            trajectory_reward = numerator / denominator

        normalized_original = (
            original_rewards.float() - self.reward_mean
        ) / self.reward_std.clamp_min(1e-6)

        normalized_log_t = (
            torch.log1p(chunk_counts.float())
            - self.log_t_mean
        ) / self.log_t_std.clamp_min(1e-6)

        normalized_log_l = (
            torch.log1p(response_lengths.float())
            - self.log_l_mean
        ) / self.log_l_std.clamp_min(1e-6)

        alpha_input = torch.stack(
            [
                normalized_original,
                terminal_prefix,
                trajectory_reward,
                normalized_log_t,
                normalized_log_l,
            ],
            dim=1,
        )

        alpha = torch.sigmoid(
            self.alpha(alpha_input).squeeze(-1)
        )

        single_chunk = chunk_counts < 2
        alpha = torch.where(
            single_chunk,
            torch.ones_like(alpha),
            alpha,
        )

        corrected_score = (
            alpha * normalized_original
            + (1.0 - alpha) * trajectory_reward
        )
        corrected_score = torch.where(
            single_chunk,
            normalized_original,
            corrected_score,
        )

        return corrected_score, alpha


def pack_questions(question_refs):
    sequences = []
    rewards = []
    labels = []
    response_lengths = []
    question_slices = []

    candidate_start = 0

    for store, uid in question_refs:
        indices = store.groups[uid]

        for index in indices:
            if store.offset_format == "boundaries":
                left = int(store.offsets[index])
                right = int(store.offsets[index + 1])
            else:
                left = int(store.offsets[index, 0])
                right = int(store.offsets[index, 1])

            sequences.append(
                store.features[left:right].float()
            )
            rewards.append(float(store.scores[index]))
            labels.append(float(store.labels[index]))
            response_lengths.append(
                int(store.response_lengths[index])
            )

        candidate_end = candidate_start + len(indices)
        question_slices.append(
            (candidate_start, candidate_end)
        )
        candidate_start = candidate_end

    chunk_counts = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        device=DEVICE,
        dtype=torch.long,
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    rewards = torch.tensor(
        rewards,
        device=DEVICE,
        dtype=torch.float32,
    )
    labels = torch.tensor(
        labels,
        device=DEVICE,
        dtype=torch.float32,
    )
    response_lengths = torch.tensor(
        response_lengths,
        device=DEVICE,
        dtype=torch.float32,
    )

    return (
        padded,
        chunk_counts,
        rewards,
        labels,
        response_lengths,
        question_slices,
    )


def question_loss(scores, labels):
    positive = labels > 0.5
    negative = ~positive

    scaled = scores
    listwise = (
        torch.logsumexp(scaled, dim=0)
        - torch.logsumexp(scaled[positive], dim=0)
    )

    margins = (
        scores[positive].unsqueeze(1)
        - scores[negative].unsqueeze(0)
    )
    bt = F.softplus(-margins).mean()

    probabilities = torch.sigmoid(scores)
    brier = torch.mean(
        (probabilities - labels) ** 2
    )

    total = (
        listwise
        + LAMBDA_BT * bt
        + LAMBDA_CAL * brier
    )
    return total


def batch_loss(scores, labels, question_slices):
    losses = []

    for left, right in question_slices:
        losses.append(
            question_loss(
                scores[left:right],
                labels[left:right],
            )
        )

    return torch.stack(losses).mean()


def calculate_metrics(store, corrected_scores):
    top1_values = []
    pair_values = []
    raw_correct = []
    corrected_correct = []

    for uid in store.question_uids:
        indices = np.asarray(
            store.groups[uid],
            dtype=np.int64,
        )
        labels = store.labels[indices]
        raw = store.scores[indices]
        corrected = corrected_scores[indices]

        raw_winner = int(np.argmax(raw))
        corrected_winner = int(np.argmax(corrected))

        raw_ok = bool(labels[raw_winner] == 1)
        corrected_ok = bool(
            labels[corrected_winner] == 1
        )

        raw_correct.append(raw_ok)
        corrected_correct.append(corrected_ok)
        top1_values.append(float(corrected_ok))

        positive = corrected[labels == 1]
        negative = corrected[labels == 0]

        pair_values.append(
            float(
                (
                    positive[:, None]
                    > negative[None, :]
                ).mean()
            )
        )

    raw_correct = np.asarray(raw_correct, dtype=bool)
    corrected_correct = np.asarray(
        corrected_correct,
        dtype=bool,
    )

    raw_wrong_mask = ~raw_correct
    raw_correct_mask = raw_correct

    correction_rate = (
        float(corrected_correct[raw_wrong_mask].mean())
        if np.any(raw_wrong_mask)
        else 0.0
    )
    damage_rate = (
        float((~corrected_correct[raw_correct_mask]).mean())
        if np.any(raw_correct_mask)
        else 0.0
    )

    return {
        "questions": len(store.question_uids),
        "candidates": len(store.scores),
        "raw_top1": float(raw_correct.mean()),
        "corrected_top1": float(
            np.mean(top1_values)
        ),
        "corrected_pair_macro_strict": float(
            np.mean(pair_values)
        ),
        "correction_rate": correction_rate,
        "damage_rate": damage_rate,
    }


def raw_pair_metric(store):
    values = []

    for uid in store.question_uids:
        indices = np.asarray(store.groups[uid])
        labels = store.labels[indices]
        scores = store.scores[indices]

        positive = scores[labels == 1]
        negative = scores[labels == 0]

        values.append(
            float(
                (
                    positive[:, None]
                    > negative[None, :]
                ).mean()
            )
        )

    return float(np.mean(values))


@torch.no_grad()
def predict_store(model, store, batch_questions=16):
    model.eval()
    predictions = np.empty(
        len(store.scores),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(store.question_uids),
        batch_questions,
    ):
        uids = store.question_uids[
            start:start + batch_questions
        ]
        refs = [(store, uid) for uid in uids]

        (
            hidden,
            counts,
            rewards,
            _,
            lengths,
            slices,
        ) = pack_questions(refs)

        scores, _ = model(
            hidden,
            counts,
            rewards,
            lengths,
        )

        scores = scores.detach().cpu().numpy()

        for uid, (left, right) in zip(uids, slices):
            indices = store.groups[uid]
            predictions[indices] = scores[left:right]

    return predictions


def evaluate_pilot(model, gsm_pilot, math_pilot):
    gsm_scores = predict_store(model, gsm_pilot)
    math_scores = predict_store(model, math_pilot)

    gsm_metrics = calculate_metrics(
        gsm_pilot,
        gsm_scores,
    )
    math_metrics = calculate_metrics(
        math_pilot,
        math_scores,
    )

    macro_top1 = (
        gsm_metrics["corrected_top1"]
        + math_metrics["corrected_top1"]
    ) / 2

    macro_pair = (
        gsm_metrics["corrected_pair_macro_strict"]
        + math_metrics["corrected_pair_macro_strict"]
    ) / 2

    macro_damage = (
        gsm_metrics["damage_rate"]
        + math_metrics["damage_rate"]
    ) / 2

    return {
        "GSM8K": gsm_metrics,
        "MATH": math_metrics,
        "dataset_macro": {
            "top1": macro_top1,
            "pair_macro_strict": macro_pair,
            "damage_rate": macro_damage,
        },
    }


def selection_is_better(current, best):
    if best is None:
        return True

    current_macro = current["dataset_macro"]
    best_macro = best["dataset_macro"]

    current_top1 = current_macro["top1"]
    best_top1 = best_macro["top1"]

    if current_top1 > best_top1 + 0.005:
        return True

    if abs(current_top1 - best_top1) <= 0.005:
        current_pair = current_macro[
            "pair_macro_strict"
        ]
        best_pair = best_macro[
            "pair_macro_strict"
        ]

        if current_pair > best_pair + 1e-10:
            return True

        if abs(current_pair - best_pair) <= 1e-10:
            return (
                current_macro["damage_rate"]
                < best_macro["damage_rate"]
            )

    return False


def repeated_shuffle(items, target_length, rng):
    result = []

    while len(result) < target_length:
        current = list(items)
        rng.shuffle(current)
        result.extend(current)

    return result[:target_length]


def main():
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    with STATS_PATH.open("r", encoding="utf-8") as file:
        normalization_manifest = json.load(file)

    normalization = normalization_manifest["modes"][
        "joint_dataset_balanced"
    ]

    gsm_train = DatasetStore(
        "gsm_train",
        "GSM8K",
    )
    math_train = DatasetStore(
        "math_train",
        "MATH",
    )
    gsm_pilot = DatasetStore(
        "gsm_pilot",
        "GSM8K",
    )
    math_pilot = DatasetStore(
        "math_pilot",
        "MATH",
    )

    model = TrajectoryCorrectionHead(
        normalization
    ).to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters != 133331:
        raise RuntimeError(
            "参数量不符合冻结设计："
            f"{trainable_parameters} != 133331"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    baseline_pilot = {
        "GSM8K": {
            "top1": float(
                np.mean(
                    [
                        gsm_pilot.labels[
                            gsm_pilot.groups[uid][
                                int(
                                    np.argmax(
                                        gsm_pilot.scores[
                                            gsm_pilot.groups[uid]
                                        ]
                                    )
                                )
                            ]
                        ]
                        for uid in gsm_pilot.question_uids
                    ]
                )
            ),
            "pair_macro_strict": raw_pair_metric(
                gsm_pilot
            ),
        },
        "MATH": {
            "top1": float(
                np.mean(
                    [
                        math_pilot.labels[
                            math_pilot.groups[uid][
                                int(
                                    np.argmax(
                                        math_pilot.scores[
                                            math_pilot.groups[uid]
                                        ]
                                    )
                                )
                            ]
                        ]
                        for uid in math_pilot.question_uids
                    ]
                )
            ),
            "pair_macro_strict": raw_pair_metric(
                math_pilot
            ),
        },
    }
    baseline_pilot["dataset_macro"] = {
        "top1": (
            baseline_pilot["GSM8K"]["top1"]
            + baseline_pilot["MATH"]["top1"]
        ) / 2,
        "pair_macro_strict": (
            baseline_pilot["GSM8K"][
                "pair_macro_strict"
            ]
            + baseline_pilot["MATH"][
                "pair_macro_strict"
            ]
        ) / 2,
    }

    print("\n===== 正式训练已开始 =====")
    print("模式：joint_dataset_balanced")
    print("训练问题：", {
        "GSM8K": len(gsm_train.question_uids),
        "MATH": len(math_train.question_uids),
    })
    print("Pilot 基线：")
    print(
        json.dumps(
            baseline_pilot,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("可训练参数：", trainable_parameters)

    initial_pilot = evaluate_pilot(
        model,
        gsm_pilot,
        math_pilot,
    )
    print("未训练校正头 Pilot：")
    print(
        json.dumps(
            initial_pilot,
            ensure_ascii=False,
            indent=2,
        )
    )

    target_questions = math.ceil(
        max(
            len(gsm_train.question_uids),
            len(math_train.question_uids),
        )
        / QUESTIONS_PER_DATASET_PER_BATCH
    ) * QUESTIONS_PER_DATASET_PER_BATCH

    best_metrics = None
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        rng = random.Random(SEED + epoch)

        gsm_order = repeated_shuffle(
            gsm_train.question_uids,
            target_questions,
            rng,
        )
        math_order = repeated_shuffle(
            math_train.question_uids,
            target_questions,
            rng,
        )

        model.train()
        epoch_losses = []

        for start in range(
            0,
            target_questions,
            QUESTIONS_PER_DATASET_PER_BATCH,
        ):
            gsm_uids = gsm_order[
                start:
                start + QUESTIONS_PER_DATASET_PER_BATCH
            ]
            math_uids = math_order[
                start:
                start + QUESTIONS_PER_DATASET_PER_BATCH
            ]

            refs = (
                [(gsm_train, uid) for uid in gsm_uids]
                + [(math_train, uid) for uid in math_uids]
            )
            rng.shuffle(refs)

            (
                hidden,
                counts,
                rewards,
                labels,
                lengths,
                question_slices,
            ) = pack_questions(refs)

            optimizer.zero_grad(set_to_none=True)

            corrected_scores, _ = model(
                hidden,
                counts,
                rewards,
                lengths,
            )

            loss = batch_loss(
                corrected_scores,
                labels,
                question_slices,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"第 {epoch} 轮出现非有限 loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        pilot_metrics = evaluate_pilot(
            model,
            gsm_pilot,
            math_pilot,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(
                np.mean(epoch_losses)
            ),
            "elapsed_seconds": round(
                time.time() - epoch_start,
                3,
            ),
            "pilot": pilot_metrics,
        }
        history.append(epoch_record)

        macro = pilot_metrics["dataset_macro"]

        print(
            f"epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={epoch_record['train_loss']:.6f} | "
            f"macro_top1={macro['top1']:.6f} | "
            f"macro_pair={macro['pair_macro_strict']:.6f} | "
            f"damage={macro['damage_rate']:.6f} | "
            f"time={epoch_record['elapsed_seconds']:.1f}s",
            flush=True,
        )

        if selection_is_better(
            pilot_metrics,
            best_metrics,
        ):
            best_metrics = copy.deepcopy(pilot_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "pilot_metrics": best_metrics,
                    "seed": SEED,
                    "normalization": normalization,
                    "configuration": {
                        "gamma": GAMMA,
                        "lambda_bt": LAMBDA_BT,
                        "lambda_cal": LAMBDA_CAL,
                    },
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"连续 {PATIENCE} 轮未改善，提前停止。",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("没有产生有效 checkpoint")

    model.load_state_dict(best_state)
    final_pilot = evaluate_pilot(
        model,
        gsm_pilot,
        math_pilot,
    )

    result = {
        "version": "trajectory_head_joint_grid_v1",
        "run_tag": RUN_TAG,
        "seed": SEED,
        "training_mode": "joint_dataset_balanced",
        "evaluation_scope": "pilot_validation_only",
        "test_used": False,
        "ood_used": False,
        "configuration": {
            "selected_layer": 28,
            "hidden_size": 2048,
            "gamma": GAMMA,
            "lambda_bt": LAMBDA_BT,
            "lambda_cal": LAMBDA_CAL,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "questions_per_dataset_per_batch":
                QUESTIONS_PER_DATASET_PER_BATCH,
        },
        "train_data": {
            "GSM8K": {
                "questions": len(
                    gsm_train.question_uids
                ),
                "candidates": len(gsm_train.scores),
            },
            "MATH": {
                "questions": len(
                    math_train.question_uids
                ),
                "candidates": len(math_train.scores),
            },
        },
        "trainable_parameters": trainable_parameters,
        "baseline_pilot": baseline_pilot,
        "initial_untrained_head_pilot": initial_pilot,
        "best_epoch": best_epoch,
        "best_pilot": final_pilot,
        "epochs_completed": len(history),
        "history": history,
        "checkpoint": str(CHECKPOINT_PATH),
        "total_elapsed_seconds": round(
            time.time() - start_time,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024 ** 3,
            3,
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== 正式 Joint 默认配置训练完成 =====")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "baseline_pilot": baseline_pilot,
                "best_pilot": final_pilot,
                "epochs_completed": len(history),
                "total_elapsed_seconds":
                    result["total_elapsed_seconds"],
                "peak_gpu_gb": result["peak_gpu_gb"],
                "checkpoint": str(CHECKPOINT_PATH),
                "result": str(OUTPUT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
