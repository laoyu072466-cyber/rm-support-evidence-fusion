from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random

import numpy as np
import torch

from trajectory_head import (
    TrajectoryCorrectionHead,
    question_ranking_loss,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs/trajectory_head.json"
NORMALIZATION_PATH = (
    PROJECT
    / "data/manifests/trajectory_normalization_stats.json"
)
CACHE_ROOT = (
    PROJECT
    / "data/cache/trajectory_features_v1"
    / "Skywork-Reward-V2-Qwen3-1.7B"
    / "layer_28"
)
OUTPUT_PATH = (
    PROJECT / "outputs/trajectory_head_real_smoke.json"
)

SEED = 20260829
EPOCHS = 30
QUESTIONS_PER_DATASET = 8
DEVICE = "cuda"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_key(dataset, question_uid):
    raw = (
        f"{SEED}|{dataset}|{question_uid}"
        .encode("utf-8")
    )
    return hashlib.sha256(raw).hexdigest()


class TrajectoryCache:
    def __init__(self, stem):
        self.features = np.load(
            CACHE_ROOT
            / f"{stem}.chunk_states_f16.npy",
            mmap_mode="r",
        )
        self.scores = np.load(
            CACHE_ROOT / f"{stem}.scores_f32.npy"
        )
        self.labels = np.load(
            CACHE_ROOT / f"{stem}.labels_i8.npy"
        )
        self.offsets = np.load(
            CACHE_ROOT / f"{stem}.offsets_i64.npy"
        )
        self.metadata = read_jsonl(
            CACHE_ROOT / f"{stem}.metadata.jsonl"
        )

        count = len(self.metadata)

        if not (
            len(self.scores)
            == len(self.labels)
            == len(self.offsets)
            == count
        ):
            raise RuntimeError(
                f"{stem} 缓存数量不一致"
            )

        self.groups = defaultdict(list)

        for index, row in enumerate(self.metadata):
            self.groups[row["question_uid"]].append(
                index
            )

    def raw_top1_correct(self, question_uid):
        indices = self.groups[question_uid]
        best_index = max(
            indices,
            key=lambda index: float(
                self.scores[index]
            ),
        )
        return bool(self.labels[best_index] == 1)

    def load_question(self, question_uid, device):
        indices = self.groups[question_uid]

        trajectories = []
        rewards = []
        lengths = []
        labels = []

        for index in indices:
            start, end = self.offsets[index]

            trajectory_array = np.array(
                self.features[int(start):int(end)],
                copy=True,
            )

            trajectories.append(
                torch.from_numpy(
                    trajectory_array
                ).to(device)
            )
            rewards.append(
                float(self.scores[index])
            )
            lengths.append(
                float(
                    self.metadata[index][
                        "response_token_length"
                    ]
                )
            )
            labels.append(
                int(self.labels[index])
            )

        return {
            "trajectories": trajectories,
            "rewards": torch.tensor(
                rewards,
                device=device,
                dtype=torch.float32,
            ),
            "lengths": torch.tensor(
                lengths,
                device=device,
                dtype=torch.float32,
            ),
            "labels": torch.tensor(
                labels,
                device=device,
                dtype=torch.long,
            ),
        }


def choose_questions(dataset, cache):
    wrong = []
    correct = []

    for question_uid in cache.groups:
        labels = cache.labels[
            cache.groups[question_uid]
        ]

        if not (
            np.any(labels == 1)
            and np.any(labels == 0)
        ):
            continue

        if cache.raw_top1_correct(question_uid):
            correct.append(question_uid)
        else:
            wrong.append(question_uid)

    wrong.sort(
        key=lambda uid: stable_key(dataset, uid)
    )
    correct.sort(
        key=lambda uid: stable_key(dataset, uid)
    )

    half = QUESTIONS_PER_DATASET // 2

    if len(wrong) < half or len(correct) < half:
        raise RuntimeError(
            f"{dataset} 没有足够的正确/错误问题"
        )

    selected = wrong[:half] + correct[:half]

    return selected, {
        "raw_wrong_selected": half,
        "raw_correct_selected": half,
        "available_raw_wrong": len(wrong),
        "available_raw_correct": len(correct),
    }


def pair_macro_for_question(scores, labels):
    positives = scores[labels == 1]
    negatives = scores[labels == 0]

    comparisons = (
        positives[:, None] > negatives[None, :]
    ).float()

    return float(comparisons.mean().item())


def evaluate(
    model,
    selected,
    caches,
    config,
):
    model.eval()

    rows = []
    alpha_values = []
    gate_values = []

    with torch.inference_mode():
        for dataset, question_uid in selected:
            batch = caches[dataset].load_question(
                question_uid,
                DEVICE,
            )

            corrected_scores, diagnostics = model(
                batch["trajectories"],
                batch["rewards"],
                batch["lengths"],
            )

            raw_scores = batch["rewards"]
            labels = batch["labels"]

            raw_best = int(
                torch.argmax(raw_scores).item()
            )
            corrected_best = int(
                torch.argmax(
                    corrected_scores
                ).item()
            )

            raw_correct = bool(
                labels[raw_best].item() == 1
            )
            corrected_correct = bool(
                labels[corrected_best].item() == 1
            )

            rows.append({
                "dataset": dataset,
                "question_uid": question_uid,
                "raw_correct": raw_correct,
                "corrected_correct": (
                    corrected_correct
                ),
                "raw_pair_macro": (
                    pair_macro_for_question(
                        raw_scores,
                        labels,
                    )
                ),
                "corrected_pair_macro": (
                    pair_macro_for_question(
                        corrected_scores,
                        labels,
                    )
                ),
            })

            for item in diagnostics:
                alpha_values.append(
                    float(item["alpha"].item())
                )
                if item["gate"].numel() > 0:
                    gate_values.extend(
                        item["gate"]
                        .detach()
                        .cpu()
                        .tolist()
                    )

    result = {}

    for dataset in ["GSM8K", "MATH", "ALL"]:
        if dataset == "ALL":
            subset = rows
        else:
            subset = [
                row for row in rows
                if row["dataset"] == dataset
            ]

        raw_wrong = [
            row for row in subset
            if not row["raw_correct"]
        ]
        raw_correct = [
            row for row in subset
            if row["raw_correct"]
        ]

        result[dataset] = {
            "questions": len(subset),
            "raw_top1": float(np.mean([
                row["raw_correct"]
                for row in subset
            ])),
            "corrected_top1": float(np.mean([
                row["corrected_correct"]
                for row in subset
            ])),
            "raw_pair_macro": float(np.mean([
                row["raw_pair_macro"]
                for row in subset
            ])),
            "corrected_pair_macro": float(
                np.mean([
                    row["corrected_pair_macro"]
                    for row in subset
                ])
            ),
            "correction_rate": (
                float(np.mean([
                    row["corrected_correct"]
                    for row in raw_wrong
                ]))
                if raw_wrong else None
            ),
            "damage_rate": (
                float(np.mean([
                    not row["corrected_correct"]
                    for row in raw_correct
                ]))
                if raw_correct else None
            ),
        }

    result["diagnostics"] = {
        "alpha_min": min(alpha_values),
        "alpha_mean": float(
            np.mean(alpha_values)
        ),
        "alpha_max": max(alpha_values),
        "gate_min": min(gate_values),
        "gate_mean": float(
            np.mean(gate_values)
        ),
        "gate_max": max(gate_values),
    }

    return result


def average_training_loss(
    model,
    selected,
    caches,
    config,
):
    model.eval()
    values = []

    with torch.inference_mode():
        for dataset, question_uid in selected:
            batch = caches[dataset].load_question(
                question_uid,
                DEVICE,
            )
            scores, _ = model(
                batch["trajectories"],
                batch["rewards"],
                batch["lengths"],
            )
            losses = question_ranking_loss(
                scores,
                batch["labels"],
                config,
            )
            values.append(
                float(losses["total"].item())
            )

    return float(np.mean(values))


def find_single_chunk_candidate(caches):
    for dataset, cache in caches.items():
        for index, row in enumerate(cache.metadata):
            if int(row["chunk_count"]) == 1:
                start, end = cache.offsets[index]
                trajectory = torch.from_numpy(
                    np.array(
                        cache.features[
                            int(start):int(end)
                        ],
                        copy=True,
                    )
                ).to(DEVICE)

                return {
                    "dataset": dataset,
                    "trajectory": trajectory,
                    "reward": torch.tensor(
                        float(cache.scores[index]),
                        device=DEVICE,
                    ),
                    "length": torch.tensor(
                        float(
                            row[
                                "response_token_length"
                            ]
                        ),
                        device=DEVICE,
                    ),
                }

    raise RuntimeError("没有找到单 chunk 候选")


def main():
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    normalization = json.loads(
        NORMALIZATION_PATH.read_text(
            encoding="utf-8"
        )
    )
    stats = normalization["modes"][
        "joint_dataset_balanced"
    ]

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    caches = {
        "GSM8K": TrajectoryCache("gsm_train"),
        "MATH": TrajectoryCache("math_train"),
    }

    selected = []
    selection_report = {}

    for dataset in ["GSM8K", "MATH"]:
        question_ids, report = choose_questions(
            dataset,
            caches[dataset],
        )
        selected.extend([
            (dataset, question_uid)
            for question_uid in question_ids
        ])
        selection_report[dataset] = report

    model = TrajectoryCorrectionHead(
        config,
        stats,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config["optimizer"]["learning_rate"]
        ),
        weight_decay=float(
            config["optimizer"]["weight_decay"]
        ),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    initial_loss = average_training_loss(
        model,
        selected,
        caches,
        config,
    )
    initial_metrics = evaluate(
        model,
        selected,
        caches,
        config,
    )

    print("选择问题：")
    print(json.dumps(
        selection_report,
        ensure_ascii=False,
        indent=2,
    ))
    print("初始平均 loss：", initial_loss)
    print("初始指标：")
    print(json.dumps(
        initial_metrics,
        ensure_ascii=False,
        indent=2,
    ))

    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        gsm_questions = [
            item for item in selected
            if item[0] == "GSM8K"
        ]
        math_questions = [
            item for item in selected
            if item[0] == "MATH"
        ]

        epoch_random = random.Random(
            SEED + epoch
        )
        epoch_random.shuffle(gsm_questions)
        epoch_random.shuffle(math_questions)

        epoch_losses = []

        for start in range(
            0,
            QUESTIONS_PER_DATASET,
            2,
        ):
            question_batch = (
                gsm_questions[start:start + 2]
                + math_questions[start:start + 2]
            )

            question_losses = []

            for dataset, question_uid in question_batch:
                batch = caches[
                    dataset
                ].load_question(
                    question_uid,
                    DEVICE,
                )

                scores, _ = model(
                    batch["trajectories"],
                    batch["rewards"],
                    batch["lengths"],
                )
                losses = question_ranking_loss(
                    scores,
                    batch["labels"],
                    config,
                )
                question_losses.append(
                    losses["total"]
                )

            loss = torch.stack(
                question_losses
            ).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "真实训练 loss 出现 NaN/Inf"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            gradients_finite = all(
                parameter.grad is None
                or torch.isfinite(
                    parameter.grad
                ).all()
                for parameter in model.parameters()
            )

            if not gradients_finite:
                raise RuntimeError(
                    "真实训练梯度出现 NaN/Inf"
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    config["optimizer"][
                        "gradient_clip_norm"
                    ]
                ),
            )
            optimizer.step()

            epoch_losses.append(
                float(loss.item())
            )

        mean_epoch_loss = float(
            np.mean(epoch_losses)
        )
        loss_history.append(mean_epoch_loss)

        if epoch in {1, 5, 10, 20, EPOCHS}:
            print(
                f"epoch {epoch:02d}/{EPOCHS}, "
                f"loss={mean_epoch_loss:.6f}",
                flush=True,
            )

    final_loss = average_training_loss(
        model,
        selected,
        caches,
        config,
    )
    final_metrics = evaluate(
        model,
        selected,
        caches,
        config,
    )

    single = find_single_chunk_candidate(caches)

    model.eval()
    with torch.inference_mode():
        single_result = model.score_one(
            single["trajectory"],
            single["reward"],
            single["length"],
        )

    single_chunk_difference = abs(
        float(
            single_result["score"].item()
            - single_result[
                "normalized_original"
            ].item()
        )
    )

    if not final_loss < initial_loss:
        raise RuntimeError(
            "小样本训练后 loss 没有下降"
        )

    if (
        final_metrics["ALL"][
            "corrected_pair_macro"
        ]
        < initial_metrics["ALL"][
            "corrected_pair_macro"
        ]
    ):
        raise RuntimeError(
            "小样本训练后 Pair-Macro 没有提高"
        )

    if single_chunk_difference > 1e-7:
        raise RuntimeError(
            "真实单 chunk 候选未保持原始奖励"
        )

    report = {
        "scope": "train_subset_overfit_smoke_only",
        "pilot_used": False,
        "selected_questions": selection_report,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "single_chunk_dataset": single["dataset"],
        "single_chunk_score_difference": (
            single_chunk_difference
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            3,
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== 真实缓存过拟合测试结果 =====")
    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("真实缓存过拟合测试通过。")


if __name__ == "__main__":
    main()
