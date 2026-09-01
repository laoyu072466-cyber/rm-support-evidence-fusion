from pathlib import Path
import copy
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import train_trajectory_head_joint_default as base


SEED = 20260901
GAMMA = 0.8

MAX_EPOCHS = 30
PATIENCE = 5
QUESTIONS_PER_DATASET_PER_BATCH = 4

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

LAMBDA_BT = 0.5
LAMBDA_CAL = 0.1

LAMBDA_PROTECT = 3.0
LAMBDA_CORRECT = 1.0
LAMBDA_ANCHOR = 0.2
TOP_MARGIN = 0.25

MIX_BETAS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
TOP1_TIE_BAND = 0.005

CHECKPOINT = (
    ROOT / "outputs/checkpoints/"
    "trajectory_head_damage_aware_seed20260901.pt"
)
RESULT = (
    ROOT / "outputs/cast_rm/"
    "damage_aware_seed20260901.json"
)
MANIFEST = (
    ROOT / "data/manifests/"
    "damage_aware_seed20260901.json"
)

DEVICE = base.DEVICE

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def question_damage_loss(
    scores,
    labels,
    raw_scores,
):
    positive = labels > 0.5
    negative = ~positive

    listwise = (
        torch.logsumexp(scores, dim=0)
        - torch.logsumexp(
            scores[positive],
            dim=0,
        )
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

    base_loss = (
        listwise
        + LAMBDA_BT * bt
        + LAMBDA_CAL * brier
    )

    raw_order = torch.argsort(raw_scores)
    raw_top = raw_order[-1]
    raw_second = raw_order[-2]

    raw_margin = (
        raw_scores[raw_top]
        - raw_scores[raw_second]
    ).detach()

    raw_top_correct = bool(
        positive[raw_top].item()
    )

    zero = scores.new_zeros(())

    if raw_top_correct:
        # 原始第一名正确：重点防止最危险错误候选越过它。
        hardest_negative = torch.max(
            scores[negative]
        )
        protect = F.softplus(
            hardest_negative
            - scores[raw_top]
            + TOP_MARGIN
        )
        correct = zero

        # 原始模型越有把握，越应保持锚定。
        anchor_multiplier = (
            1.0 + torch.sigmoid(raw_margin)
        )
    else:
        # 原始第一名错误：推动当前最强正确候选越过它。
        best_positive = torch.max(
            scores[positive]
        )
        correct = F.softplus(
            scores[raw_top]
            - best_positive
            + TOP_MARGIN
        )
        protect = zero

        # 原始模型已经选错，允许更自由地校正。
        anchor_multiplier = scores.new_tensor(
            0.25
        )

    anchor = (
        anchor_multiplier
        * F.smooth_l1_loss(
            scores,
            raw_scores,
            reduction="mean",
        )
    )

    total = (
        base_loss
        + LAMBDA_PROTECT * protect
        + LAMBDA_CORRECT * correct
        + LAMBDA_ANCHOR * anchor
    )

    return {
        "total": total,
        "base": base_loss,
        "protect": protect,
        "correct": correct,
        "anchor": anchor,
        "raw_top_correct": float(
            raw_top_correct
        ),
    }


def batch_damage_loss(
    scores,
    labels,
    raw_scores,
    question_slices,
):
    records = []

    for left, right in question_slices:
        records.append(
            question_damage_loss(
                scores[left:right],
                labels[left:right],
                raw_scores[left:right],
            )
        )

    means = {}
    for key in (
        "total",
        "base",
        "protect",
        "correct",
        "anchor",
    ):
        means[key] = torch.stack([
            row[key] for row in records
        ]).mean()

    means["raw_top_accuracy"] = float(
        np.mean([
            row["raw_top_correct"]
            for row in records
        ])
    )

    return means


def selection_better(current, best):
    if best is None:
        return True

    current_macro = current["dataset_macro"]
    best_macro = best["dataset_macro"]

    current_top1 = current_macro["top1"]
    best_top1 = best_macro["top1"]

    if current_top1 > best_top1 + TOP1_TIE_BAND:
        return True
    if current_top1 < best_top1 - TOP1_TIE_BAND:
        return False

    current_pair = current_macro[
        "pair_macro_strict"
    ]
    best_pair = best_macro[
        "pair_macro_strict"
    ]

    if current_pair > best_pair + 1e-10:
        return True
    if current_pair < best_pair - 1e-10:
        return False

    return (
        current_macro["damage_rate"]
        < best_macro["damage_rate"]
    )


def mixed_scores(store, corrected, beta, norm):
    raw_normalized = (
        store.scores.astype(np.float32)
        - float(norm["reward_mean"])
    ) / float(norm["reward_std"])

    return (
        raw_normalized
        + beta * (
            corrected - raw_normalized
        )
    )


def evaluate_mixed(
    store,
    corrected,
    beta,
    norm,
):
    scores = mixed_scores(
        store,
        corrected,
        beta,
        norm,
    )

    result = base.calculate_metrics(
        store,
        scores,
    )

    raw_pair = base.raw_pair_metric(store)

    result["raw_pair_macro_strict"] = raw_pair
    result["pair_delta"] = (
        result["corrected_pair_macro_strict"]
        - raw_pair
    )
    result["top1_delta"] = (
        result["corrected_top1"]
        - result["raw_top1"]
    )
    result["mix_beta"] = beta

    return result


def main():
    started = time.time()

    normalization_manifest = json.loads(
        (
            ROOT / "data/manifests/"
            "trajectory_normalization_stats.json"
        ).read_text(encoding="utf-8")
    )
    normalization = normalization_manifest[
        "modes"
    ]["joint_dataset_balanced"]

    print("===== Damage-aware 轨迹头训练 =====")
    print("随机种子：", SEED)
    print(
        "损失权重：",
        {
            "protect": LAMBDA_PROTECT,
            "correct": LAMBDA_CORRECT,
            "anchor": LAMBDA_ANCHOR,
            "top_margin": TOP_MARGIN,
        },
    )

    gsm_train = base.DatasetStore(
        "gsm_train",
        "GSM8K",
    )
    math_train = base.DatasetStore(
        "math_train",
        "MATH",
    )
    gsm_pilot = base.DatasetStore(
        "gsm_pilot",
        "GSM8K",
    )
    math_pilot = base.DatasetStore(
        "math_pilot",
        "MATH",
    )

    model = base.TrajectoryCorrectionHead(
        normalization
    ).to(DEVICE)
    model.gamma = GAMMA

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    target_questions = (
        math.ceil(
            max(
                len(gsm_train.question_uids),
                len(math_train.question_uids),
            )
            / QUESTIONS_PER_DATASET_PER_BATCH
        )
        * QUESTIONS_PER_DATASET_PER_BATCH
    )

    best_state = None
    best_metrics = None
    best_epoch = None
    no_improvement = 0
    history = []

    print("正式训练开始。")

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.time()
        rng = random.Random(SEED + epoch)

        gsm_order = base.repeated_shuffle(
            gsm_train.question_uids,
            target_questions,
            rng,
        )
        math_order = base.repeated_shuffle(
            math_train.question_uids,
            target_questions,
            rng,
        )

        model.train()
        epoch_records = []

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
                slices,
            ) = base.pack_questions(refs)

            optimizer.zero_grad(set_to_none=True)

            corrected, _ = model(
                hidden,
                counts,
                rewards,
                lengths,
            )

            raw_normalized = (
                rewards.float()
                - model.reward_mean
            ) / model.reward_std.clamp_min(1e-6)

            losses = batch_damage_loss(
                corrected,
                labels,
                raw_normalized,
                slices,
            )
            loss = losses["total"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"epoch {epoch} 出现非有限 loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
            optimizer.step()

            epoch_records.append({
                key: (
                    float(value.item())
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in losses.items()
            })

        pilot = base.evaluate_pilot(
            model,
            gsm_pilot,
            math_pilot,
        )

        loss_mean = {
            key: float(np.mean([
                row[key]
                for row in epoch_records
            ]))
            for key in epoch_records[0]
        }

        record = {
            "epoch": epoch,
            "loss": loss_mean,
            "pilot": pilot,
            "elapsed_seconds": round(
                time.time() - epoch_started,
                3,
            ),
        }
        history.append(record)

        macro = pilot["dataset_macro"]

        print(
            f"epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={loss_mean['total']:.6f} | "
            f"protect={loss_mean['protect']:.6f} | "
            f"correct={loss_mean['correct']:.6f} | "
            f"Top1={macro['top1']:.6f} | "
            f"Pair={macro['pair_macro_strict']:.6f} | "
            f"Damage={macro['damage_rate']:.6f} | "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

        if selection_better(pilot, best_metrics):
            best_metrics = copy.deepcopy(pilot)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print(
                f"连续 {PATIENCE} 轮未改善，提前停止。"
            )
            break

    if best_state is None:
        raise RuntimeError("没有得到有效 checkpoint")

    model.load_state_dict(best_state)
    model.eval()

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                "lambda_protect": LAMBDA_PROTECT,
                "lambda_correct": LAMBDA_CORRECT,
                "lambda_anchor": LAMBDA_ANCHOR,
                "top_margin": TOP_MARGIN,
            },
        },
        CHECKPOINT,
    )

    print("\n===== Pilot 选择残差强度 =====")

    gsm_pilot_scores = base.predict_store(
        model,
        gsm_pilot,
    )
    math_pilot_scores = base.predict_store(
        model,
        math_pilot,
    )

    beta_results = []

    for beta in MIX_BETAS:
        gsm_result = evaluate_mixed(
            gsm_pilot,
            gsm_pilot_scores,
            beta,
            normalization,
        )
        math_result = evaluate_mixed(
            math_pilot,
            math_pilot_scores,
            beta,
            normalization,
        )

        macro = {
            "top1": float(np.mean([
                gsm_result["corrected_top1"],
                math_result["corrected_top1"],
            ])),
            "pair": float(np.mean([
                gsm_result[
                    "corrected_pair_macro_strict"
                ],
                math_result[
                    "corrected_pair_macro_strict"
                ],
            ])),
            "damage": float(np.mean([
                gsm_result["damage_rate"],
                math_result["damage_rate"],
            ])),
        }

        beta_results.append({
            "beta": beta,
            "GSM8K": gsm_result,
            "MATH": math_result,
            "macro": macro,
        })

        print(
            f"beta={beta:.2f} | "
            f"Top1={macro['top1']:.6f} | "
            f"Pair={macro['pair']:.6f} | "
            f"Damage={macro['damage']:.6f}"
        )

    best_beta_top1 = max(
        row["macro"]["top1"]
        for row in beta_results
    )

    beta_band = [
        row for row in beta_results
        if row["macro"]["top1"]
        >= best_beta_top1 - TOP1_TIE_BAND
    ]

    selected_beta_row = max(
        beta_band,
        key=lambda row: (
            row["macro"]["pair"],
            -row["macro"]["damage"],
            -row["beta"],
        ),
    )
    selected_beta = float(
        selected_beta_row["beta"]
    )

    print("最终 mix beta：", selected_beta)

    print("\n===== 当前三个测试集 =====")

    test_specs = {
        "GSM8K_ID": (
            "gsm_id_test",
            "GSM8K",
        ),
        "MATH_ID": (
            "math_id_test",
            "MATH",
        ),
        "SVAMP_OOD": (
            "svamp_ood",
            "SVAMP",
        ),
    }

    test_results = {}

    for name, (prefix, dataset) in test_specs.items():
        store = base.DatasetStore(
            prefix,
            dataset,
        )
        corrected = base.predict_store(
            model,
            store,
        )
        result = evaluate_mixed(
            store,
            corrected,
            selected_beta,
            normalization,
        )
        test_results[name] = result

        print(
            f"{name}: "
            f"Top1={result['raw_top1']:.6f}"
            f" -> {result['corrected_top1']:.6f} "
            f"({result['top1_delta']:+.6f}), "
            f"Pair={result['raw_pair_macro_strict']:.6f}"
            f" -> "
            f"{result['corrected_pair_macro_strict']:.6f} "
            f"({result['pair_delta']:+.6f}), "
            f"Damage={result['damage_rate']:.6f}"
        )

    test_macro = {
        "top1": float(np.mean([
            row["corrected_top1"]
            for row in test_results.values()
        ])),
        "top1_delta": float(np.mean([
            row["top1_delta"]
            for row in test_results.values()
        ])),
        "pair": float(np.mean([
            row["corrected_pair_macro_strict"]
            for row in test_results.values()
        ])),
        "pair_delta": float(np.mean([
            row["pair_delta"]
            for row in test_results.values()
        ])),
        "damage": float(np.mean([
            row["damage_rate"]
            for row in test_results.values()
        ])),
    }

    print(
        "Macro:",
        json.dumps(
            test_macro,
            ensure_ascii=False,
            indent=2,
        ),
    )

    final = {
        "version": "damage_aware_trajectory_v1",
        "seed": SEED,
        "best_epoch": best_epoch,
        "loss_configuration": {
            "lambda_protect": LAMBDA_PROTECT,
            "lambda_correct": LAMBDA_CORRECT,
            "lambda_anchor": LAMBDA_ANCHOR,
            "top_margin": TOP_MARGIN,
        },
        "best_pilot": best_metrics,
        "beta_search": beta_results,
        "selected_mix_beta": selected_beta,
        "test": test_results,
        "test_macro": test_macro,
        "history": history,
        "checkpoint": str(CHECKPOINT),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "peak_gpu_gb": round(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3),
            3,
        ),
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(
        final,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    RESULT.write_text(text, encoding="utf-8")
    MANIFEST.write_text(text, encoding="utf-8")

    print("\ncheckpoint：", CHECKPOINT)
    print("结果：", RESULT)
    print("清单：", MANIFEST)
    print("总耗时秒：", final["elapsed_seconds"])


if __name__ == "__main__":
    main()
