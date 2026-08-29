from pathlib import Path
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryCorrectionHead(nn.Module):
    def __init__(
        self,
        config,
        normalization_stats,
    ):
        super().__init__()

        architecture = config["architecture"]
        method = config["default_method"]

        self.hidden_size = int(config["hidden_size"])
        self.gamma = float(method["gamma"])

        gate_hidden = int(
            architecture["gate_hidden_size"]
        )
        alpha_hidden = int(
            architecture["alpha_hidden_size"]
        )

        self.prefix_probe = nn.Linear(
            self.hidden_size,
            1,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(
                self.hidden_size * 2 + 1,
                gate_hidden,
            ),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(5, alpha_hidden),
            nn.GELU(),
            nn.Linear(alpha_hidden, 1),
        )

        nn.init.normal_(
            self.prefix_probe.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(self.prefix_probe.bias)

        alpha_initial = float(
            architecture["alpha_initial_value"]
        )
        alpha_logit = math.log(
            alpha_initial / (1.0 - alpha_initial)
        )

        nn.init.zeros_(self.alpha_mlp[-1].weight)
        nn.init.constant_(
            self.alpha_mlp[-1].bias,
            alpha_logit,
        )

        self.register_buffer(
            "reward_mean",
            torch.tensor(
                float(normalization_stats["reward_mean"]),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "reward_std",
            torch.tensor(
                float(normalization_stats["reward_std"]),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_t_mean",
            torch.tensor(
                float(normalization_stats["log_t_mean"]),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_t_std",
            torch.tensor(
                float(normalization_stats["log_t_std"]),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_l_mean",
            torch.tensor(
                float(normalization_stats["log_l_mean"]),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_l_std",
            torch.tensor(
                float(normalization_stats["log_l_std"]),
                dtype=torch.float32,
            ),
        )

    def normalize_original_reward(self, reward):
        return (
            reward.float() - self.reward_mean
        ) / self.reward_std.clamp_min(1e-6)

    def score_one(
        self,
        trajectory,
        original_reward,
        response_token_length,
    ):
        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory 必须是 [T, hidden_size]"
            )

        if trajectory.shape[1] != self.hidden_size:
            raise ValueError("hidden_size 不一致")

        hidden = F.layer_norm(
            trajectory.float(),
            normalized_shape=(self.hidden_size,),
        )

        z = self.prefix_probe(hidden).squeeze(-1)
        chunk_count = int(hidden.shape[0])

        normalized_original = (
            self.normalize_original_reward(
                original_reward
            )
        )

        log_t = torch.log1p(
            torch.tensor(
                float(chunk_count),
                device=hidden.device,
            )
        )
        log_l = torch.log1p(
            response_token_length.float()
        )

        normalized_t = (
            log_t - self.log_t_mean
        ) / self.log_t_std.clamp_min(1e-6)

        normalized_l = (
            log_l - self.log_l_mean
        ) / self.log_l_std.clamp_min(1e-6)

        if chunk_count < 2:
            return {
                "score": normalized_original,
                "normalized_original": (
                    normalized_original
                ),
                "z": z,
                "e": z.new_empty((0,)),
                "gate": z.new_empty((0,)),
                "r_traj": z.new_zeros(()),
                "alpha": z.new_ones(()),
            }

        evidence_change = z[1:] - z[:-1]

        gate_input = torch.cat(
            [
                hidden[:-1],
                hidden[1:],
                evidence_change[:, None],
            ],
            dim=-1,
        )
        gate = torch.sigmoid(
            self.gate_mlp(gate_input).squeeze(-1)
        )

        exponents = torch.arange(
            chunk_count - 2,
            -1,
            -1,
            device=hidden.device,
            dtype=torch.float32,
        )
        decay = torch.pow(
            torch.tensor(
                self.gamma,
                device=hidden.device,
                dtype=torch.float32,
            ),
            exponents,
        )

        trajectory_reward = (
            decay * gate * evidence_change
        ).sum() / (
            (decay * gate).sum() + 1e-6
        )

        alpha_input = torch.stack(
            [
                normalized_original,
                z[-1],
                trajectory_reward,
                normalized_t,
                normalized_l,
            ]
        )
        alpha = torch.sigmoid(
            self.alpha_mlp(alpha_input).squeeze()
        )

        score = (
            alpha * normalized_original
            + (1.0 - alpha) * trajectory_reward
        )

        return {
            "score": score,
            "normalized_original": (
                normalized_original
            ),
            "z": z,
            "e": evidence_change,
            "gate": gate,
            "r_traj": trajectory_reward,
            "alpha": alpha,
        }

    def forward(
        self,
        trajectories,
        original_rewards,
        response_token_lengths,
    ):
        outputs = []

        for index, trajectory in enumerate(
            trajectories
        ):
            outputs.append(
                self.score_one(
                    trajectory=trajectory,
                    original_reward=(
                        original_rewards[index]
                    ),
                    response_token_length=(
                        response_token_lengths[index]
                    ),
                )
            )

        scores = torch.stack([
            output["score"]
            for output in outputs
        ])

        return scores, outputs


def question_ranking_loss(
    scores,
    labels,
    config,
):
    method = config["default_method"]

    tau_list = float(method["tau_list"])
    tau_bt = float(method["tau_bt"])
    tau_cal = float(method["tau_cal"])
    lambda_bt = float(method["lambda_bt"])
    lambda_cal = float(method["lambda_cal"])

    positive_mask = labels == 1
    negative_mask = labels == 0

    if not positive_mask.any():
        raise ValueError("问题中没有正候选")

    if not negative_mask.any():
        raise ValueError("问题中没有负候选")

    scaled_scores = scores / tau_list

    listwise_loss = -(
        torch.logsumexp(
            scaled_scores[positive_mask],
            dim=0,
        )
        - torch.logsumexp(
            scaled_scores,
            dim=0,
        )
    )

    pair_differences = (
        scores[positive_mask][:, None]
        - scores[negative_mask][None, :]
    )

    bt_loss = F.softplus(
        -pair_differences / tau_bt
    ).mean()

    pair_probability = torch.sigmoid(
        pair_differences / tau_cal
    )
    brier_loss = (
        pair_probability - 1.0
    ).square().mean()

    total_loss = (
        listwise_loss
        + lambda_bt * bt_loss
        + lambda_cal * brier_loss
    )

    return {
        "total": total_loss,
        "listwise": listwise_loss,
        "bt": bt_loss,
        "brier": brier_loss,
    }


def run_self_test(config_path):
    config = json.loads(
        Path(config_path).read_text(encoding="utf-8")
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(20260829)

    stats = {
        "reward_mean": 0.0,
        "reward_std": 1.0,
        "log_t_mean": 1.5,
        "log_t_std": 0.5,
        "log_l_mean": 5.0,
        "log_l_std": 1.0,
    }

    model = TrajectoryCorrectionHead(
        config,
        stats,
    ).to(device)

    trajectories = [
        torch.randn(
            1,
            config["hidden_size"],
            device=device,
        ),
        torch.randn(
            4,
            config["hidden_size"],
            device=device,
        ),
        torch.randn(
            3,
            config["hidden_size"],
            device=device,
        ),
    ]
    rewards = torch.tensor(
        [1.0, -0.8, 0.4],
        device=device,
    )
    lengths = torch.tensor(
        [60.0, 180.0, 120.0],
        device=device,
    )
    labels = torch.tensor(
        [1, 0, 1],
        device=device,
    )

    scores, diagnostics = model(
        trajectories,
        rewards,
        lengths,
    )
    losses = question_ranking_loss(
        scores,
        labels,
        config,
    )

    losses["total"].backward()

    gradients_finite = all(
        parameter.grad is None
        or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    single_chunk_difference = abs(
        float(
            diagnostics[0]["score"].item()
            - diagnostics[0][
                "normalized_original"
            ].item()
        )
    )

    alpha_values = [
        float(item["alpha"].item())
        for item in diagnostics
    ]
    gate_values = torch.cat([
        item["gate"]
        for item in diagnostics
        if item["gate"].numel() > 0
    ])

    report = {
        "device": device,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "scores": [
            float(value)
            for value in scores.detach().cpu()
        ],
        "loss": {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        },
        "alpha_values": alpha_values,
        "gate_min": float(gate_values.min().item()),
        "gate_max": float(gate_values.max().item()),
        "single_chunk_score_difference": (
            single_chunk_difference
        ),
        "gradients_finite": bool(
            gradients_finite
        ),
        "prefix_bce_present": False,
    }

    if not torch.isfinite(losses["total"]):
        raise RuntimeError("loss 出现 NaN/Inf")

    if not gradients_finite:
        raise RuntimeError("梯度出现 NaN/Inf")

    if single_chunk_difference > 1e-7:
        raise RuntimeError(
            "T<2 时没有严格保持原始奖励"
        )

    if not all(
        0.0 <= value <= 1.0
        for value in alpha_values
    ):
        raise RuntimeError("alpha 超出 [0,1]")

    if not (
        0.0 <= report["gate_min"]
        <= report["gate_max"]
        <= 1.0
    ):
        raise RuntimeError("gate 超出 [0,1]")

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("轨迹校正头数学自检通过。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    parser.add_argument(
        "--config",
        default="configs/trajectory_head.json",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.config)
    else:
        parser.error("当前请使用 --self-test")
