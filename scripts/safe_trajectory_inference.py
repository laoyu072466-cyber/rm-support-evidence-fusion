import numpy as np
import torch


def safe_residual_score(
    normalized_original,
    learned_score,
    beta=0.1,
):
    if not 0.0 <= beta <= 1.0:
        raise ValueError(
            f"beta 必须位于 [0, 1]，实际为 {beta}"
        )

    return (
        normalized_original
        + beta
        * (learned_score - normalized_original)
    )


def ensemble_learned_scores(
    learned_scores,
    method="mean",
):
    if len(learned_scores) == 0:
        raise ValueError("至少需要一个校正头分数")

    first = learned_scores[0]

    if torch.is_tensor(first):
        stacked = torch.stack(learned_scores, dim=0)

        if method == "mean":
            return stacked.mean(dim=0)
        if method == "median":
            return stacked.median(dim=0).values

    else:
        stacked = np.stack(learned_scores, axis=0)

        if method == "mean":
            return stacked.mean(axis=0)
        if method == "median":
            return np.median(stacked, axis=0)

    raise ValueError(f"未知集成方式：{method}")


def safe_ensemble_score(
    normalized_original,
    learned_scores,
    beta=0.1,
    ensemble_method="mean",
):
    ensemble = ensemble_learned_scores(
        learned_scores,
        method=ensemble_method,
    )
    return safe_residual_score(
        normalized_original,
        ensemble,
        beta=beta,
    )


def self_test():
    original = torch.tensor([1.0, -1.0])
    heads = [
        torch.tensor([3.0, 1.0]),
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0, 2.0]),
    ]

    ensemble = ensemble_learned_scores(heads)
    safe = safe_ensemble_score(
        original,
        heads,
        beta=0.1,
    )

    assert torch.allclose(
        ensemble,
        torch.tensor([2.0, 2.0]),
    )
    assert torch.allclose(
        safe,
        torch.tensor([1.1, -0.7]),
    )
    assert torch.allclose(
        safe_ensemble_score(
            original,
            heads,
            beta=0.0,
        ),
        original,
    )
    assert torch.allclose(
        safe_ensemble_score(
            original,
            heads,
            beta=1.0,
        ),
        ensemble,
    )

    print("安全残差推理组件自检通过。")


if __name__ == "__main__":
    self_test()
