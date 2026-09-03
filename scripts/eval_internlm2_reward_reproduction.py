from pathlib import Path
import sys


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import eval_multi_reward_reproduction as base


base.MODEL_SPECS["internlm2_1p8b"] = {
    "name": "InternLM2-1.8B-Reward",
    "score_root": (
        ROOT
        / "data/cache/reward_scores_full_v1/"
        "InternLM2-1.8B-Reward"
    ),
    "score_manifest": (
        ROOT
        / "data/manifests/"
        "multi_reward_scores_internlm2_1p8b_v1.json"
    ),
}


if __name__ == "__main__":
    base.main()
