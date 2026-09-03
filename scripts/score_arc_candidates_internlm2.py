from pathlib import Path
import sys


ROOT = Path("/root/autodl-tmp/rm_traj_project")
sys.path.insert(0, str(ROOT / "scripts"))

import score_internlm2_reward as adapter


adapter.base.CONFIG_PATH = (
    ROOT / "configs/"
    "arc_multi_reward_scoring_v1.json"
)


if __name__ == "__main__":
    adapter.base.main()
