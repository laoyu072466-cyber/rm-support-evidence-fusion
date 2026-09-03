from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_full_candidates_multi_reward as base


base.CONFIG_PATH = (
    ROOT / "configs/"
    "arc_multi_reward_scoring_v1.json"
)


if __name__ == "__main__":
    base.main()
