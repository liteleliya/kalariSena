"""Stage B: CoM recoverability refinement.
Loads tracking_best.pt, activates com_support observation block,
adds Stage B reward terms, oversamples single_support and stable_stance families.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit("Missing dependency: pyyaml. Install it before running.") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/com.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/tracking_best.pt")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg["checkpoint"] = args.checkpoint

    assert "obs" in cfg, "Expected 'obs' block in config"
    assert "rewards" in cfg, "Expected 'rewards' block in config"

    raise SystemExit(
        "TODO: connect to MuJoCo env loop; implement family-aware sampler; "
        "save com_best.pt on val improvement."
    )


if __name__ == "__main__":
    main()
