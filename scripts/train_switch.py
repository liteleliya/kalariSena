"""Stage F: Switch integration.
Integrates nominal + fall + recovery policies with ModeSwitch.
Start with threshold-based switching (configs/switch.yaml), not learned switch.
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
    parser.add_argument("--config", type=str, default="configs/switch.yaml")
    parser.add_argument("--nominal", type=str, default="models/momentum_best.pt")
    parser.add_argument("--fall", type=str, default="models/fall_best.pt")
    parser.add_argument("--recovery", type=str, default="models/recovery_best.pt")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    assert "delta1" in cfg, "Expected delta1 in switch config"

    raise SystemExit(
        "TODO: load all three policy checkpoints; instantiate ModeSwitch; "
        "run integrated eval loop and log transitions."
    )


if __name__ == "__main__":
    main()
