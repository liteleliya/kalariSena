"""Stage A launcher for unitree_rl_mjlab G1 tracking training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJLAB_ROOT = PROJECT_ROOT / "unitree_rl_mjlab"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/tracking.yaml")
    parser.add_argument("--motion-file", type=Path, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--gpu-ids", type=str, default=None, help="Example: 0 or '0 1' or all")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _coerce_value(value: str) -> object:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_config(path: Path) -> dict:
    """Load the small Stage A YAML subset without adding a dependency."""
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    cfg: dict[str, object] = {}
    current_section: dict[str, object] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                cfg[key] = _coerce_value(value)
                current_section = None
            else:
                current_section = {}
                cfg[key] = current_section
        elif current_section is not None:
            key, _, value = line.strip().partition(":")
            current_section[key.strip()] = _coerce_value(value)
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    if not MJLAB_ROOT.exists():
        raise SystemExit(
            f"Missing unitree_rl_mjlab at {MJLAB_ROOT}. Run scripts/install_subprojects.sh first."
        )

    task = args.task or cfg.get("task", "Unitree-G1-Tracking-No-State-Estimation")
    motion_file = args.motion_file or PROJECT_ROOT / cfg.get("motion_file", "")
    motion_file = motion_file.expanduser().resolve()
    if not motion_file.exists() and not args.dry_run:
        raise SystemExit(
            f"Motion file not found: {motion_file}\n"
            "Generate it first with scripts/export_stage_a_motions.py."
        )

    num_envs = args.num_envs or int((cfg.get("env") or {}).get("num_envs", 4096))
    max_iterations = args.max_iterations or int(
        (cfg.get("agent") or {}).get("max_iterations", 30001)
    )

    cmd = [
        sys.executable,
        "scripts/train.py",
        task,
        f"--motion_file={motion_file}",
        f"--env.scene.num-envs={num_envs}",
        f"--agent.max-iterations={max_iterations}",
    ]
    if args.gpu_ids:
        cmd.append("--gpu-ids")
        cmd.extend(args.gpu_ids.split())
    if args.video:
        cmd.append("--video")

    print("Running:")
    print(" ".join(str(part) for part in cmd))
    if not motion_file.exists():
        print(f"[dry-run warning] Motion file does not exist yet: {motion_file}")
    if args.dry_run:
        return

    subprocess.run(cmd, cwd=MJLAB_ROOT, check=True)


if __name__ == "__main__":
    main()
