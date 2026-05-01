from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(">> " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _find_retarget_csv(output_root: Path, motion_id: str) -> Path:
    base = output_root / motion_id
    candidates = [
        base / f"{motion_id}_retarget_g1.csv",
        base / f"{motion_id}_retarget_g1_from_bvh.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    matches = list(output_root.rglob(f"{motion_id}_retarget_g1*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Retarget CSV not found under {output_root}. Expected {candidates[0]}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--motion-id", type=str, default=None)

    parser.add_argument("--gemx-dir", type=str, default="GEM-X")
    parser.add_argument("--output-root", type=str, default="cloud_outputs")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--static-cam", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-onnx", action="store_true")
    parser.add_argument("--no-retarget", action="store_true")

    parser.add_argument("--skip-gemx", action="store_true")
    parser.add_argument("--skip-annotate", action="store_true")

    parser.add_argument("--urdf", type=str, default="assets/unitree_g1/g1.urdf")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--root-euler-order", type=str, default="xyz")
    parser.add_argument("--angles-deg", action="store_true")
    parser.add_argument("--angles-rad", action="store_true")
    parser.add_argument("--quat-order", type=str, default="xyzw")
    parser.add_argument("--include-from-bvh", action="store_true")

    parser.add_argument("--output-dir", type=str, default="data/motions_retargeted")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    motion_id = args.motion_id or video_path.stem

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_gemx:
        gemx_dir = Path(args.gemx_dir)
        if not gemx_dir.is_absolute():
            gemx_dir = ROOT / gemx_dir
        if not gemx_dir.exists():
            raise SystemExit(f"GEM-X directory not found: {gemx_dir}")

        script_name = "demo_soma_onnx.py" if args.use_onnx else "demo_soma.py"
        demo_script = gemx_dir / "scripts" / "demo" / script_name
        if not demo_script.exists():
            raise SystemExit(f"GEM-X demo not found: {demo_script}")

        cmd = [
            sys.executable,
            str(demo_script),
            "--video",
            str(video_path),
            "--output_root",
            str(output_root),
        ]
        if not args.no_retarget:
            cmd.append("--retarget")
        if args.ckpt:
            cmd.extend(["--ckpt", args.ckpt])
        if args.static_cam:
            cmd.append("--static_cam")
        if args.verbose:
            cmd.append("--verbose")

        _run(cmd, cwd=gemx_dir)

    if args.skip_annotate:
        print("Annotation skipped. Pipeline done.")
        return

    if args.angles_rad:
        args.angles_deg = False

    retarget_csv = _find_retarget_csv(output_root, motion_id)

    annotate_script = ROOT / "scripts" / "annotate_motion_library.py"
    if not annotate_script.exists():
        raise SystemExit(f"Annotation script not found: {annotate_script}")

    cmd = [
        sys.executable,
        str(annotate_script),
        "--csv-root",
        str(output_root),
        "--output-dir",
        str(Path(args.output_dir)),
        "--splits-dir",
        str(Path(args.splits_dir)),
        "--urdf",
        args.urdf,
        "--fps",
        str(args.fps),
        "--pos-scale",
        str(args.pos_scale),
        "--root-euler-order",
        args.root_euler_order,
        "--quat-order",
        args.quat_order,
    ]
    if args.angles_deg:
        cmd.append("--angles-deg")
    if args.angles_rad:
        cmd.append("--angles-rad")
    if args.include_from_bvh:
        cmd.append("--include-from-bvh")

    _run(cmd)

    print("Pipeline complete.")
    print(f"Retarget CSV: {retarget_csv}")
    print(f"Annotated NPZ dir: {args.output_dir}")


if __name__ == "__main__":
    main()
