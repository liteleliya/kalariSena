from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _find_videos(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full private HF dataset flow: download MP4 clips, retarget each clip, "
            "and upload G1 references back to HF."
        )
    )

    parser.add_argument("--repo-id", required=True, help="HF dataset id, e.g. user/my-dataset")
    parser.add_argument("--revision", default="main")

    parser.add_argument("--mode", default="all", choices=["all", "pull", "process", "push"])

    parser.add_argument("--remote-video-prefix", default="videos")
    parser.add_argument("--local-video-dir", default="inputs/hf_videos")
    parser.add_argument("--video-pattern", default="*.mp4")

    parser.add_argument("--output-root", default="cloud_outputs")
    parser.add_argument("--gemx-dir", default="GEM-X")
    parser.add_argument("--urdf", default="assets/unitree_g1/g1.urdf")
    parser.add_argument("--angles-deg", action="store_true")
    parser.add_argument("--angles-rad", action="store_true")
    parser.add_argument("--include-from-bvh", action="store_true")
    parser.add_argument("--static-cam", action="store_true")
    parser.add_argument("--use-onnx", action="store_true")
    parser.add_argument("--no-retarget", action="store_true")
    parser.add_argument("--skip-existing-motion", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--preserve-subdirs",
        action="store_true",
        help="Preserve dataset subfolder layout under output_root to avoid name collisions.",
    )

    parser.add_argument("--remote-retarget-prefix", default="retargeted_g1")
    parser.add_argument("--skip-existing-remote", action="store_true")
    parser.add_argument("--dry-run-upload", action="store_true")

    return parser.parse_args()


def _sync_download(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sync_hf_dataset.py"),
        "--repo-id",
        args.repo_id,
        "--revision",
        args.revision,
        "--mode",
        "download",
        "--remote-video-prefix",
        args.remote_video_prefix,
        "--download-dir",
        args.local_video_dir,
    ]
    _run(cmd, cwd=ROOT)


def _process_all(args: argparse.Namespace) -> None:
    video_root = Path(args.local_video_dir).expanduser()
    if not video_root.is_absolute():
        video_root = ROOT / video_root
    if not video_root.exists():
        raise SystemExit(f"Video directory not found: {video_root}")

    output_root = Path(args.output_root).expanduser()
    output_root_is_abs = output_root.is_absolute()
    if not output_root_is_abs:
        output_root = ROOT / output_root

    videos = _find_videos(video_root, args.video_pattern)
    if not videos:
        raise SystemExit(f"No videos matched '{args.video_pattern}' under {video_root}")

    failures = 0
    for idx, video in enumerate(videos, start=1):
        rel_dir = Path(".")
        if args.preserve_subdirs:
            rel_dir = video.relative_to(video_root).parent

        per_video_root = output_root / rel_dir
        output_root_arg = per_video_root if output_root_is_abs else Path(args.output_root) / rel_dir

        motion_id = video.stem
        expected_csv = per_video_root / motion_id / f"{motion_id}_retarget_g1.csv"

        if args.skip_existing_motion and expected_csv.exists():
            print(f"[{idx}/{len(videos)}] Skip existing motion: {motion_id}")
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_pipeline.py"),
            "--video",
            str(video),
            "--motion-id",
            motion_id,
            "--output-root",
            str(output_root_arg),
            "--gemx-dir",
            args.gemx_dir,
            "--urdf",
            args.urdf,
        ]

        if args.angles_deg:
            cmd.append("--angles-deg")
        if args.angles_rad:
            cmd.append("--angles-rad")
        if args.include_from_bvh:
            cmd.append("--include-from-bvh")
        if args.static_cam:
            cmd.append("--static-cam")
        if args.use_onnx:
            cmd.append("--use-onnx")
        if args.no_retarget:
            cmd.append("--no-retarget")

        print(f"[{idx}/{len(videos)}] Processing {video.name}")
        try:
            _run(cmd, cwd=ROOT)
        except subprocess.CalledProcessError:
            failures += 1
            if not args.continue_on_error:
                raise
            print(f"Failed for {video.name}; continuing due to --continue-on-error")

    if failures:
        print(f"Processing finished with {failures} failure(s).")
    else:
        print("Processing finished with no failures.")


def _sync_upload(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sync_hf_dataset.py"),
        "--repo-id",
        args.repo_id,
        "--revision",
        args.revision,
        "--mode",
        "upload",
        "--upload-source",
        args.output_root,
        "--remote-retarget-prefix",
        args.remote_retarget_prefix,
    ]

    if args.skip_existing_remote:
        cmd.append("--skip-existing")
    if args.dry_run_upload:
        cmd.append("--dry-run")

    _run(cmd, cwd=ROOT)


def main() -> None:
    args = _parse_args()

    if args.mode in {"all", "pull"}:
        _sync_download(args)

    if args.mode in {"all", "process"}:
        _process_all(args)

    if args.mode in {"all", "push"}:
        _sync_upload(args)

    print("HF end-to-end flow complete.")


if __name__ == "__main__":
    main()
