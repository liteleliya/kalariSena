from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_DOWNLOAD_PATTERNS = ["*.mp4", "**/*.mp4"]
DEFAULT_UPLOAD_PATTERNS = [
    "**/*_retarget_g1.csv",
    "**/*_retarget_g1_from_bvh.csv",
    "**/*_g1_retarget.mp4",
]


def _split_csv_patterns(raw: str) -> list[str]:
    items = [p.strip() for p in raw.split(",")]
    return [p for p in items if p]


def _match_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _resolve_token(token_arg: str | None) -> str:
    token = token_arg or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Provide --token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
        )
    return token


def _download_videos(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: Path,
    remote_prefix: str,
    patterns: list[str],
    token: str,
) -> int:
    all_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    prefix = _normalize_prefix(remote_prefix)
    matched = []
    for file_path in all_files:
        if prefix and not file_path.startswith(prefix + "/") and file_path != prefix:
            continue
        rel_for_match = file_path[len(prefix) + 1 :] if prefix and file_path.startswith(prefix + "/") else file_path
        if _match_any(rel_for_match, patterns) or _match_any(file_path, patterns):
            matched.append(file_path)

    local_dir.mkdir(parents=True, exist_ok=True)

    for file_path in matched:
        out_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_path,
            revision=revision,
            token=token,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded: {file_path} -> {out_path}")

    if not matched:
        print("No remote files matched download patterns.")
    return len(matched)


def _iter_local_files(root: Path, patterns: list[str]) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    out: list[Path] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        if _match_any(rel, patterns) or _match_any(p.name, patterns):
            out.append(p)
    return sorted(out)


def _upload_retarget_outputs(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_root: Path,
    remote_prefix: str,
    patterns: list[str],
    token: str,
    skip_existing: bool,
    dry_run: bool,
    commit_message: str,
) -> int:
    if not local_root.exists():
        raise SystemExit(f"Upload source directory not found: {local_root}")

    candidates = _iter_local_files(local_root, patterns)
    if not candidates:
        print("No local files matched upload patterns.")
        return 0

    existing_remote: set[str] = set()
    if skip_existing:
        existing_remote = set(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )
        )

    prefix = _normalize_prefix(remote_prefix)
    uploaded = 0

    for local_path in candidates:
        rel = local_path.relative_to(local_root).as_posix()
        remote_path = f"{prefix}/{rel}" if prefix else rel

        if skip_existing and remote_path in existing_remote:
            print(f"Skip existing: {remote_path}")
            continue

        if dry_run:
            print(f"[DRY RUN] Upload: {local_path} -> {remote_path}")
            uploaded += 1
            continue

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            commit_message=commit_message,
        )
        print(f"Uploaded: {local_path} -> {remote_path}")
        uploaded += 1

    return uploaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MP4s from a private HF dataset and upload retargeted outputs."
    )
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. user/my-private-dataset")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")

    parser.add_argument("--mode", default="both", choices=["download", "upload", "both"])

    parser.add_argument("--remote-video-prefix", default="videos")
    parser.add_argument("--download-dir", default="inputs/hf_videos")
    parser.add_argument(
        "--download-patterns",
        default=",".join(DEFAULT_DOWNLOAD_PATTERNS),
        help="Comma-separated glob patterns matched against remote file paths",
    )

    parser.add_argument("--upload-source", default="cloud_outputs")
    parser.add_argument("--remote-retarget-prefix", default="retargeted_g1")
    parser.add_argument(
        "--upload-patterns",
        default=",".join(DEFAULT_UPLOAD_PATTERNS),
        help="Comma-separated glob patterns matched against local relative file paths",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit-message", default="Upload retargeted G1 references")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    token = _resolve_token(args.token)
    api = HfApi(token=token)

    download_patterns = _split_csv_patterns(args.download_patterns)
    upload_patterns = _split_csv_patterns(args.upload_patterns)

    download_dir = Path(args.download_dir).expanduser().resolve()
    upload_source = Path(args.upload_source).expanduser().resolve()

    downloaded_count = 0
    uploaded_count = 0

    if args.mode in {"download", "both"}:
        downloaded_count = _download_videos(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=download_dir,
            remote_prefix=args.remote_video_prefix,
            patterns=download_patterns,
            token=token,
        )

    if args.mode in {"upload", "both"}:
        uploaded_count = _upload_retarget_outputs(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_root=upload_source,
            remote_prefix=args.remote_retarget_prefix,
            patterns=upload_patterns,
            token=token,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            commit_message=args.commit_message,
        )

    print("Summary:")
    print(f"  Downloaded files: {downloaded_count}")
    print(f"  Uploaded files:   {uploaded_count}")


if __name__ == "__main__":
    main()
