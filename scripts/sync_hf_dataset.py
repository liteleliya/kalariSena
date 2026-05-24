from __future__ import annotations

import argparse
import fnmatch
import os
import re
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_DOWNLOAD_PATTERNS = ["*.mp4", "**/*.mp4"]
DEFAULT_UPLOAD_PATTERNS = [
    "**/*_retarget_g1.csv",
    "**/*_retarget_g1_from_bvh.csv",
    "**/*_g1_retarget.mp4",
]
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_SECONDS = 5.0
DEFAULT_RETRY_MAX_SECONDS = 300.0

_RATE_LIMIT_T_RE = re.compile(r"t=(\d+)")


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


def _extract_retry_after(headers) -> float | None:
    if not headers:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            return None

    rate_limit = headers.get("RateLimit")
    if rate_limit:
        matches = [int(m) for m in _RATE_LIMIT_T_RE.findall(rate_limit)]
        if matches:
            return float(max(matches))
    return None


def _is_rate_limited(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    if resp is None:
        return False
    return getattr(resp, "status_code", None) == 429


def _call_with_retry(
    fn,
    label: str,
    max_retries: int,
    base_delay: float,
    max_delay: float,
):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limited(exc) or attempt >= max_retries:
                raise

            resp = getattr(exc, "response", None)
            retry_after = _extract_retry_after(getattr(resp, "headers", None))
            delay = retry_after if retry_after is not None else base_delay * (2**attempt)
            delay = min(max_delay, max(base_delay, delay))
            print(
                f"[RateLimit] {label} hit 429. Sleeping {delay:.1f}s before retry "
                f"{attempt + 1}/{max_retries}."
            )
            time.sleep(delay)
            attempt += 1


def _download_videos(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: Path,
    remote_prefix: str,
    patterns: list[str],
    token: str,
    list_only: bool,
) -> int:
    all_files = _call_with_retry(
        lambda: api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        ),
        label="list_repo_files",
        max_retries=DEFAULT_MAX_RETRIES,
        base_delay=DEFAULT_RETRY_BASE_SECONDS,
        max_delay=DEFAULT_RETRY_MAX_SECONDS,
    )

    prefix = _normalize_prefix(remote_prefix)
    matched = []
    for file_path in all_files:
        if prefix and not file_path.startswith(prefix + "/") and file_path != prefix:
            continue
        rel_for_match = file_path[len(prefix) + 1 :] if prefix and file_path.startswith(prefix + "/") else file_path
        if _match_any(rel_for_match, patterns) or _match_any(file_path, patterns):
            matched.append(file_path)

    if not list_only:
        local_dir.mkdir(parents=True, exist_ok=True)

    for file_path in matched:
        if list_only:
            out_path = local_dir / file_path
            print(f"[DRY RUN] Download: {file_path} -> {out_path}")
            continue

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
    upload_mode: str,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> int:
    if not local_root.exists():
        raise SystemExit(f"Upload source directory not found: {local_root}")

    candidates = _iter_local_files(local_root, patterns)
    if not candidates:
        print("No local files matched upload patterns.")
        return 0

    if upload_mode == "folder" and skip_existing:
        print("[Upload] skip-existing is not supported in folder mode. Falling back to per-file.")
        upload_mode = "file"

    existing_remote: set[str] = set()
    if skip_existing:
        existing_remote = set(
            _call_with_retry(
                lambda: api.list_repo_files(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                ),
                label="list_repo_files",
                max_retries=max_retries,
                base_delay=retry_base_seconds,
                max_delay=retry_max_seconds,
            )
        )

    prefix = _normalize_prefix(remote_prefix)
    uploaded = 0

    if upload_mode == "folder":
        if dry_run:
            for local_path in candidates:
                rel = local_path.relative_to(local_root).as_posix()
                remote_path = f"{prefix}/{rel}" if prefix else rel
                print(f"[DRY RUN] Upload: {local_path} -> {remote_path}")
            return len(candidates)

        path_in_repo = prefix or None
        _call_with_retry(
            lambda: api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(local_root),
                path_in_repo=path_in_repo,
                revision=revision,
                token=token,
                commit_message=commit_message,
                allow_patterns=patterns,
            ),
            label="upload_folder",
            max_retries=max_retries,
            base_delay=retry_base_seconds,
            max_delay=retry_max_seconds,
        )
        return len(candidates)

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

        _call_with_retry(
            lambda: api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                commit_message=commit_message,
            ),
            label=f"upload_file:{remote_path}",
            max_retries=max_retries,
            base_delay=retry_base_seconds,
            max_delay=retry_max_seconds,
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
    parser.add_argument(
        "--list-download",
        action="store_true",
        help="List matching remote files and local targets without downloading.",
    )

    parser.add_argument("--upload-source", default="cloud_outputs")
    parser.add_argument("--remote-retarget-prefix", default="retargeted_g1")
    parser.add_argument(
        "--upload-patterns",
        default=",".join(DEFAULT_UPLOAD_PATTERNS),
        help="Comma-separated glob patterns matched against local relative file paths",
    )
    parser.add_argument(
        "--upload-mode",
        default="folder",
        choices=["folder", "file"],
        help="Upload strategy: folder (batched) or file (per-file).",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit-message", default="Upload retargeted G1 references")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-base-seconds", type=float, default=DEFAULT_RETRY_BASE_SECONDS)
    parser.add_argument("--retry-max-seconds", type=float, default=DEFAULT_RETRY_MAX_SECONDS)
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
            list_only=args.list_download,
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
            upload_mode=args.upload_mode,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
        )

    print("Summary:")
    print(f"  Downloaded files: {downloaded_count}")
    print(f"  Uploaded files:   {uploaded_count}")


if __name__ == "__main__":
    main()
