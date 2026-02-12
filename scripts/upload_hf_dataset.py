#!/usr/bin/env python3
"""
Upload a local folder (default: videos/) to a Hugging Face Dataset repo.

Example:
  python scripts/upload_hf_dataset.py \
    --repo-id kv-compression/deep-forcing-ablation \
    --local-dir videos
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        default="kv-compression/deep-forcing-ablation",
        help="Target HF repo id, e.g. org/name",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("videos"),
        help="Local folder to upload",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Destination path inside the dataset repo (empty means repo root)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token. If omitted, uses HF_TOKEN/HUGGING_FACE_HUB_TOKEN env or cached login.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload Deep Forcing ablation videos",
        help="Commit message for upload.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Upload all files (default uploads common dataset artifacts only).",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def call_with_supported_kwargs(method, kwargs: dict):
    """
    Call method with only the kwargs supported by the installed huggingface_hub version.
    """
    sig = inspect.signature(method)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = [k for k in kwargs.keys() if k not in accepted]
    if dropped:
        print(f"[upload] {method.__name__} does not support args: {dropped}; skipping them.")
    return method(**filtered)


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir.resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    token = resolve_token(args.token)
    api = HfApi(token=token)

    # Create dataset repo if missing.
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    allow_patterns = None
    if not args.include_all:
        allow_patterns = [
            "*.mp4",
            "*.csv",
            "*.txt",
            "*.json",
            "*.md",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.webp",
        ]

    path_in_repo = args.path_in_repo.strip("/")
    common_kwargs = dict(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        commit_message=args.commit_message,
    )

    # Prefer upload_large_folder when available for large trees.
    if hasattr(api, "upload_large_folder"):
        try:
            call_with_supported_kwargs(api.upload_large_folder, common_kwargs)
        except TypeError:
            # Older versions may have stricter signatures; fallback to upload_folder.
            call_with_supported_kwargs(api.upload_folder, common_kwargs)
    else:
        call_with_supported_kwargs(api.upload_folder, common_kwargs)

    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
