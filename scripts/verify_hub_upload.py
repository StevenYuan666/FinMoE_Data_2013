"""Confirm every local shard is on the Hub with identical bytes.

The Hub stores a SHA-256 for each LFS file, so integrity can be proven without
downloading anything: hash the local shard and compare against the recorded
LFS oid. Falls back to a size comparison for any file the Hub did not track as
LFS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = HfApi()
    info = api.dataset_info(args.repo_id, files_metadata=True)
    remote = {sibling.rfilename: sibling for sibling in info.siblings}

    local_shards = sorted((args.output_root / "data" / "train").glob("*.parquet"))
    problems: list[str] = []
    checked = 0
    hashed = 0
    size_only = 0

    for shard in local_shards:
        path_in_repo = f"data/train/{shard.name}"
        sibling = remote.get(path_in_repo)
        if sibling is None:
            problems.append(f"{path_in_repo}: missing on the Hub")
            continue
        checked += 1

        local_size = shard.stat().st_size
        if sibling.size is not None and sibling.size != local_size:
            problems.append(
                f"{path_in_repo}: size {sibling.size} on Hub != {local_size} local"
            )

        lfs = getattr(sibling, "lfs", None)
        remote_sha = None
        if lfs is not None:
            remote_sha = getattr(lfs, "sha256", None) or (
                lfs.get("sha256") if isinstance(lfs, dict) else None
            )
        if remote_sha:
            if sha256_file(shard) != remote_sha:
                problems.append(f"{path_in_repo}: sha256 mismatch")
            hashed += 1
        else:
            size_only += 1

    extra = sorted(
        name
        for name in remote
        if name.startswith("data/train/")
        and name.endswith(".parquet")
        and name not in {f"data/train/{shard.name}" for shard in local_shards}
    )
    if extra:
        problems.append(f"unexpected extra shards on the Hub: {extra}")

    result = {
        "repo_id": args.repo_id,
        "local_shards": len(local_shards),
        "checked": checked,
        "verified_by_sha256": hashed,
        "verified_by_size_only": size_only,
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
