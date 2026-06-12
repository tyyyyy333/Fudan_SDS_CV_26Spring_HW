#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D")
    shard_paths = [root / f"subset_manifest.shard_{i}.json" for i in range(4)]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing shard manifests: " + ", ".join(missing))
    shard_manifests = [json.loads(p.read_text()) for p in shard_paths]
    base = dict(shard_manifests[0])
    shards = []
    for manifest in shard_manifests:
        shards.extend(manifest.get("shards", []))
    base["selected_shards"] = [0, 1, 2, 3]
    base["shards"] = sorted(shards, key=lambda item: item["environment"])
    base["actual_size_gb"] = round(dir_size(root) / 1024**3, 3)
    base["actual_frames"] = sum(s.get("written_frames", 0) for s in base["shards"])
    base["merged_from"] = [str(p) for p in shard_paths]
    (root / "subset_manifest.json").write_text(json.dumps(base, indent=2) + "\n")
    print(json.dumps({"root": str(root), "actual_size_gb": base["actual_size_gb"], "actual_frames": base["actual_frames"]}, indent=2))


if __name__ == "__main__":
    main()
