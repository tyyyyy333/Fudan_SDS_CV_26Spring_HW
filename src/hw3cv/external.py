from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ExternalRepository:
    name: str
    url: str


EXTERNAL_REPOSITORIES: Dict[str, ExternalRepository] = {
    "2d-gaussian-splatting": ExternalRepository(
        name="2d-gaussian-splatting",
        url="https://github.com/hbb1/2d-gaussian-splatting.git",
    ),
    "threestudio": ExternalRepository(
        name="threestudio",
        url="https://github.com/threestudio-project/threestudio.git",
    ),
    "Magic123": ExternalRepository(
        name="Magic123",
        url="https://github.com/guochengqian/Magic123.git",
    ),
    "lerobot": ExternalRepository(
        name="lerobot",
        url="https://github.com/huggingface/lerobot.git",
    ),
    "calvin": ExternalRepository(
        name="calvin",
        url="https://github.com/mees/calvin.git",
    ),
    "tiny-cuda-nn": ExternalRepository(
        name="tiny-cuda-nn",
        url="https://github.com/NVlabs/tiny-cuda-nn.git",
    ),
}


def bootstrap_commands(external_dir: Path) -> List[List[str]]:
    commands: List[List[str]] = [["mkdir", "-p", str(external_dir)]]
    for repository in EXTERNAL_REPOSITORIES.values():
        target = external_dir / repository.name
        if not target.exists():
            commands.append(["git", "clone", repository.url, str(target)])
    return commands
