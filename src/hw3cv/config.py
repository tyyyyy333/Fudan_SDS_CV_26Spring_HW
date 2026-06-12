from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OutputsConfig:
    task1: Path
    task2: Path


@dataclass(frozen=True)
class Task1ObjectConfig:
    output: Path
    source: Optional[Path] = None
    prompt: Optional[str] = None
    image: Optional[Path] = None


@dataclass(frozen=True)
class Task1BackgroundConfig:
    source: Path
    output: Path


@dataclass(frozen=True)
class Task1SceneConfig:
    manifest: Path
    render_output: Path


@dataclass(frozen=True)
class Task1Config:
    objects: Dict[str, Task1ObjectConfig]
    background: Task1BackgroundConfig
    scene: Task1SceneConfig


@dataclass(frozen=True)
class Task2ExperimentConfig:
    train_envs: List[str]
    eval_env: str
    output: Path


@dataclass(frozen=True)
class Task2ActConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    chunk_size: int


@dataclass(frozen=True)
class Task2Config:
    calvin_root: Path
    environments: Dict[str, Path]
    experiments: Dict[str, Task2ExperimentConfig]
    act: Task2ActConfig


@dataclass(frozen=True)
class HW3Config:
    project_root: Path
    external_dir: Path
    outputs: OutputsConfig
    task1: Task1Config
    task2: Task2Config


def default_config_path() -> Path:
    return Path.cwd() / "configs" / "hw3_default.json"


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_config(config_path: Optional[Path] = None) -> HW3Config:
    path = Path(config_path) if config_path else default_config_path()
    project_root = Path.cwd() if config_path is None else path.parent
    data = json.loads(path.read_text(encoding="utf-8"))

    outputs = OutputsConfig(
        task1=_resolve_path(project_root, data["outputs"]["task1"]),
        task2=_resolve_path(project_root, data["outputs"]["task2"]),
    )

    task1_data = data["task1"]
    objects = {
        name: Task1ObjectConfig(
            source=_optional_path(project_root, value.get("source")),
            prompt=value.get("prompt"),
            image=_optional_path(project_root, value.get("image")),
            output=_resolve_path(project_root, value["output"]),
        )
        for name, value in task1_data["objects"].items()
    }
    task1 = Task1Config(
        objects=objects,
        background=Task1BackgroundConfig(
            source=_resolve_path(project_root, task1_data["background"]["source"]),
            output=_resolve_path(project_root, task1_data["background"]["output"]),
        ),
        scene=Task1SceneConfig(
            manifest=_resolve_path(project_root, task1_data["scene"]["manifest"]),
            render_output=_resolve_path(project_root, task1_data["scene"]["render_output"]),
        ),
    )

    task2_data = data["task2"]
    task2 = Task2Config(
        calvin_root=_resolve_path(project_root, task2_data["calvin_root"]),
        environments={
            name: _resolve_path(project_root, env_path)
            for name, env_path in task2_data["environments"].items()
        },
        experiments={
            name: Task2ExperimentConfig(
                train_envs=list(value["train_envs"]),
                eval_env=value["eval_env"],
                output=_resolve_path(project_root, value["output"]),
            )
            for name, value in task2_data["experiments"].items()
        },
        act=Task2ActConfig(
            batch_size=int(task2_data["act"]["batch_size"]),
            learning_rate=float(task2_data["act"]["learning_rate"]),
            epochs=int(task2_data["act"]["epochs"]),
            chunk_size=int(task2_data["act"]["chunk_size"]),
        ),
    )

    return HW3Config(
        project_root=project_root,
        external_dir=_resolve_path(project_root, data["external_dir"]),
        outputs=outputs,
        task1=task1,
        task2=task2,
    )


def _optional_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return _resolve_path(project_root, value)
