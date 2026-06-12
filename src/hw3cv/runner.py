from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


Command = List[str]


@dataclass(frozen=True)
class CommandResult:
    command: Command
    returncode: int


def format_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_or_print(command: Command, execute: bool = False, cwd: Optional[Path] = None) -> CommandResult:
    print(format_command(command))
    if not execute:
        return CommandResult(command=command, returncode=0)
    completed = subprocess.run(command, cwd=cwd, check=False)
    return CommandResult(command=command, returncode=completed.returncode)
