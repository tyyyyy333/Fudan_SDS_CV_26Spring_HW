"""Data download helpers for HW3 datasets.

Provides commands to download:
  - CALVIN benchmark episodes (environments A, B, C, D)
  - Mip-NeRF 360 background scenes (garden, bicycle, counter, etc.)
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# ======================================================================
# CALVIN
# ======================================================================

CALVIN_DOWNLOAD_SCRIPT = r"""#!/usr/bin/env bash
# Download CALVIN dataset splits.
# Source: https://github.com/mees/calvin/blob/main/dataset/download_data.sh
set -euo pipefail

SAVE_DIR="${1:-$PWD}"
SPLIT="${2:-ABCD}"

BASE_URL="http://calvin.cs.uni-freiburg.de/dataset"

declare -A FILES
FILES["D"]="task_D_D.zip"
FILES["debug"]="calvin_debug_dataset.zip"
FILES["ABC"]="task_ABC_D.zip"
FILES["ABCD"]="task_ABCD_D.zip"

FILENAME="${FILES[$SPLIT]}"
if [ -z "$FILENAME" ]; then
    echo "Unknown split: $SPLIT (use D, ABC, ABCD, or debug)"
    exit 1
fi

URL="$BASE_URL/$FILENAME"
ZIP_PATH="$SAVE_DIR/$FILENAME"

if [ -d "$SAVE_DIR/training" ] && [ -d "$SAVE_DIR/validation" ]; then
    echo "[skip] CALVIN data already exists at $SAVE_DIR"
    exit 0
fi
if find "$SAVE_DIR" -maxdepth 2 -type d \( -name training -o -name validation \) | grep -q .; then
    echo "[skip] CALVIN data already exists under $SAVE_DIR"
    exit 0
fi

if [ -f "$ZIP_PATH" ] && [ ! -s "$ZIP_PATH" ]; then
    echo "Removing empty partial download: $ZIP_PATH"
    rm "$ZIP_PATH"
fi

echo "Downloading $FILENAME ..."
if command -v wget &>/dev/null; then
    wget -c --show-progress "$URL" -O "$ZIP_PATH"
elif command -v curl &>/dev/null; then
    curl --fail -L -C - -o "$ZIP_PATH" "$URL"
else
    echo "Need wget or curl to download."
    exit 1
fi

echo "Unzipping ..."
unzip -qo "$ZIP_PATH" -d "$SAVE_DIR"
rm "$ZIP_PATH"
echo "CALVIN ($SPLIT) ready at $SAVE_DIR"
"""

# Expected structure after unzipping follows CALVIN upstream:
#   data/calvin/task_ABCD_D/{training,validation}/
#   data/calvin/task_ABC_D/{training,validation}/
#   data/calvin/task_D_D/{training,validation}/
#   data/calvin/calvin_debug_dataset/{training,validation}/


def download_calvin_command(target_dir: Path, split: str = "ABCD") -> List[str]:
    """Build a command to download CALVIN data using the bash script above."""
    target_dir.mkdir(parents=True, exist_ok=True)
    script_path = target_dir / "_download_calvin.sh"
    script_path.write_text(CALVIN_DOWNLOAD_SCRIPT)
    script_path.chmod(0o755)
    return ["bash", str(script_path), str(target_dir), split]


# ======================================================================
# Mip-NeRF 360
# ======================================================================

# Mip-NeRF 360: all 9 scenes bundled in a single zip (~8 GB)
# After unzipping:  360_v2/garden/  360_v2/bicycle/  ...  360_v2/treehill/
MIPNERF360_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

MIPNERF360_SCENES = [
    "garden", "bicycle", "counter", "kitchen", "room", "stump", "bonsai",
    "flowers", "treehill",
]


def download_mipnerf360_command(target_dir: Path, scene: str = "garden") -> List[str]:
    """Download the Mip-NeRF 360 dataset and extract one scene.

    Downloads the full 360_v2.zip (~8 GB, ~30 min depending on network),
    unzips it, then symlinks the requested scene to target_dir/<scene>/.
    If you need multiple scenes, they're all in target_dir/360_v2/ after unzip.
    """
    if scene not in MIPNERF360_SCENES:
        raise ValueError(f"Unknown scene: {scene}. Choose from: {MIPNERF360_SCENES}")
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "360_v2.zip"
    extracted_dir = target_dir / "360_v2"
    scene_dir = target_dir / scene

    script = f"""#!/usr/bin/env bash
set -euo pipefail
if [ -d "{scene_dir}/images" ]; then
    echo "[skip] {scene} already ready at {scene_dir}"
    exit 0
fi
if [ -d "{target_dir}/{scene}/images" ]; then
    echo "[skip] {scene} already ready at {target_dir}/{scene}"
    exit 0
fi
if [ ! -d "{extracted_dir}" ]; then
    echo "Downloading Mip-NeRF 360 (~8 GB, this may take a while) ..."
    curl --fail -L -C - -o "{zip_path}" "{MIPNERF360_URL}"
    echo "Unzipping ..."
    unzip -qo "{zip_path}" -d "{target_dir}"
    rm "{zip_path}"
fi
if [ -d "{extracted_dir}/{scene}" ]; then
    ln -sfn "{extracted_dir}/{scene}" "{scene_dir}"
elif [ -d "{target_dir}/{scene}" ]; then
    true
else
    echo "Scene '{scene}' not found in archive."
    echo "Available: $(find {target_dir} -maxdepth 2 -type d -name images -printf '%h\n' | sort)"
    exit 1
fi
echo "{scene} ready at {scene_dir}"
"""
    script_path = target_dir / f"_download_{scene}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return ["bash", str(script_path)]


# ======================================================================
# Background removal (for object_c — Magic123 expects clean foreground)
# ======================================================================

def remove_background_rembg_command(image_path: Path, output_path: Path) -> List[str]:
    """Use rembg (https://github.com/danielgatis/rembg) to remove background.

    Install: pip install rembg
    """
    return [
        sys.executable, "-m", "rembg", "i",
        str(image_path), str(output_path),
    ]


def remove_background_clipdrop_command(image_path: Path, output_path: Path) -> str:
    """Alternative: manual instruction to use ClipDrop / Photoshop / etc."""
    return (
        f"# Remove background from {image_path} using any tool (ClipDrop, Photoshop, rembg, etc.)\n"
        f"# Save the foreground-only image to {output_path}\n"
        f"# CLI option: pip install rembg && rembg i {shlex.quote(str(image_path))} {shlex.quote(str(output_path))}"
    )
