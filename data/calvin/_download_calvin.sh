#!/usr/bin/env bash
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
