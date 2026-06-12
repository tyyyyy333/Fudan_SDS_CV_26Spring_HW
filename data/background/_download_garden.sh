#!/usr/bin/env bash
set -euo pipefail
if [ -d "data/background/garden/images" ]; then
    echo "[skip] garden already ready at data/background/garden"
    exit 0
fi
if [ ! -d "data/background/360_v2" ]; then
    echo "Downloading Mip-NeRF 360 (~8 GB, this may take a while) ..."
    curl -L -o "data/background/360_v2.zip" "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    echo "Unzipping ..."
    unzip -qo "data/background/360_v2.zip" -d "data/background"
    rm "data/background/360_v2.zip"
fi
if [ ! -d "data/background/360_v2/garden" ]; then
    echo "Scene 'garden' not found in archive."
    echo "Available: $(ls data/background/360_v2)"
    exit 1
fi
# symlink for convenience
ln -sfn "data/background/360_v2/garden" "data/background/garden"
echo "garden ready at data/background/garden"
