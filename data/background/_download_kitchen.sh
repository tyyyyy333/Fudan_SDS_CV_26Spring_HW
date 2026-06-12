#!/usr/bin/env bash
set -euo pipefail
if [ -d "data/background/kitchen/images" ]; then
    echo "[skip] kitchen already ready at data/background/kitchen"
    exit 0
fi
if [ -d "data/background/kitchen/images" ]; then
    echo "[skip] kitchen already ready at data/background/kitchen"
    exit 0
fi
if [ ! -d "data/background/360_v2" ]; then
    echo "Downloading Mip-NeRF 360 (~8 GB, this may take a while) ..."
    curl --fail -L -C - -o "data/background/360_v2.zip" "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    echo "Unzipping ..."
    unzip -qo "data/background/360_v2.zip" -d "data/background"
    rm "data/background/360_v2.zip"
fi
if [ -d "data/background/360_v2/kitchen" ]; then
    ln -sfn "data/background/360_v2/kitchen" "data/background/kitchen"
elif [ -d "data/background/kitchen" ]; then
    true
else
    echo "Scene 'kitchen' not found in archive."
    echo "Available: $(find data/background -maxdepth 2 -type d -name images -printf '%h
' | sort)"
    exit 1
fi
echo "kitchen ready at data/background/kitchen"
