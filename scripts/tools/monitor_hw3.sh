#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "== Processes =="
pgrep -af "huiwon/calvin_task_ABC_D|snapshot_download|train.py -s .*data/(background/kitchen|task1/object_a_undistorted)|threestudio|launch.py|Magic123|main.py" || true

echo
echo "== Disk =="
df -h .
du -sh data/calvin_hf_fast_40g data/background/kitchen data/task1/object_a outputs/task1/final outputs/task1/experiments outputs/task2 2>/dev/null || true

echo
echo "== CALVIN HF progress =="
if [ -d data/calvin_hf/huiwon_calvin_task_ABC_D ]; then
    printf "files: "
    find data/calvin_hf/huiwon_calvin_task_ABC_D -type f 2>/dev/null | wc -l
    echo "expected files: 53635"
fi

echo
echo "== Background 2DGS output =="
find outputs/task1/final/environment/kitchen_2dgs -maxdepth 4 -type f 2>/dev/null | sort | tail -n 20 || true

echo
echo "== Object A 2DGS output =="
find outputs/task1/final/objects/object_a/model -maxdepth 4 -type f 2>/dev/null | sort | tail -n 20 || true

echo
echo "== Background log tail =="
tail -n 40 logs/background_2dgs_7k.log 2>/dev/null || true

echo
echo "== Object A log tail =="
tail -n 40 logs/object_a_2dgs_white30k.log 2>/dev/null || true

echo
echo "== Object B log tail =="
tail -n 40 logs/object_b_kiwi_v1.log 2>/dev/null || true

echo
echo "== GPU =="
nvidia-smi || true
