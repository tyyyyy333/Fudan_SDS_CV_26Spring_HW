#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-hw3t2}"
source "$ROOT/scripts/tools/conda_env.sh"
resolve_conda_env "$CONDA_ENV"
PYTHON="$CONDA_ENV_PYTHON"
DATA_ROOT="${DATA_ROOT:-data/calvin_hf_fast_40g/huiwon_calvin_task_ABC_D}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/task2/runs/${RUN_ID}}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
LR="${LR:-0.0001}"
DATA_FILE_MB="${DATA_FILE_MB:-100}"
VIDEO_FILE_MB="${VIDEO_FILE_MB:-50}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
FORCE="${FORCE:-0}"
ALLOW_HF_SHARD_SURROGATE="${ALLOW_HF_SHARD_SURROGATE:-0}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

if [[ "$DATA_ROOT" == *"calvin_hf"* ]]; then
  log "using HF CALVIN mapping: 0_4=A, 1_4=B, 2_4=C, 3_4=D; abc_0_1_2=ABC"
fi

dataset_version() {
  local root="$1"
  "$PYTHON" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = root / "meta" / "info.json"
if not info.exists():
    print("missing")
else:
    data = json.loads(info.read_text())
    print(data.get("codebase_version") or data.get("version") or "unknown")
PY
}

dataset_summary() {
  local root="$1"
  "$PYTHON" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info_path = root / "meta" / "info.json"
out = {
    "root": str(root),
    "exists": root.exists(),
}
if info_path.exists():
    info = json.loads(info_path.read_text())
    for key in [
        "repo_id",
        "codebase_version",
        "total_episodes",
        "total_frames",
        "fps",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
    ]:
        if key in info:
            out[key] = info[key]
out["size_gb"] = round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1024**3, 3) if root.exists() else 0
print(json.dumps(out, indent=2))
PY
}

convert_if_needed() {
  local shard="$1"
  local root="${DATA_ROOT}/calvin_task_ABC_D_lerobot_${shard}_4"
  local version

  if [[ ! -d "$root" ]]; then
    log "ERROR missing dataset shard: $root"
    exit 1
  fi

  version="$(dataset_version "$root")"
  if [[ "$version" == "v3.0" ]]; then
    log "shard ${shard} already v3.0: $root"
    return
  fi

  log "converting shard ${shard} from ${version} to v3.0 with ${VIDEO_FILE_MB}MB video chunks"
  conda run -n "$CONDA_ENV" python external/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id "local/calvin_task_ABC_D_lerobot_${shard}_4" \
    --root "$root" \
    --data-file-size-in-mb "$DATA_FILE_MB" \
    --video-file-size-in-mb "$VIDEO_FILE_MB" \
    --push-to-hub false
}

aggregate_abc() {
  local aggr_root="${DATA_ROOT}/calvin_task_ABC_D_lerobot_abc_0_1_2"
  if [[ "$FORCE" == "1" && -d "$aggr_root" ]]; then
    log "FORCE=1 removing previous aggregate: $aggr_root"
    rm -rf "$aggr_root"
  fi
  if [[ "$(dataset_version "$aggr_root")" == "v3.0" ]]; then
    log "ABC aggregate already prepared: $aggr_root"
    return
  fi

  if [[ -d "$aggr_root" ]]; then
    log "removing incomplete aggregate: $aggr_root"
    rm -rf "$aggr_root"
  fi

  log "aggregating shards 0,1,2 into $aggr_root"
  conda run -n "$CONDA_ENV" python -c '
import sys
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets

data_root = Path(sys.argv[1])
aggr_root = Path(sys.argv[2])
data_mb = int(sys.argv[3])
video_mb = int(sys.argv[4])
roots = [data_root / f"calvin_task_ABC_D_lerobot_{i}_4" for i in (0, 1, 2)]
repo_ids = [f"local/calvin_task_ABC_D_lerobot_{i}_4" for i in (0, 1, 2)]
aggregate_datasets(
    repo_ids=repo_ids,
    aggr_repo_id="local/calvin_task_ABC_D_lerobot_abc_0_1_2",
    roots=roots,
    aggr_root=aggr_root,
    data_files_size_in_mb=data_mb,
    video_files_size_in_mb=video_mb,
)
' "$DATA_ROOT" "$aggr_root" "$DATA_FILE_MB" "$VIDEO_FILE_MB"
}

train_act() {
  local name="$1"
  local repo_id="$2"
  local dataset_root="$3"
  local output_dir="$4"
  local train_log="$RUN_ROOT/logs/${name}.log"

  if [[ "$FORCE" == "1" && -d "$output_dir" ]]; then
    log "FORCE=1 removing previous training output: $output_dir"
    rm -rf "$output_dir"
  fi
  if [[ -f "$output_dir/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]]; then
    log "training checkpoint exists, skipping ${name}: $output_dir"
    return
  fi
  if [[ -d "$output_dir" ]]; then
    log "removing incomplete training output before retry: $output_dir"
    rm -rf "$output_dir"
  fi

  log "starting ${name} ACT training: steps=${STEPS} batch=${BATCH_SIZE} chunk=${CHUNK_SIZE}"
  PYTHONUNBUFFERED=1 stdbuf -oL -eL conda run -n "$CONDA_ENV" lerobot-train \
    --policy.type act \
    --policy.device cuda \
    --dataset.repo_id "$repo_id" \
    --dataset.root "$dataset_root" \
    --dataset.video_backend pyav \
    --tolerance_s 0.01 \
    --output_dir "$output_dir" \
    --job_name "hw3_${name}" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --policy.optimizer_lr "$LR" \
    --policy.chunk_size "$CHUNK_SIZE" \
    --policy.n_action_steps "$CHUNK_SIZE" \
    --policy.push_to_hub false \
    --policy.repo_id "hw3_${name}_act" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --save_freq "$STEPS" \
    --log_freq 20 \
    --wandb.enable false 2>&1 | tee "$train_log"
}

extract_training_record() {
  local name="$1"
  local output_dir="$2"
  local train_log="$3"
  local result_json="$RUN_ROOT/results/${name}.json"
  "$PYTHON" - "$name" "$output_dir" "$train_log" "$STEPS" "$result_json" <<'PY'
import json
import re
import sys
from pathlib import Path

name, output_dir, train_log, steps, result_json = sys.argv[1:6]
text = Path(train_log).read_text(errors="ignore") if Path(train_log).exists() else ""
losses = re.findall(r"loss:([0-9.eE+-]+)", text)
lrs = re.findall(r"lr:([0-9.eE+-]+)", text)
out = {
    "name": name,
    "output_dir": output_dir,
    "log": train_log,
    "target_steps": int(steps),
    "completed": ("End of training" in text or f"{steps}/{steps}" in text or Path(output_dir, "checkpoints", f"{int(steps):06d}", "pretrained_model", "model.safetensors").exists()),
    "final_loss": float(losses[-1]) if losses else None,
    "final_lr": float(lrs[-1]) if lrs else None,
    "checkpoint": str(Path(output_dir) / "checkpoints" / f"{int(steps):06d}" / "pretrained_model" / "model.safetensors"),
}
Path(result_json).write_text(json.dumps(out, indent=2) + "\n")
PY
}

write_final_record() {
  local single_output="$1"
  local abc_output="$2"
  local single_log="$3"
  local abc_log="$4"
  "$PYTHON" - "$RUN_ROOT" "$DATA_ROOT" "$single_output" "$abc_output" "$single_log" "$abc_log" "$STEPS" "$BATCH_SIZE" "$CHUNK_SIZE" "$NUM_WORKERS" "$LR" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root, data_root, single_output, abc_output, single_log, abc_log, steps, batch, chunk, workers, lr = sys.argv[1:12]
run_root = Path(run_root)
records = {}
for name in ["single_b", "abc_to_d"]:
    path = run_root / "results" / f"{name}.json"
    records[name] = json.loads(path.read_text()) if path.exists() else {"name": name, "completed": False}

def disk_line():
    try:
        out = subprocess.check_output(["df", "-h", "."], text=True).strip().splitlines()
        return out[-1] if out else ""
    except Exception:
        return ""

summary = {
    "run_root": str(run_root),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "python": platform.python_version(),
    "data_root": data_root,
    "note": "HF huiwon/calvin_task_ABC_D is used with directory mapping 0_4=A, 1_4=B, 2_4=C, 3_4=D. The ABC aggregate is built from A/B/C and D is recorded as the held-out eval environment. No simulator rollout metric is fabricated here.",
    "hyperparameters": {
        "steps": int(steps),
        "batch_size": int(batch),
        "chunk_size": int(chunk),
        "num_workers": int(workers),
        "lr": float(lr),
        "video_backend": "pyav",
        "tolerance_s": 0.01,
    },
    "experiments": records,
    "evaluation": {
        "status": "not_run",
        "reason": "CALVIN D simulator/evaluator is not configured for this local HF shard surrogate in this script.",
        "eval_d_dataset_root": f"{data_root}/calvin_task_ABC_D_lerobot_3_4",
    },
    "disk": disk_line(),
}
(run_root / "results" / "task2_record.json").write_text(json.dumps(summary, indent=2) + "\n")

md = [
    "# Task 2 Run Record",
    "",
    f"- Run root: `{run_root}`",
    f"- Data root: `{data_root}`",
    f"- Steps: `{steps}`",
    f"- Batch size: `{batch}`",
    f"- Chunk size: `{chunk}`",
    f"- Num workers: `{workers}`",
    f"- LR: `{lr}`",
    "",
    "## Training Results",
    "",
    "| Experiment | Completed | Final loss | Checkpoint | Log |",
    "| --- | --- | ---: | --- | --- |",
]
for name in ["single_b", "abc_to_d"]:
    rec = records.get(name, {})
    md.append(
        f"| {name} | {rec.get('completed')} | {rec.get('final_loss')} | "
        f"`{rec.get('checkpoint')}` | `{rec.get('log')}` |"
    )
md += [
    "",
    "## Evaluation Record",
    "",
    "Simulator rollout evaluation was not run by this script. The script records the checkpoint paths and the shard-3 eval data location only.",
]
(run_root / "results" / "task2_record.md").write_text("\n".join(md) + "\n")
PY
}

log "task2 all-in-one run root: $RUN_ROOT"
log "free space before run:"
df -h .
log "gpu before run:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || log "nvidia-smi query failed; continuing"

for shard in 0 1 2 3; do
  convert_if_needed "$shard"
  dataset_summary "${DATA_ROOT}/calvin_task_ABC_D_lerobot_${shard}_4" > "$RUN_ROOT/results/shard_${shard}.json"
done

single_dataset="${DATA_ROOT}/calvin_task_ABC_D_lerobot_1_4"
single_output="$RUN_ROOT/single_b_train"
existing_single="outputs/task2/single_b_train_1000_b8_smallvid"
if [[ "$REUSE_EXISTING" == "1" && "$STEPS" == "1000" && -f "$existing_single/checkpoints/001000/pretrained_model/model.safetensors" ]]; then
  log "reusing existing single_b checkpoint: $existing_single"
  single_output="$existing_single"
  cp -f logs/task2_single_b_train_1000_b8_smallvid.log "$RUN_ROOT/logs/single_b.log" 2>/dev/null || true
else
  train_act "single_b" "local/calvin_task_ABC_D_lerobot_1_4" "$single_dataset" "$single_output"
fi
extract_training_record "single_b" "$single_output" "$RUN_ROOT/logs/single_b.log"

abc_dataset="${DATA_ROOT}/calvin_task_ABC_D_lerobot_abc_0_1_2"
aggregate_abc
dataset_summary "$abc_dataset" > "$RUN_ROOT/results/abc_dataset.json"
abc_output="$RUN_ROOT/abc_to_d_train"
train_act "abc_to_d" "local/calvin_task_ABC_D_lerobot_abc_0_1_2" "$abc_dataset" "$abc_output"
extract_training_record "abc_to_d" "$abc_output" "$RUN_ROOT/logs/abc_to_d.log"

write_final_record "$single_output" "$abc_output" "$RUN_ROOT/logs/single_b.log" "$RUN_ROOT/logs/abc_to_d.log"

log "task2 script finished"
log "record: $RUN_ROOT/results/task2_record.md"
