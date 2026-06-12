#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXTERNAL_DIR="${1:-"$ROOT_DIR/external"}"

mkdir -p "$EXTERNAL_DIR"

clone_if_missing() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local patch="${4:-}"
  local target="$EXTERNAL_DIR/$name"

  if [[ -d "$target/.git" ]]; then
    echo "[skip] $name already exists at $target"
  else
    echo "[clone] $url -> $target"
    git clone "$url" "$target"
  fi

  if [[ "$(git -C "$target" rev-parse HEAD)" != "$commit" ]]; then
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
      echo "Refusing to change revision of dirty repository: $target" >&2
      exit 2
    fi
    git -C "$target" fetch origin "$commit"
    git -C "$target" checkout --detach "$commit"
  fi
  git -C "$target" submodule update --init --recursive

  if [[ -n "$patch" ]]; then
    local patch_path="$ROOT_DIR/patches/$patch"
    if git -C "$target" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
      echo "[skip] patch already applied: $patch"
    elif git -C "$target" apply --check "$patch_path"; then
      git -C "$target" apply "$patch_path"
      echo "[apply] $patch"
    else
      echo "Patch does not apply cleanly: $patch_path" >&2
      exit 3
    fi
  fi
}

clone_if_missing \
  "2d-gaussian-splatting" \
  "https://github.com/hbb1/2d-gaussian-splatting.git" \
  "335ad612f2e783a4e57b9cbc4d1e167bd599fc98" \
  "2d-gaussian-splatting.patch"
clone_if_missing \
  "threestudio" \
  "https://github.com/threestudio-project/threestudio.git" \
  "28d9d80d9d00f308244adfcf3be8b17ca0cb6465"
clone_if_missing \
  "Magic123" \
  "https://github.com/guochengqian/Magic123.git" \
  "c2eb289f0b9e03e5cf39cf1417f05ca33e9eb0a5" \
  "Magic123.patch"
clone_if_missing \
  "lerobot" \
  "https://github.com/huggingface/lerobot.git" \
  "b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf" \
  "lerobot.patch"
clone_if_missing \
  "calvin" \
  "https://github.com/mees/calvin.git" \
  "fa03f01f19c65920e18cf37398a9ce859274af76"
clone_if_missing \
  "tiny-cuda-nn" \
  "https://github.com/NVlabs/tiny-cuda-nn.git" \
  "749dd70c5afc5a9dadb85e5652ed65d55e0ba187"
