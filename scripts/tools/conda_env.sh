#!/usr/bin/env bash

resolve_conda_env() {
  local env_name="$1"
  local conda_cmd="${CONDA_EXE:-}"

  if [[ -z "$conda_cmd" ]]; then
    conda_cmd="$(command -v conda || true)"
  fi
  if [[ -z "$conda_cmd" ]]; then
    echo "Conda was not found. Install Miniconda/Anaconda or set CONDA_EXE." >&2
    return 127
  fi

  CONDA_ENV_PREFIX="$(
    "$conda_cmd" run -n "$env_name" python -c 'import sys; print(sys.prefix)'
  )"
  CONDA_ENV_PYTHON="$CONDA_ENV_PREFIX/bin/python"
  CONDA_ENV_BIN="$CONDA_ENV_PREFIX/bin"
  export CONDA_ENV_PREFIX CONDA_ENV_PYTHON CONDA_ENV_BIN
}
