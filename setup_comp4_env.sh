#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/mdjpt-comp4-py310}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTHON_BIN="${PYTHON_BIN:-/vePFS-0x0d/home/cx/.miniforge3/envs/mdjpt-mdd/bin/python}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to a Python 3.10 executable before rerunning." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -i "$PIP_INDEX_URL" torch==2.3.1 torchvision==0.18.1
"$VENV_DIR/bin/pip" install -i "$PIP_INDEX_URL" \
  hydra-core \
  transformers==4.45.2 \
  pytorch_lightning==2.4.0 \
  tqdm==4.66.5 \
  scipy \
  h5py \
  mne==1.8.0 \
  wandb==0.18.3 \
  hdf5storage==0.1.19 \
  tensorboard==2.18.0 \
  scikit-learn \
  matplotlib
"$VENV_DIR/bin/pip" install -i "$PIP_INDEX_URL" --no-deps timm==1.0.9

cat <<EOF
Environment ready.

Activate with:
source "$VENV_DIR/bin/activate"

Quick sanity check:
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
EOF
