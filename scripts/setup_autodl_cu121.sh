#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${MACH_PYTHON:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit('Use Python 3.10–3.12 (set MACH_PYTHON to its executable).')
print('Python:', sys.version.split()[0])
PY
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
python -m pip install -r requirements.txt
python - <<'PY'
import dgl
import numpy
import scipy
import sklearn
import torch
import torchdata

print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('DGL:', dgl.__version__)
print('torchdata:', torchdata.__version__)
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('scikit-learn:', sklearn.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; check the host driver and installed wheels before training.')
print('GPU:', torch.cuda.get_device_name(0))
PY
