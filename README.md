# MACH for drug–disease association prediction

This repository contains the MACH implementation, pinned Python dependencies, the four dataset-specific configurations, and commands for a new training run. It currently contains **source code and environment instructions only**. New fold checkpoints, predictions, metrics and run logs will be added after the cloud rerun and checked against the manuscript before release.

Parts of the implementation build on [AdaDR](https://github.com/xinliangSun/AdaDR). Its original MIT copyright and permission notice are retained in [`third_party/AdaDR-LICENSE.txt`](third_party/AdaDR-LICENSE.txt). No separate license for MACH-authored additions is declared here.

## Repository contents

| Path | Purpose |
| --- | --- |
| `model.py`, `layers.py`, `data.py`, `utils.py` | Model, graph/data loading and utilities |
| `drug_train.py`, `training_protocol.py`, `evaluate.py` | Training, fold handling and evaluation |
| `prepare_benchmarks.py` | Validate and stage benchmark inputs |
| `predict_pairs.py` | Score pairs using a trained checkpoint and its fold-specific training pairs |
| `configs/*.json` | Parameter records for Gdataset, Cdataset, LRSSL and Ldataset |
| `scripts/setup_autodl_cu121.sh`, `scripts/train_one.sh` | Linux/CUDA setup and single-dataset training commands |

## Input data

Obtain the four benchmark matrices from the [AdaDR data release](https://github.com/xinliangSun/AdaDR), subject to its terms. Place its `raw_data/drug_data` folder anywhere accessible, then from this repository's root run:

```bash
python prepare_benchmarks.py published \
  --source /path/to/AdaDR/raw_data/drug_data \
  --output raw_data/drug_data
```

The validator checks binary association labels and similarity-matrix dimensions. Gdataset and Cdataset use MAT files, LRSSL uses tab-delimited matrices, and Ldataset uses the `lagcn` CSV matrices. Data files are intentionally excluded from this repository; the published similarity matrices are used as supplied. Their upstream construction has not been independently reconstructed here.

## Environment

The local CPU import/smoke environment used Python 3.12.7, PyTorch 2.3.0, DGL 2.2.1, torchdata 0.7.1 and the versions in [`requirements.txt`](requirements.txt). For an AutoDL Linux host with an NVIDIA GPU and a CUDA 12.1-compatible driver, run:

```bash
bash scripts/setup_autodl_cu121.sh
source .venv/bin/activate
```

The setup script accepts Python 3.10–3.12 (select a non-default executable with `MACH_PYTHON=python3.12`), installs CUDA 12.1 wheels for PyTorch and DGL, installs the remaining pinned dependencies, and verifies that both libraries import and PyTorch sees the GPU. This CUDA environment is a **new run recipe**, not a claim that the historical cloud image has been recovered. If the host's driver or Python differs, use the matching official [PyTorch](https://docs.pytorch.org/get-started/previous-versions/) and [DGL](https://www.dgl.ai/pages/start.html) wheels before training.

## Train one dataset

Run from the repository root after activating the environment and preparing the data:

```bash
bash scripts/train_one.sh Gdataset
bash scripts/train_one.sh Cdataset
bash scripts/train_one.sh LRSSL
bash scripts/train_one.sh Ldataset
```

Run **one command at a time** on a single GPU. Each command applies the parameters in the corresponding `configs/*.json` file, trains for 6,000 epochs per fold, and evaluates the final epoch. The script's current outer loop performs **one 10-fold pair-wise cross-validation run**, not ten repetitions. It writes fold checkpoints, train/test pair tables, predictions, metrics and `final_results.txt` under `neighbor_num1/<data_name>_1time/fixed_epoch/`; the shell script also captures a console log in `logs/`. Keep the entire output tree for review. For a CPU-only diagnostic, set `MACH_DEVICE=-1` before invoking the script, but a full 6,000-epoch run will be slow.

The four JSON files are copied from the selected full-model run records. `LRSSL` is passed to the Python program as `lrssl`, matching its dataset loader. The archived Gdataset record omits explicit feature/topology branch flags; both are enabled by the current program defaults.

## Evaluation scope and later release

The loader creates a shuffled, stratified-by-label pair split with `KFold(n_splits=10, shuffle=True, random_state=1024)` for each label, and combines matching folds. Fold test pairs are withheld from that fold's training topology; all drug/disease nodes and supplied similarity matrices remain available. These runs assess transductive pair prediction. Because the released similarity matrices' construction from labels has not been verified, an evaluation designed to eliminate that possible path needs fold-specific similarity construction. The selected parameters were developed within the cross-validation workflow; this rerun alone is a reproducibility check, not a new independent model-selection estimate.

After the cloud rerun, we plan to release a manifest of code commit, input-data identity, environment versions, exact commands, ten fold-level metrics and prediction files per dataset, and checkpoints or a size-appropriate download location. The current repository does **not** contain those new results and should not be cited as their verification yet. `predict_pairs.py` becomes usable once a checkpoint and its matching `outer_train.csv` are available; use `python predict_pairs.py --help` for the required arguments.
