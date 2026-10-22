#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dataset="${1:-}"
case "$dataset" in
  Gdataset) data_name=Gdataset; k=4; heads=3; attention_dropout=0.2; contrastive_weight=0.01 ;;
  Cdataset) data_name=Cdataset; k=4; heads=3; attention_dropout=0.4; contrastive_weight=0.01 ;;
  LRSSL) data_name=lrssl; k=16; heads=5; attention_dropout=0.1; contrastive_weight=0.01 ;;
  Ldataset) data_name=Ldataset; k=16; heads=3; attention_dropout=0.3; contrastive_weight=0.1 ;;
  *) echo 'Usage: bash scripts/train_one.sh {Gdataset|Cdataset|LRSSL|Ldataset}' >&2; exit 2 ;;
esac

case "$dataset" in
  Gdataset|Cdataset) input="raw_data/drug_data/${dataset}/${dataset}.mat" ;;
  LRSSL) input=raw_data/drug_data/lrssl/drug_dis.txt ;;
  Ldataset) input=raw_data/drug_data/Ldataset/lagcn/drug_dis.csv ;;
esac
if [[ ! -f "$input" ]]; then
  echo "Benchmark input not found: $input. Run prepare_benchmarks.py first." >&2
  exit 1
fi

mkdir -p logs
python -u drug_train.py \
  --device "${MACH_DEVICE:-0}" \
  --data_name "$data_name" \
  --training_protocol fixed_epoch \
  --train_max_iter 6000 \
  --train_lr 0.001 \
  --dropout 0.4 \
  --layers 2 \
  --gcn_agg_units 840 \
  --gcn_out_units 75 \
  --nhid1 500 \
  --nhid2 75 \
  --model_activation tanh \
  --num_neighbor "$k" \
  --attention_mode modality_tokens \
  --attention_heads "$heads" \
  --attention_dropout "$attention_dropout" \
  --fusion_mode parallel \
  --use_fusion_gate \
  --use_contrastive \
  --contrastive_weight "$contrastive_weight" \
  --temperature 0.2 \
  --multiscale 2,4 \
  --use_msf \
  --use_feature_branch \
  --use_topology_branch \
  --focal_gamma 2.0 \
  --margin 0.3 \
  --train_grad_clip 1.0 \
  --inner_split_seed 125 \
  --seed 125 \
  2>&1 | tee "logs/${dataset}.log"
