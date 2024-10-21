#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_logits"
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_similarity"
)

for f in "${filenames[@]}"
do
  python train.py --base_log_dir logs-21-10 --config_path "configs/$f.yaml"
  python eval.py --log_dir "logs-21-10/$f"
done
