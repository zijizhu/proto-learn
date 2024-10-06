#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-base"
"dinov2_vitb14-fine_tune-n_splits_1"
"dinov2_vitb14-fine_tune"
)

for f in "${filenames[@]}"
do
  python train_dino.py --base_log_dir logs-06-10 --config_path "configs/$f.yaml"
  python dino_eval.py --log_dir "logs-06-10/$f"
done
