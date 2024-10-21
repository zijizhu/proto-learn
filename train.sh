#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_logits"
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_similarity"
"dinov2_vitb14-fine_tune-n_splits_1"
"dinov2_vits14-fine_tune-n_splits_1"
)

for f in "${filenames[@]}"
do
  python train.py --base_log_dir logs-22-10 --config_path "configs/concept/$f.yaml"
  python evaluate.py --log_dir "logs-22-10/concept/$f"
done

declare -a filenames=(
"dinov2_vits14-fine_tune-n_splits_1-l_patch_logits"
"dinov2_vits14-fine_tune-n_splits_1-l_patch_similarity"
"dinov2_vits14-fine_tune-n_splits_1"
)

for f in "${filenames[@]}"
do
  python train.py --base_log_dir logs-22-10 --config_path "configs/$f.yaml"
  python evaluate.py --log_dir "logs-22-10/$f"
done
