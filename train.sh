#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_clst_sep"
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_logits-lr_decay"
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_logits-weight_decay"
"dinov2_vitb14-fine_tune-n_splits_1-l_patch_similarity-lr_decay"
)

for f in "${filenames[@]}"
do
  python train_dino.py --base_log_dir logs-16-10 --config_path "configs/$f.yaml"
  python dino_eval.py --log_dir "logs-16-10/$f"
done
