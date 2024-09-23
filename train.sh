#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-adam"
"dinov2_vitb14-base"
"dinov2_vitb14-contrast"
"dinov2_vitb14-fine_tune-always_optimize"
"dinov2_vitb14-fine_tune-clst_sep"
"dinov2_vitb14-fine_tune-n_prototypes_8"
"dinov2_vitb14-fine_tune-n_prototypes_10"
"dinov2_vitb14-fine_tune-papr"
"dinov2_vitb14-fine_tune-sa_lr_1e-4"
"dinov2_vitb14-fine_tune-temp_0.5"
"dinov2_vitb14-fine_tune-temp_1.0"
"dinov2_vitb14-fine_tune"
)

for f in "${filenames[@]}"
do
  python train_dino.py --config_path "configs/$f.yaml"
  python dino_eval.py --log_dir "logs-23-09/$f"
done
