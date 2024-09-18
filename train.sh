#!/bin/bash

declare -a filenames=(
"dinov2_vits14-full_fine_tune"
"dinov2_vitb14-fine_tune-temp_0.5"
"dinov2_vitb14-fine_tune-temp_1.0"
"dinov2_vits14-full_fine_tune-l_patch"
)

for f in "${filenames[@]}"
do
  python train_dino.py --config_path "experiments/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done
