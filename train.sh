#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-adapt-adam"
"dinov2_vitb14-adapt-bottleneck"
"dinov2_vitb14-adapt-n_prototype_5-bottleneck"
"dinov2_vitb14-adapt-n_prototype_5"
"dinov2_vitb14-adapt-n_prototype_8"
"dinov2_vitb14-adapt"
)

for f in "${filenames[@]}"
do
  python train_dino_adapt.py --config_path "experiments/$f.yaml"
  # python dino_eval.py --log_dir "logs/$f"
done
