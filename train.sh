#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-fine_tune-alternate"
"dinov2_vitb14-fine_tune-always_optimize"
"dinov2_vitb14-fine_tune-bg_class_weight_0.01"
"dinov2_vitb14-fine_tune-bg_class_weight_0.1"
"dinov2_vitb14-fine_tune-bg_class_weight_0.5"
"dinov2_vitb14-fine_tune-bg_class_weight_1.0"
"dinov2_vitb14-fine_tune-temperature_0.1"
"dinov2_vitb14-fine_tune-temperature_0.5"
"dinov2_vitb14-fine_tune-temperature_1.0"
)

for f in "${filenames[@]}"
do
  python train_dino.py --resume_ckpt checkpoints/dino_v2_proto.pth --config_path "experiments/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done
