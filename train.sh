#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-fine_tune-bg_class_weight_0.01"
"dinov2_vitb14-fine_tune-bg_class_weight_0.5"
"dinov2_vitb14-fine_tune-bg_class_weight_1"
"dinov2_vitb14-fine_tune-temperature_0.2"
"dinov2_vitb14-fine_tune-temperature_0.3"
"dinov2_vitb14-fine_tune-temperature_0.5"
"dinov2_vitb14-fine_tune-temperature_1.0"
)

for f in "${filenames[@]}"
do
  python train_dino.py --resume_ckpt checkpoints/dino_v2_proto.pth --config_path "configs/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done
