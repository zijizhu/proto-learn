#!/bin/bash

declare -a filenames=(
"dinov2-few_shot-5_shots"
"dinov2-few_shot-10_shots-gamma_0.5"
"dinov2-few_shot-10_shots-gamma_0.7"
"dinov2-few_shot-10_shots-gamma_0.9"
"dinov2-few_shot-10_shots"
"dinov2-few_shot-15_shots"
)

for f in "${filenames[@]}"
do
  python train_dino.py --base_log_dir logs-09-10 --config_path "configs/few-shot/$f.yaml"
  python dino_eval.py --log_dir "logs-09-10/$f"
done
