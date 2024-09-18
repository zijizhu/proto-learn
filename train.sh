#!/bin/bash

declare -a filenames=(
"dinov2_vitb14"
"dinov2_vitb14-fine_tune"
"dinov2_vitb14-fine_tune-n_prototypes_10"
"dinov2_vitb14-fine_tune-always_optimize"
)

for f in "${filenames[@]}"
do
  python train_dino.py --config_path "experiments/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done
