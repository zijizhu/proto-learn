#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-base"
"dinov2_vitb14-fine_tune"
)

for f in "${filenames[@]}"
do
  python train_dino.py --config_path "configs/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done

declare -a filenames=(
"dinov2_vitb14-n_prototypes_5-fine_tune"
"dinov2_vitb14-n_prototypes_8-fine_tune"
)

for f in "${filenames[@]}"
do
  python train_dino.py --config_path "experiments_new/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done
