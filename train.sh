#!/bin/bash

declare -a filenames=(
"dinov2_vitb14-fg_resnet18_0.4-n_proto_5-all_losses-l_orth_coef_0.001"
"dinov2_vitb14-fg_resnet18_0.4-n_proto_5-all_losses"
"dinov2_vitb14-fg_resnet18_0.4-n_proto_6-all_losses-l_orth_coef_0.001"
"dinov2_vitb14-fg_resnet18_0.4-n_proto_6-all_losses-l_orth_coef_0.003"
"dinov2_vitb14-fg_resnet18_0.4-n_proto_6-all_losses"
)

for f in "${filenames[@]}"
do
  python train_dino.py --config_path "experiments_new/$f.yaml"
  python dino_eval.py --log_dir "logs/$f"
done
