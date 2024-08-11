#!/bin/bash

set -x

python train_dino.py --config_path experiments/dinov2_non_proj_max_avg.yaml
python train_dino.py --config_path experiments/dinov2_non_proj_avg_avg.yaml

python train_dino.py --config_path experiments/dinov2_aug_non_proj_avg_fc.yaml
python train_dino.py --config_path experiments/dinov2_non_proj_max_fc.yaml
python train_dino.py --config_path experiments/dinov2_non_proj_avg_fc.yaml