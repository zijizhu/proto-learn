#!/bin/bash

set -x

python train_dino.py --config_path experiments/dinov2_non_proj_max_max.yaml
python train_dino.py --config_path experiments/dinov2_non_proj_avgpool_avg.yaml
