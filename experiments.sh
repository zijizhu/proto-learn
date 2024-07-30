#!/bin/bash

python train_with_dino.py --config_path configs/dinov2_fixed.yaml
python train_with_dino.py --config_path experiments/dinov2_fixed_256.yaml
python train_with_dino.py --config_path experiments/dinov2_fixed_spatial.yaml
