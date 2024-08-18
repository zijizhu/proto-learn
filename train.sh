#!/bin/bash

set -x

python train_dino.py --config_path experiments/vitb14-aug-adapter_mlp-full-max_sa.yaml
python train_dino.py --config_path experiments/vits14-aug-block_exp-n_splits_3-full-max_sa.yaml
python train_dino.py --config_path experiments/vitb14-aug-adapter_bottleneck-full-max_sa.yaml
