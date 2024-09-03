#!/bin/bash

set -x

python dino_eval.py --log_dir logs/dinov2_vitb14-base

python dino_eval.py --log_dir logs/dinov2_vitb14-fine_tuned

python dino_eval.py --log_dir logs/dinov2_vits14-base

python dino_eval.py --log_dir logs/dinov2_vits14-fine_tuned

python train_dino.py --config_path experiments_new/dinov2_vitb14-lr_0.01.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-lr_0.01

python train_dino.py --config_path experiments_new/dinov2_vitb14-lr_0.005.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-lr_0.005

python train_dino.py --config_path experiments_new/dinov2_vitb14-lr_0.008.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-lr_0.008

python train_dino.py --config_path experiments_new/dinov2_vitb14-scale_lr_0.001.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-scale_lr_0.001

python train_dino.py --config_path experiments_new/dinov2_vits14-scale_lr_0.0001.yaml
python dino_eval.py --log_dir logs/dinov2_vits14-scale_lr_0.0001

python train_dino.py --config_path experiments/clip_vitb16-base.yaml
python dino_eval.py --log_dir logs/clip_vitb16-base
