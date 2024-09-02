#!/bin/bash

set -x

python train_dino.py --config_path experiments/dinov2_vitb14-base.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-base

python train_dino.py --config_path experiments/dinov2_vitb14-fine_tuned.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-fine_tuned

python train_dino.py --config_path experiments/dinov2_vits14-base.yaml
python dino_eval.py --log_dir logs/dinov2_vits14-base

python train_dino.py --config_path experiments/dinov2_vits14-fine_tuned.yaml
python dino_eval.py --log_dir logs/dinov2_vits14-fine_tuned

python train_dino.py --config_path e./experiments_new/dinov2_vitb14-lr_0.01.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-lr_0.01

python train_dino.py --config_path e./experiments_new/dinov2_vitb14-lr_0.005.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-lr_0.005

python train_dino.py --config_path e./experiments_new/dinov2_vitb14-lr_0.008.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-lr_0.008

python train_dino.py --config_path e./experiments_new/dinov2_vitb14-scale_lr_0.001.yaml
python dino_eval.py --log_dir logs/dinov2_vitb14-scale_lr_0.001

python train_dino.py --config_path e./experiments_new/dinov2_vits14-scale_lr_0.0001.yaml
python dino_eval.py --log_dir logs/dinov2_vits14-scale_lr_0.0001

python train_dino.py --config_path experiments/clip_vitb16-base.yaml
python dino_eval.py --log_dir logs_latest/clip_vitb16-base
