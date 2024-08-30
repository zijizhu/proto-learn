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

python train_dino.py --config_path experiments/clip_vitb16-base.yaml
python dino_eval.py --log_dir logs_latest/clip_vitb16-base
