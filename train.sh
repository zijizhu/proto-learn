#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vits14-n_splits_1-new.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_1-new

python train_dino.py --config_path experiments_new/vits14-n_splits_2-new.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_2-new

python train_dino.py --config_path experiments_new/vits14-n_splits_3-new-lr_1e-4.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3-new-lr_1e-4

python train_dino.py --config_path experiments_new/vits14-n_splits_3-new-scale_2.0.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3-new-scale_2.0

python train_dino.py --config_path experiments_new/vits14-n_splits_3-new-scale_4.0.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3-new-scale_4.0
