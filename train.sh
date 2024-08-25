#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vitb14-n_splits_2.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_2

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-learn_scale.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-learn_scale

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-scale_5.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-scale_5

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-seed_1.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-seed_1

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-seed_2.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-seed_2

python train_dino.py --config_path experiments_new/vitb14-n_splits_3.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3

python train_dino.py --config_path experiments_new/vits14-n_splits_3.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3
