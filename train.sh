#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-scale_init_1.0-sa_init_2.0.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-scale_init_1.0-sa_init_2.0

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-scale_init_1.0-sa_init_2.5.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-scale_init_1.0-sa_init_2.5

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-scale_init_1.0-sa_init_2.0-fc_lr_1e-4.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-scale_init_1.0-sa_init_2.0-fc_lr_1e-4
