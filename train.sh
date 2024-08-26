#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vits14-n_splits_3.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-scale_6.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-scale_6

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-learn_scale_1e-4.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-learn_scale_1e-4
