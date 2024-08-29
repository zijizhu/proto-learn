#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vits14-n_splits_3-new-scale_1.5.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3-new-scale_1.5

python train_dino.py --config_path experiments_new/vits14-n_splits_3-new-scale_1.5_learn.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3-new-scale_1.5_learn
