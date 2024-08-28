#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-new.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-new

python train_dino.py --config_path experiments_new/vits14-n_splits_3-new.yaml
python dino_eval.py --log_dir logs_latest/vits14-n_splits_3-new

