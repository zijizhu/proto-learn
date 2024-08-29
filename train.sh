#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-best.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-best

python train_dino.py --config_path experiments_new/vitb14-n_splits_3-l_seg_0.5.yaml
python dino_eval.py --log_dir logs_latest/vitb14-n_splits_3-l_seg_0.5
