#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vitb14-cosine.yaml
python train_dino.py --config_path experiments_new/vitb14-lr_005.yaml
python train_dino.py --config_path experiments_new/vitb14-lr_01.yaml
python train_dino.py --config_path experiments_new/vitb14-n_split_2-cosine.yaml
python train_dino.py --config_path experiments_new/vitb14-n_split_2.yaml
python train_dino.py --config_path experiments_new/vitb14-n_split_3-cosine.yaml
python train_dino.py --config_path experiments_new/vitb14-n_split_3.yaml
python train_dino.py --config_path experiments_new/vitb14-aug.yaml
