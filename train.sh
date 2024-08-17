#!/bin/bash

set -x

python train_dino.py --config_path experiments_new/vitb14-proj1.yaml
python train_dino.py --config_path experiments_new/vitb14-proj2.yaml
python train_dino.py --config_path experiments_new/vitb14-proj3.yaml
python train_dino.py --config_path experiments_new/vitb14-proj4.yaml
python train_dino.py --config_path experiments_new/vitb14-proj5.yaml
