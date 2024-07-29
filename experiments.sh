#!/bin/bash

python train.py --config_path configs/resnet34_base.yaml
python train.py --config_path configs/resnet34_base.yaml --options optim.seed=1
