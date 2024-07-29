#!/bin/bash

python train.py --config_path configs/resnet34.yaml
python train.py --config_path configs/resnet34.yaml --options optim.seed=1
