#!/bin/bash

python train.py --config_path configs/vgg19_base.yaml
python train.py --config_path configs/densenet121_base.yaml

python train_with_concepts.py --config_path configs/vgg19_base.yaml
python train_with_concepts.py --config_path configs/densenet121_base.yaml
