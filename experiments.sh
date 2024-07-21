#!/bin/bash

python train.py --config_path configs/vgg19_base.yaml
python train.py --config_path configs/densenet121_base.yaml
python train.py --config_path configs/resnet50_base.yaml

python eval.py --log_dir logs/vgg19_base
python eval.py --log_dir logs/densenet121_base
python eval.py --log_dir logs/resnet50_base

python train_with_concepts.py --config_path configs/vgg19_base.yaml
python train_with_concepts.py --config_path configs/densenet121_base.yaml
python train_with_concepts.py --config_path configs/resnet50_base.yaml