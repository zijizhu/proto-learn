#!/bin/bash

python train.py --config_path configs/densenet121_base.yaml
python train.py --config_path configs/resnet50_base.yaml

settings=(
    "optim.final=null"
    "model.proj_layers=512,relu,128,sigmoid"
    "model.proj_layers=128,relu"
    "model.proj_layers=128,sigmoid"
)

for s in "${settings[@]}"
do
    python train.py --config_path configs/resnet50_base.yaml --options "$s"
done

for s in "${settings[@]}"
do
    python train.py --config_path configs/densenet121_base.yaml --options "$s"
done

