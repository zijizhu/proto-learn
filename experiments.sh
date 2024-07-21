#!/bin/bash

python train.py --config_path configs/densenet121_base.yaml
python train.py --config_path configs/resnet50_base.yaml

#settings=("")
#
#for s in "${settings[@]}"
#do
#    python train.py --config_path configs/resnet50_base.yaml --options "$s"
#done
#
#for s in "${settings[@]}"
#do
#    python train.py --config_path configs/densenet121_base.yaml --options "$s"
#done

