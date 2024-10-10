#!/bin/bash
srun --gres=gpu:1 \
     -c 4 \
     --mem 20GB \
     --job-name conf \
     --time=07:40:00 \
     python main.py \
     --dataset tissue \
     --model resnet18 \
     --lr 0.05 \
     --weight-decay 0 \
     --batch-size 128 \
     --epochs 10 \
     --gamma 0.01 \
     --num-samples 165466 \
     --noise-rate 60 \
     --noise-type symmetric \
     --clustering none \
     --device cuda \
     > conf_tissue_symmetric06_large_3.log 2>&1

# 165466 46928