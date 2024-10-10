#!/bin/bash
srun --gres=gpu:1 \
     -c 4 \
     --mem 20GB \
     --job-name noCorr_G \
     --time=07:40:00 \
     python main.py \
     --dataset tissue \
     --model resnet18 \
     --lr 0.05 \
     --weight-decay 0 \
     --batch-size 128 \
     --epochs 10 \
     --gamma 0.01 \
     --noise-rate 40 \
     --noise-type clean \
     --device cuda \
     > cross_tissue_clean_large_6.log 2>&1

# 165466