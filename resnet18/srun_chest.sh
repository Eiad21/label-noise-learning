#!/bin/bash
srun --gres=gpu:1 \
     -c 4 \
     --mem 20GB \
     --job-name noCorr_G \
     python main.py \
     --dataset chest \
     --model resnet34 \
     --lr 0.0001 \
     --weight-decay 0.1 \
     --gamma 0.1 \
     --batch-size 32 \
     --optimizer adam \
     --epochs 20 \
     --pretrain 0 \
     --num-samples 5163 \
     --noise-rate 0.4 \
     --noise-type symmetric \
     --num-classes 3 \
     --clustering gmm \
     > chest_resnet_clean.log 2>&1

# 165466