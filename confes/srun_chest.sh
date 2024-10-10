#!/bin/bash
srun --gres=gpu:1 \
     -c 4 \
     --mem 20GB \
     --job-name conf \
     --time=07:40:00 \
     python main.py \
     --dataset chest \
     --model resnet34 \
     --lr 0.0001 \
     --weight-decay 0.1 \
     --batch-size 32 \
     --epochs 20 \
     --optimizer adam \
     --pretrain 1 \
     --num-samples 5163 \
     --noise-rate 0.6 \
     --noise-type instance \
     > conf_chest_clean_unbal_thresh_gmm2.log 2>&1

# 165466 46928