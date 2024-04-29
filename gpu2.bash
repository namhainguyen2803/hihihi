#!/bin/bash
# Define methods
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
weight_fsw_values=(4.0)
# Loop through each method


for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES=3 python3 train.py \
            --lr 0.001 \
            --epochs 30000 \
            --batch-size 1000 \
            --batch-size-test 1000 \
            --log-epoch-interval 1000 \
            --dataset mnist \
            --datadir data \
            --outdir trash \
            --optimizer rmsprop \
            --alpha 0.9 \
            --weight_swd 8 \
            --weight_fsw "$weight_fsw" \
            --method "$method" \
            --seed 42 \
            --store-best False \
            --store-end True
    done
done