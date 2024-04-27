#!/bin/bash
# Define methods
methods=(EFBSW FBSW lowerboundFBSW BSW None)

# Loop through each method
for method in "${methods[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python3 train.py \
        --lr 0.0005 \
        --epochs 500 \
        --batch-size 1000 \
        --batch-size-test 500 \
        --log-epoch-interval 20 \
        --dataset mnist \
        --datadir data \
        --outdir result \
        --optimizer adam \
        --weight_swd 8 \
        --weight_fsw 0.3 \
        --method "$method" \
        --store-best False \
        --store-end True
done
