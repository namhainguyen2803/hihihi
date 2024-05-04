gpu_id=0

CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
    --dataset mnist \
    --num-classes 10 \
    --datadir data \
    --outdir result4 \
    --distribution circle \
    --optimizer rmsprop \
    --epochs 300 \
    --lr 0.001 \
    --batch-size 1000 \
    --batch-size-test 1000 \
    --lr 0.001 \
    --seed 42 \
    --weight_swd 8 \
    --weight_fsw 0 \
    --method None \
    --log-epoch-interval 100