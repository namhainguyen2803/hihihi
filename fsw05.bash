weight_fsw_values=(0.1 0.5 1.0 2.0 4.0)
obsw_weights=(0.1)

for weight_fsw in "${weight_fsw_values[@]}"; do
    for lmbd in "${obsw_weights[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python3 train_fid.py \
            --dataset cifar10 \
            --num-classes 10 \
            --datadir data \
            --outdir result6 \
            --distribution uniform \
            --epochs 300 \
            --batch-size 1000 \
            --batch-size-test 1000 \
            --optimizer adam \
            --beta1 0.5 \
            --beta2 0.999 \
            --lr 0.0005 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --log-epoch-interval 100 \
            --store-end True \
            --store-best False
    done
done

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python3 train.py \
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