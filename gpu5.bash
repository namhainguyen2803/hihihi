weight_fsw_values=(0.5)
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 10.0)
gpu_id=6

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator_mnist.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result3 \
            --distribution circle \
            --batch-size 128 \
            --batch-size-test 128 \
            --lr 0.001 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --checkpoint-period 300 \
            --method "$method"
    done
done

for weight_fsw in "${weight_fsw_values[@]}"; do
    for lmbd in "${obsw_weights[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator_mnist.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result3 \
            --distribution circle \
            --batch-size 128 \
            --batch-size-test 128 \
            --lr 0.001 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --checkpoint-period 300
    done
done

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
            --lr 0.001 \
            --epochs 3000 \
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