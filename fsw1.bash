weight_fsw_values=(0.1 0.5 1.0 2.0 4.0)
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
gpu_id=6

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator_mnist.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result3 \
            --images-path images3 \
            --distribution circle \
            --batch-size 100 \
            --batch-size-test 100 \
            --lr 0.001 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --checkpoint-period 100 \
            --method "$method"
    done
done

obsw_weights=(0.1 10.0)
for weight_fsw in "${weight_fsw_values[@]}"; do
    for lmbd in "${obsw_weights[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator_mnist.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result3 \
            --images-path images3 \
            --distribution circle \
            --batch-size 100 \
            --batch-size-test 100 \
            --lr 0.001 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --checkpoint-period 100 \
            --method OBSW \
            --lambda-obsw "$lmbd"
    done
done

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
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