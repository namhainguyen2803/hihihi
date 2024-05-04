weight_fsw_values=(2.0)
gpu_id=0

methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 10.0)
checkpoint_periods=(300 300 300 300 300 300 300)

for ckp in "${checkpoint_periods[@]}"; do
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
                --checkpoint-period "$checkpoint_periods" \
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
                --checkpoint-period "$checkpoint_periods"
        done
    done
done