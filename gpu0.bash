weight_fsw_values=(0.1 0.5 1.0 2.0 4.0)
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 10.0)
gpu_id=3

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator_mnist.py \
            --dataset cifar10 \
            --num-classes 10 \
            --datadir data \
            --outdir result6 \
            --stat-dir stats \
            --images-path images3 \
            --distribution uniform \
            --batch-size 128 \
            --batch-size-test 128 \
            --lr 0.0005 \
            --dims 2048 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --checkpoint-period 200 \
            --times 7\
            --method "$method"
    done
done

for weight_fsw in "${weight_fsw_values[@]}"; do
    for lmbd in "${obsw_weights[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator_mnist.py \
            --dataset cifar10 \
            --num-classes 10 \
            --datadir data \
            --outdir result6 \
            --stat-dir stats \
            --images-path images3 \
            --distribution uniform \
            --batch-size 128 \
            --batch-size-test 128 \
            --lr 0.0005 \
            --dims 2048 \
            --seed 42 \
            --weight_fsw "$weight_fsw" \
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --checkpoint-period 200 \
            --times 7
    done
done