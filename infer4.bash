methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 10.0)
checkpoint_periods=(300 300)

for ckp in "${checkpoint_periods[@]}"; do

    for method in "${methods[@]}"; do
        python3 evaluator_mnist.py \
            --no-cuda \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result3 \
            --distribution circle \
            --batch-size 128 \
            --batch-size-test 128 \
            --lr 0.001 \
            --seed 42 \
            --weight_fsw 1.0 \
            --checkpoint-period "$checkpoint_periods" \
            --method "$method"
    done

    for lmbd in "${obsw_weights[@]}"; do
        python3 evaluator_mnist.py \
            --no-cuda \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result3 \
            --distribution circle \
            --batch-size 128 \
            --batch-size-test 128 \
            --lr 0.001 \
            --seed 42 \
            --weight_fsw 1.0 \
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --checkpoint-period "$checkpoint_periods"
    done

    python3 evaluator_mnist.py \
        --no-cuda \
        --dataset mnist \
        --num-classes 10 \
        --datadir data \
        --outdir result3 \
        --distribution circle \
        --batch-size 128 \
        --batch-size-test 128 \
        --lr 0.001 \
        --seed 42 \
        --weight_fsw 0 \
        --checkpoint-period "$ckp" \
        --method None
done