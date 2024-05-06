weight_fsw_values=(1.0)
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 1.0 10.0)

python3 train.py \
  --dataset mnist \
  --num-classes 10 \
  --datadir data \
  --outdir result \
  --distribution circle \
  --epochs 300 \
  --optimizer rmsprop \
  --lr 0.001 \
  --alpha 0.9 \
  --batch-size 1000 \
  --batch-size-test 128 \
  --seed 42 \
  --weight_swd 8.0 \
  --weight_fsw 0.0 \
  --method None \
  --saved-model-interval 100 \

for weight_fsw in "${weight_fsw_values[@]}"; do

    for method in "${methods[@]}"; do
        python3 train.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result \
            --distribution circle \
            --epochs 300 \
            --optimizer rmsprop \
            --lr 0.001 \
            --alpha 0.9 \
            --batch-size 1000 \
            --batch-size-test 128 \
            --seed 42 \
            --weight_swd 8.0 \
            --weight_fsw "$weight_fsw" \
            --method "$method" \
            --saved-model-interval 100
    done

    for lmbd in "${obsw_weights[@]}"; do
        python3 train.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir data \
            --outdir result \
            --distribution circle \
            --epochs 300 \
            --optimizer rmsprop \
            --lr 0.001 \
            --alpha 0.9 \
            --batch-size 1000 \
            --batch-size-test 128 \
            --seed 42 \
            --weight_swd 8.0 \
            --weight_fsw "$weight_fsw" \
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --saved-model-interval 100
    done
done