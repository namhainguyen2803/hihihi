%%bash

methods=(EFBSW)

for method in "${methods[@]}"
do
    python3 mnist_trainer.py --lr 0.001 --epochs 100 --dataset mnist --datadir data --outdir result --optimizer adam --weight_swd 8 --weight_fsw 0.5 --method "$method"
done
