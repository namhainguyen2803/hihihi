%%bash

methods=(EFBSW FBSW lowerboundFBSW BSW None)

for method in "${methods[@]}"
do
    python3 mnist_trainer.py --lr 0.001 --epochs 10000 --log-epoch-interval 10 --dataset mnist --datadir data --outdir result --optimizer adam --weight_swd 8 --weight_fsw 0.3 --method $method
done