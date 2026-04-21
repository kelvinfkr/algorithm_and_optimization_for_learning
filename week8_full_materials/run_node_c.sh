#!/bin/bash
set -e
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python train_8gpu.py --optimizer Lion          --gpu 0 \
    > logs/Lion_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_8gpu.py --optimizer Sophia        --gpu 0 \
    > logs/Sophia_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train_8gpu.py --optimizer CautiousAdamW --gpu 0 \
    > logs/CautiousAdamW_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_8gpu.py --optimizer Shampoo       --gpu 0 \
    > logs/Shampoo_stdout.log 2>&1 &

echo "Node C launched 4 jobs, PIDs: $(jobs -p)"
wait
echo "Node C all done"