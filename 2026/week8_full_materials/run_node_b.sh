#!/bin/bash
set -e
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python train_8gpu.py --optimizer SignGD --gpu 0 \
    > logs/SignGD_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_8gpu.py --optimizer Adam   --gpu 0 \
    > logs/Adam_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train_8gpu.py --optimizer AdamW  --gpu 0 \
    > logs/AdamW_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_8gpu.py --optimizer Muon   --gpu 0 \
    > logs/Muon_stdout.log 2>&1 &

echo "Node B launched 4 jobs, PIDs: $(jobs -p)"
wait
echo "Node B all done"