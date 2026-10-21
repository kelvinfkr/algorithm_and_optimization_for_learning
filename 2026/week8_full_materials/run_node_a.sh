#!/bin/bash
set -e
cd /path/to/your/code
mkdir -p logs

# 每个优化器用一张卡，后台跑，日志落盘
CUDA_VISIBLE_DEVICES=0 python train_8gpu.py --optimizer SGD       --gpu 0 \
    > logs/SGD_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_8gpu.py --optimizer HeavyBall --gpu 0 \
    > logs/HeavyBall_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train_8gpu.py --optimizer Nesterov  --gpu 0 \
    > logs/Nesterov_stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_8gpu.py --optimizer SGD_WD    --gpu 0 \
    > logs/SGD_WD_stdout.log 2>&1 &

echo "Node A launched 4 jobs, PIDs: $(jobs -p)"
wait
echo "Node A all done"