#!/bin/bash

# 定义参数
CHECKPOINT='data/checkpoints/can_ph_1.ckpt'  # 单个 checkpoint
DEVICE=0  # 对应的 GPU 设备
SPEEDS=(1 2 3 4)  # 要使用的 speed 参数
OUTPUTDIR='data/can_eval/'  # 对应的 output_dir

# 遍历 SPEEDS，每个任务都用4个speed
for SPEED in "${SPEEDS[@]}"; do
  # 为每个 speed 创建对应的 output_dir
  SPEED_OUTPUTDIR="${OUTPUTDIR}/speed_${SPEED}"

  # 启动任务，并行执行
  python eval.py --checkpoint ${CHECKPOINT} --output_dir ${SPEED_OUTPUTDIR} --device cuda:${DEVICE} --speed ${SPEED} &
done

# 等待所有后台任务完成
wait
