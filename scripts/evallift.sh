#!/bin/bash

# 定义参数
CHECKPOINT="data/checkpoints/lift_ph_1.ckpt"  # 单个 checkpoint
DEVICE=2                                      # 对应的 GPU 设备
OUTPUTDIR="data/lift_eval/"                   # 输出目录

# 遍历 SPEEDS
for SPEED in {1..4}; do
  # 为每个 speed 创建对应的 output_dir
  SPEED_OUTPUTDIR="${OUTPUTDIR}/speed_plot${SPEED}"
  
  # 启动任务
  python eval.py --checkpoint "$CHECKPOINT" --output_dir "$SPEED_OUTPUTDIR" --device cuda:"$DEVICE" --speed "$SPEED" &
done

# 等待所有后台任务完成
wait
