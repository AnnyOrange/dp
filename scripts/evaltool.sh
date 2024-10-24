#!/bin/bash

# 定义参数
CHECKPOINT="data/checkpoints/tool_ph_2.ckpt"  # 单个 checkpoint
OUTPUTDIR="data/tool_eval/"                   # 输出目录

# 遍历 SPEEDS 并分配不同的 GPU 设备
for SPEED in 1; do
  # 为第一个 speed 创建对应的 output_dir (在 cuda:4 上运行)
  SPEED_OUTPUTDIR="${OUTPUTDIR}/speed_plot_${SPEED}"
  python eval.py --checkpoint "$CHECKPOINT" --output_dir "$SPEED_OUTPUTDIR" --device cuda:4 --speed "$SPEED" &
done

for SPEED in 2; do
  # 为第二个 speed 创建对应的 output_dir (在 cuda:5 上运行)
  SPEED_OUTPUTDIR="${OUTPUTDIR}/speed_plot_${SPEED}"
  python eval.py --checkpoint "$CHECKPOINT" --output_dir "$SPEED_OUTPUTDIR" --device cuda:5 --speed "$SPEED" &
done

for SPEED in 3; do
  # 为第三个 speed 创建对应的 output_dir (在 cuda:6 上运行)
  SPEED_OUTPUTDIR="${OUTPUTDIR}/speed_plot_${SPEED}"
  python eval.py --checkpoint "$CHECKPOINT" --output_dir "$SPEED_OUTPUTDIR" --device cuda:6 --speed "$SPEED" &
done

for SPEED in 4; do
  # 为第四个 speed 创建对应的 output_dir (在 cuda:7 上运行)
  SPEED_OUTPUTDIR="${OUTPUTDIR}/speed_plot_${SPEED}"
  python eval.py --checkpoint "$CHECKPOINT" --output_dir "$SPEED_OUTPUTDIR" --device cuda:7 --speed "$SPEED" &
done

# 等待所有后台任务完成
wait
