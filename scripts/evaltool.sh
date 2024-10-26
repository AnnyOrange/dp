#!/bin/bash

# 定义参数
CHECKPOINT="data/checkpoints/tool_ph_2.ckpt"  # 单个 checkpoint
OUTPUTDIR="data/tool_eval/"                   # 输出目录
CLOSELOOP=True
TE=True
# 遍历 SPEEDS 并分配到同一个 GPU 设备
for SPEED in 2 3 4; do
  # 为每个 speed 创建对应的 output_dir (在 cuda:0 上运行)
  SPEED_OUTPUTDIR="${OUTPUTDIR}/tool_TE_${TE}_speed_${SPEED}_closeloop${CLOSELOOP}_no_prior"
  python eval.py --checkpoint "$CHECKPOINT" --output_dir "$SPEED_OUTPUTDIR" --device cuda:3 --speed "$SPEED" --closeloop "$CLOSELOOP" --te "$TE"&
done

# 等待所有后台任务完成
wait
