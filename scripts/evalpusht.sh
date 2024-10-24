#!/bin/bash

# 定义参数
CHECKPOINT='data/outputs/2024.10.13/13.08.19_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=1600-test_mean_score=0.954.ckpt'  # 单个 checkpoint
DEVICE=0                                          # 单个 GPU 设备
SPEEDS=(1 2 3 4)                                  # 要使用的 speed 参数
OUTPUTDIR='data/pusht_eval/'                      # 单个 output_dir

# 遍历 SPEEDS，执行任务
for SPEED in "${SPEEDS[@]}"; do
  # 为每个 speed 创建对应的 output_dir
  SPEED_OUTPUTDIR="${OUTPUTDIR}/max_step_sp${SPEED}"
  
  # 启动任务，并行执行
  python eval.py --checkpoint ${CHECKPOINT} --output_dir ${SPEED_OUTPUTDIR} --device cuda:${DEVICE} --speed ${SPEED} &
done

# 等待所有后台任务完成
wait
