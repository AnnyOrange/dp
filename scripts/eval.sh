#!/bin/bash

# 定义参数
CHECKPOINT='data/checkpoints/can_ph_1.ckpt'  # 单个 checkpoint
DEVICE=4  # 对应的 GPU 设备
SPEEDS=(2 3)  # 要使用的 speed 参数
OUTPUTDIR='data/can_eval/'  # 对应的 output_dir
CLOSELOOP=True
TE=True
ENTROPY=False
# 遍历 SPEEDS，每个任务都用4个speed
for SPEED in "${SPEEDS[@]}"; do
  # 为每个 speed 创建对应的 output_dir
  SPEED_OUTPUTDIR="${OUTPUTDIR}/can_TE_${TE}_speed_${SPEED}_closeloop${CLOSELOOP}_no_prior_8_entropy_${ENTROPY}"

  # 启动任务，并行执行
  python eval.py --checkpoint ${CHECKPOINT} --output_dir ${SPEED_OUTPUTDIR} --device cuda:${DEVICE} --speed ${SPEED} --closeloop ${CLOSELOOP} --te ${TE} --is_entropy ${ENTROPY}&
done

# 等待所有后台任务完成
wait
