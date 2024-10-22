#!/bin/bash

# 定义参数
CHECKPOINTS=('data/checkpoints/lift_ph_1.ckpt')  # 填入多个 checkpoint
DEVICES=(0)                                          # 对应每个任务使用的 GPU 设备，和上面的checkpoint对应
SPEEDS=(1 2 3 4)                                         # 要使用的 speed 参数，也就是几倍速
OUTPUTDIRS=('data/lift_eval/')        # 对应每个 checkpoint 的 output_dir

# 遍历 checkpoint, output_dir 和 device，并运行任务
for i in "${!CHECKPOINTS[@]}"; do
  CHECKPOINT=${CHECKPOINTS[$i]}
  OUTPUTDIR=${OUTPUTDIRS[$i]}
  DEVICE=${DEVICES[$i]}

  # 遍历 SPEEDS，每个任务都用4个speed
  for SPEED in "${SPEEDS[@]}"; do
    # 为每个 speed 创建对应的 output_dir
    SPEED_OUTPUTDIR="${OUTPUTDIR}/speed${SPEED}"
    
    # 启动任务，并行执行
    python eval.py --checkpoint ${CHECKPOINT} --output_dir ${SPEED_OUTPUTDIR} --device cuda:${DEVICE} --speed ${SPEED} &
  done
done

# 等待所有后台任务完成
wait
