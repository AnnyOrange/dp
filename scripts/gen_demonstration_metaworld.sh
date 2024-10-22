#!/bin/bash
# bash scripts/gen_demonstration_metaworld.sh
#  "coffee-pull" "coffee-push" "bin-picking" "soccer" "stick-push" "pick-place" "window-open" "window-close" "reach"
# 定义任务名称的数组
task_names=(
"peg-unplug-side"
"sweep"
"handle-pull-side"
"soccer"
"push")


# task_names=("basketball" "box-close" "button-press-topdown" "button-press-topdown-wall" "button-press" "button-press-wall" "coffee-button" "dial-turn" "disassemble" "door-close" "door-lock" "door-open" "door-unlock" "hand-insert" "drawer-close" "drawer-open" 
#             "faucet-open" 
#             "faucet-close" 
#             "hammer" 
#             "handle-press-side" 
#             "handle-press" 
#             "handle-pull-side" 
#             "handle-pull" 
#             "lever-pull" 
#             "peg-insert-side" 
#             "pick-place-wall" 
#             "pick-out-of-hole" 
#             "push-back" 
#             "push" 
#             "pick-place" 
#             "plate-slide" 
#             "plate-slide-side" 
#             "plate-slide-back" 
#             "plate-slide-back-side" 
#             "peg-unplug-side" 
#             "stick-push" 
#             "stick-pull" 
#             "push-wall" 
#             "reach-wall" 
#             "shelf-place" 
#             "sweep-into" 
#             "sweep")
# 进入指定目录
cd third_party/Metaworld

# 指定可见的 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 循环遍历每个任务
for task_name in "${task_names[@]}"
do
    # 打印当前执行的任务名称
    echo "Running task: ${task_name}"

    # 执行 Python 脚本，每个任务运行一次
    python gen_demonstration_expert_video.py --env_name="${task_name}" \
        --num_episodes 10 \
        --root_dir "../../3D-Diffusion-Policy/data/num_episodes10"
    
    # 检查上一个命令是否执行成功，如果失败则退出循环
    if [ $? -ne 0 ]; then
        echo "Error occurred in task: ${task_name}. Exiting..."
        exit 1
    fi
done

echo "All tasks completed!"
