#!/usr/bin/env bash

# franka_auto_restart.sh

# 替换成你的实际 workspace 路径
WORKSPACE_SETUP="/home/cps/your_ws/install/setup.bash"   # ← 修改这里

# 第一次运行前确保 source 环境（可以加到 ~/.bashrc，但这里显式写出来）
source /opt/ros/humble/setup.bash   # 假设用 humble，改成你的版本
source "$WORKSPACE_SETUP"

while true; do
    echo "========================================"
    echo "Starting Franka CRISP launch at $(date)"
    echo "========================================"

    ros2 launch crisp_controllers_robot_demos franka.launch.py \
        arm_id:=fr3 \
        robot_ip:=172.16.0.2 \
        use_fake_hardware:=false \
        use_rviz:=false

    EXIT_CODE=$?
    echo "Launch exited with code $EXIT_CODE at $(date)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Normal exit (code 0), stopping auto-restart loop."
        break
    fi

    echo "Will restart in 5 seconds... (Ctrl+C to stop)"
    sleep 5
done

echo "Auto-restart loop ended."