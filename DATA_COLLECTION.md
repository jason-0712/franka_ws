# Data Collection Guide

## Terminal 1: Camera

Run on the Ubuntu host:

```bash
cd /home/dase-hw101/franka_ws/src/camera_driver
docker-compose up -d
docker-compose restart
docker logs --tail 80 realsense_ros2
```

Good camera log:

```text
RealSense Node Is Up!
```

## Terminal 2: Franka Controller

Run on the Ubuntu host:

```bash
docker start franka
docker exec -it franka bash
```

Inside Docker:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
unset ROS_DOMAIN_ID
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch crisp_controllers_robot_demos franka.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

Leave this terminal running.

## Terminal 3: Leader Arm

Run on the Ubuntu host:

```bash
docker exec -it franka bash

ros2 run leader_arm servo_reader.py --ros-args   -p 
serial_port:=/dev/ttyUSB0   -p publish_rate:=2.0

if permision denied, run: sudo chmod 666 /dev/ttyUSB0(1)


## Terminal 3: Recorder

Run on the Ubuntu host:

```bash
docker exec -it franka bash
```

Inside Docker:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
unset XAUTHORITY
unset ROS_DOMAIN_ID
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```


```Start the recorder:

```bash
python src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py \
  --repo-id hku/franka_test_001 \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --follower-config franka \
  --no-push-to-hub
```

Press `r` in this terminal to start/stop recording.



