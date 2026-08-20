1. franka FCI
https://172.16.0.2/desk/ 

username franka
password 103m103m or franka123
username franka_so (for error reset the robot) 
password 103m103m or franka123
SET PC Wired IPV4 to 172.16.0.1
unlock
activate FCI 

![activate FCI](./assets/1.png)

2. camera
```bash
sudo docker restart realsense_ros2
```

docker run -it \
  --name franka \
  --network host \
  --privileged \
  -v /home/dase-hw101/franka_ws/src:/home/ros/ros2_ws/src \
  -v /home/dase-hw101/franka_ws/dataset:/home/ros/.cache/huggingface/lerobot/ \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /dev:/dev \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/root/.Xauthority \
  --cap-add=SYS_NICE \
  --ulimit rtprio=99 \
  --ulimit rttime=-1 \
  --ulimit memlock=8428281856 \ror loading config.toml: Failed to read project config file /home/dase-hw101/.codex/config.toml: Permission denied (os error 13)

  -w /home/ros/ros2_ws \
  franka:latest 
  
3. controller
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
sudo docker start franka
sudo docker exec -it franka bash
source install/setup.bash
ros2 launch crisp_controllers_robot_demos franka.launch.py \
  arm_id:=fr3 robot_ip:=172.16.0.2 use_fake_hardware:=false \
  use_rviz:=false

# new terminal
sudo docker exec -it franka bash
source install/setup.bash
# vla
python src/crisp_py/examples/deploy_franka.py
```

5. teleop without recording
```bash
if permision denied, run: sudo chmod 666 /dev/ttyUSB0(1)
ros2 run leader_arm servo_reader.py
python src/crisp_py/examples/02_joint_control.py
```
6. record
```bash
conda activate lerobot
ros2 run leader_arm servo_reader.py
python src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py
```
Franka_data huggingface Token 


outside docker contianer run :
```bash

cd src/crisp_py/examples/
python deploy_franka.py \
    --ctrl_freq 15 \
    --delta_scale 1.0 \
    --exec_mode playback \
    --max_action_age 0.5


sudo cpupower frequency-set -g performance
