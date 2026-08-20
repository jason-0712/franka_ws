# Franka Workspace Backup and Reproduction Plan

Date: 2026-08-20

This repository is the top-level backup for the Quest 3 + Franka teleoperation,
data collection, and StarVLA deployment workspace.

## Backup Split

Use GitHub for code and reproducibility metadata:

- top-level runbooks and summaries
- `scripts/`
- `tests/`
- `assets/`
- `dataset_manifests/`
- non-submodule source packages such as `src/franka_description`,
  `src/franka_ros2`, `src/camera_driver`, and `src/leader_arm`
- git submodule pointers for external repos under `src/`
- git submodule pointers for StarVLA/deployment repos under `third_party/`
- local patch files for modified submodules under `patches/nested_repos/`

Use DockerHub for runnable environments:

- `crisp_controllers_demos:franka-overlay`
- `franka:latest`
- `realsense_ros2:latest`

Use HuggingFace or external artifact storage for datasets and large training
bundles:

- `dataset/`
- `artifacts/`
- `*.tar`
- `*.tar.gz`
- videos, parquet files, image frame dumps

These large files are intentionally ignored by Git.

## GitHub Backup

The intended remote is:

```bash
git remote add origin https://github.com/jason-0712/franka_ws.git
```

If `origin` already exists:

```bash
git remote set-url origin https://github.com/jason-0712/franka_ws.git
```

Generate fresh nested-repo patches before committing:

```bash
bash scripts/export_nested_repo_patches.sh
```

Then stage the reproducible code backup:

```bash
git add .gitignore .gitmodules BACKUP_REPRODUCTION.md
git add README.md README_hanyu.md *.md docs assets scripts tests dataset_manifests
git add src/camera_driver src/franka_description src/franka_ros2 src/leader_arm
git add src/crisp_controllers src/crisp_controllers_demos src/crisp_gym src/crisp_py
git add src/franka_broadcasters src/libfranka src/piper-vr-teleop
git add third_party/RLinf third_party/lingbot-vla third_party/starVLA third_party/starVLA_rl_libero
git add patches/nested_repos
git status
```

Commit and push:

```bash
git commit -m "Backup Franka Quest3 teleop and StarVLA deployment workspace"
git branch -M main
git push -u origin main
```

## DockerHub Backup

Log in first:

```bash
docker login
```

Set your DockerHub namespace:

```bash
export DOCKERHUB_USER=<your-dockerhub-username>
export BACKUP_TAG=20260820
```

Tag and push the images:

```bash
bash scripts/push_franka_images_to_dockerhub.sh "$DOCKERHUB_USER" "$BACKUP_TAG"
```

Recommended immutable tags:

- `franka-overlay-20260820`
- `franka-base-20260820`
- `realsense-ros2-20260820`

## Reproduce From GitHub

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/jason-0712/franka_ws.git
cd franka_ws
```

If submodules were cloned without recursion:

```bash
git submodule update --init --recursive
```

Apply local patches:

```bash
bash scripts/apply_nested_repo_patches.sh
```

Pull Docker images:

```bash
export DOCKERHUB_USER=<your-dockerhub-username>
export BACKUP_TAG=20260820

docker pull $DOCKERHUB_USER/franka-ws:franka-overlay-$BACKUP_TAG
docker pull $DOCKERHUB_USER/franka-ws:franka-base-$BACKUP_TAG
docker pull $DOCKERHUB_USER/franka-ws:realsense-ros2-$BACKUP_TAG
```

## Runtime: Camera

```bash
cd /home/dase-hw101/franka_ws/src/camera_driver
docker-compose up -d
docker-compose restart
docker logs --tail 80 realsense_ros2

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30
ros2 topic hz /right/right_third_person_camera/color/image_raw
```

## Runtime: Franka Cartesian Controller

Run inside the Franka container:

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=30

ros2 launch /home/ros/ros2_ws/src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka_cartesian_impedance.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

Use this Cartesian launch for Quest 3 teleoperation. Do not use
`franka.launch.py` for the final Quest 3 Cartesian data collection path.

## Runtime: Quest App and Bridge

Launch the Quest native passthrough app:

```bash
cd /home/ros/ros2_ws/src/piper-vr-teleop
scripts/launch_quest_mr_passthrough_app.sh
```

Check that passthrough is healthy:

```bash
adb logcat | grep wE9ryARX
```

Expected:

```text
OpenXR session: running
Passthrough: OK
```

Start the Quest ROS bridge:

```bash
cd /home/ros/ros2_ws/src/piper-vr-teleop
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/piper-vr-teleop:$PYTHONPATH

python3 scripts/quest_reader_ros_bridge.py --transport adb_logcat --rate 30
```

## Runtime: Quest Adapter

This adapter publishes 7D delta end-pose compatible commands:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/crisp_gym:/home/ros/ros2_ws/src/crisp_py:$PYTHONPATH

python -m crisp_gym.scripts.quest3_stream_adapter --ros-args \
  -p quest_pose_topic:=/quest/right_controller/pose \
  -p quest_joy_topic:=/quest/right_controller/joy \
  -p translation_scale:=0.35 \
  -p x_scale:=1.1 \
  -p y_scale:=1.1 \
  -p z_scale:=1.4 \
  -p max_translation_step:=0.004 \
  -p rotation_scale:=0.04 \
  -p max_rotation_step:=0.005 \
  -p rotation_representation:=euler_xyz \
  -p 'x_axis:="-x"' \
  -p 'y_axis:="z"' \
  -p 'z_axis:="y"' \
  -p deadman_button:=4 \
  -p deadman_latch_button:=-1 \
  -p reset_button:=-1 \
  -p pause_button:=-1 \
  -p gripper_button:=6 \
  -p trigger_axis:=5 \
  -p trigger_threshold:=0.8 \
  -p gripper_release_delay:=0.25 \
  -p record_transition_topic:=/record_transition \
  -p record_button:=4 \
  -p save_button:=0 \
  -p delete_button:=1 \
  -p gripper_requires_deadman:=false \
  -p franka_gripper_command_topic:=/gripper/gripper_position_controller/commands
```

## Runtime: LeRobot Recording

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/crisp_gym:/home/ros/ros2_ws/src/crisp_py:$PYTHONPATH

python -m crisp_gym.scripts.record_lerobot_format_leader_follower \
  --use-quest3-controller \
  --follower-config franka \
  --streamed-pose-topic /phone_pose \
  --streamed-gripper-topic /phone_gripper \
  --recording-manager-type ros \
  --repo-id snkdjn/quest3_franka_dualcam_backup_test \
  --tasks "pick up the cube and place it on the box" \
  --num-episodes 3 \
  --fps 15 \
  --push-to-hub \
  --use-current-pose-as-episode-start \
  --streamed-teleop-timeout 60 \
  --disable-recorder-gripper-control
```

## Runtime: StarVLA Deployment Path

For final alignment, use the 7D delta end-pose route:

```text
model output:
[dx, dy, dz, droll, dpitch, dyaw, gripper]

client:
integrate delta end pose -> publish /target_pose

controller:
cartesian_impedance_controller subscribes /target_pose
```

The older leader-arm joint path is a different action space and should not be
mixed directly with this Quest 3 7D Cartesian data unless converted.
