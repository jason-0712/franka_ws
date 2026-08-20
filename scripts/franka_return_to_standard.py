#!/usr/bin/env python3
"""Safely return the real Franka to the dual-camera training start pose.

The script is a dry run unless ``--execute`` is supplied.  When executing it
first lifts vertically if the tool is low, then interpolates the full Cartesian
pose, applies a few small feedback corrections, and opens the gripper only once
the tool is back at the high reset pose.
"""

import argparse
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import Float64MultiArray


# Mean start pose measured from the accepted dual-camera demonstrations.
DEFAULT_POSITION = (0.30885236, 0.00086458, 0.58483693)
DEFAULT_RPY = (-3.13512795, 0.00900896, -0.76660654)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safely return Franka to the dual-camera training start pose."
    )
    parser.add_argument(
        "--standard-position", nargs=3, type=float, default=DEFAULT_POSITION,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--standard-rpy", nargs=3, type=float, default=DEFAULT_RPY,
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--safe-lift-z", type=float, default=0.35)
    parser.add_argument("--lift-duration", type=float, default=5.0)
    parser.add_argument("--move-duration", type=float, default=8.0)
    parser.add_argument("--correction-duration", type=float, default=2.0)
    parser.add_argument("--settle-duration", type=float, default=1.0)
    parser.add_argument("--max-corrections", type=int, default=4)
    parser.add_argument("--feedback-gain", type=float, default=0.8)
    parser.add_argument("--max-position-correction", type=float, default=0.02)
    parser.add_argument("--max-rotation-correction-deg", type=float, default=6.0)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--open-duration", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--current-pose-topic", default="/current_pose")
    parser.add_argument("--target-pose-topic", default="/target_pose")
    parser.add_argument(
        "--gripper-command-topic",
        default="/gripper/gripper_position_controller/commands",
    )
    parser.add_argument("--frame-id", default="base")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually move the robot. Without this flag the script only prints the plan.",
    )
    return parser.parse_args()


class SafeResetNode(Node):
    def __init__(self, args):
        super().__init__("franka_return_to_standard")
        self.args = args
        self.actual_position = None
        self.actual_rotation = None
        self.pose_publisher = None
        self.gripper_publisher = None
        self.create_subscription(PoseStamped, args.current_pose_topic, self._pose_cb, 20)

    def _pose_cb(self, msg):
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        if np.all(np.isfinite(position)) and np.all(np.isfinite(quat)):
            self.actual_position = position
            self.actual_rotation = Rotation.from_quat(quat)

    def wait_for_pose(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and self.actual_position is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.actual_position is None:
            raise TimeoutError(f"未收到 {self.args.current_pose_topic}")

    def publish_pose(self, position, rotation):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
        quat = rotation.as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y = map(float, quat[:2])
        msg.pose.orientation.z, msg.pose.orientation.w = map(float, quat[2:])
        self.pose_publisher.publish(msg)

    def interpolate(self, start_position, start_rotation, end_position, end_rotation, duration):
        validate_segment(start_position, start_rotation, end_position, end_rotation)
        rotations = Rotation.from_quat(
            np.vstack((start_rotation.as_quat(), end_rotation.as_quat()))
        )
        slerp = Slerp([0.0, 1.0], rotations)
        start_time = time.monotonic()
        period = 1.0 / self.args.rate
        next_tick = start_time
        while rclpy.ok():
            linear_t = min(1.0, (time.monotonic() - start_time) / duration)
            smooth_t = linear_t * linear_t * (3.0 - 2.0 * linear_t)
            position = start_position + smooth_t * (end_position - start_position)
            self.publish_pose(position, slerp([smooth_t])[0])
            rclpy.spin_once(self, timeout_sec=0.0)
            if linear_t >= 1.0:
                return
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
        raise KeyboardInterrupt

    def hold_pose(self, position, rotation, duration):
        deadline = time.monotonic() + duration
        period = 1.0 / self.args.rate
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_pose(position, rotation)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def open_gripper(self):
        if self.count_subscribers(self.args.gripper_command_topic) < 1:
            print(f"警告：{self.args.gripper_command_topic} 没有订阅者，未发送夹爪打开命令")
            return False
        command = Float64MultiArray()
        command.data = [1.0]
        deadline = time.monotonic() + self.args.open_duration
        period = 1.0 / min(self.args.rate, 30.0)
        while rclpy.ok() and time.monotonic() < deadline:
            self.gripper_publisher.publish(command)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        return True


def validate_position(position):
    if not np.all(np.isfinite(position)):
        raise ValueError(f"位姿包含非有限值：{position}")
    if not (0.20 <= position[0] <= 0.65 and -0.30 <= position[1] <= 0.30):
        raise ValueError(f"XY 超出安全回位工作区：{position}")
    if not (0.10 <= position[2] <= 0.70):
        raise ValueError(f"Z 超出安全回位工作区：{position[2]:.4f}")


def validate_segment(start_position, start_rotation, end_position, end_rotation):
    validate_position(start_position)
    validate_position(end_position)
    distance = float(np.linalg.norm(end_position - start_position))
    angle = float((start_rotation.inv() * end_rotation).magnitude())
    if distance > 0.40:
        raise RuntimeError(f"单段位移 {distance:.3f} m 超过 0.40 m 安全限制")
    if angle > math.pi / 2:
        raise RuntimeError(f"单段旋转 {math.degrees(angle):.1f} deg 超过 90 deg 安全限制")


def clipped_vector(vector, max_norm):
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


def pose_error(target_position, target_rotation, actual_position, actual_rotation):
    position_error = target_position - actual_position
    local_rotation_error = actual_rotation.inv() * target_rotation
    return position_error, local_rotation_error


def print_error(prefix, position_error, rotation_error):
    print(
        f"{prefix}: position_error={1000.0 * np.linalg.norm(position_error):.2f} mm, "
        f"rotation_error={math.degrees(rotation_error.magnitude()):.2f} deg"
    )


def validate_args(args):
    positive = {
        "--safe-lift-z": args.safe_lift_z,
        "--lift-duration": args.lift_duration,
        "--move-duration": args.move_duration,
        "--correction-duration": args.correction_duration,
        "--settle-duration": args.settle_duration,
        "--position-tolerance": args.position_tolerance,
        "--rotation-tolerance-deg": args.rotation_tolerance_deg,
        "--open-duration": args.open_duration,
        "--rate": args.rate,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if args.max_corrections < 0:
        raise ValueError("--max-corrections 不能小于 0")
    if not (0.0 < args.feedback_gain <= 1.0):
        raise ValueError("--feedback-gain 必须在 (0, 1] 内")
    if args.max_position_correction <= 0 or args.max_rotation_correction_deg <= 0:
        raise ValueError("反馈补偿限值必须大于 0")


def main():
    args = parse_args()
    validate_args(args)
    target_position = np.asarray(args.standard_position, dtype=float)
    target_rotation = Rotation.from_euler("xyz", args.standard_rpy)
    validate_position(target_position)
    if not (0.10 <= args.safe_lift_z <= target_position[2]):
        raise ValueError("--safe-lift-z 必须在 0.10 m 与标准位 Z 之间")

    rclpy.init()
    node = SafeResetNode(args)
    try:
        node.wait_for_pose()
        start_position = node.actual_position.copy()
        start_rotation = node.actual_rotation
        needs_lift = start_position[2] < args.safe_lift_z
        planned_start = start_position.copy()
        if needs_lift:
            planned_start[2] = args.safe_lift_z
            validate_segment(
                start_position, start_rotation, planned_start, start_rotation
            )
        validate_segment(
            planned_start, start_rotation, target_position, target_rotation
        )

        position_error, rotation_error = pose_error(
            target_position, target_rotation, start_position, start_rotation
        )
        print(f"当前位置：{np.array2string(start_position, precision=6)}")
        print(f"标准位置：{np.array2string(target_position, precision=6)}")
        print_error("当前误差", position_error, rotation_error)
        print(f"安全路径：{'先垂直抬升，再恢复完整位姿' if needs_lift else '直接平滑恢复完整位姿'}")
        print(f"execute={args.execute}")
        if not args.execute:
            print("DRY RUN：机器人未移动；确认后加 --execute。")
            return

        if node.count_publishers(args.target_pose_topic) > 0:
            raise RuntimeError(f"{args.target_pose_topic} 已有其他发布者，拒绝抢占控制")
        if node.count_publishers(args.gripper_command_topic) > 0:
            raise RuntimeError(f"{args.gripper_command_topic} 已有其他发布者，拒绝抢占夹爪")

        node.pose_publisher = node.create_publisher(PoseStamped, args.target_pose_topic, 1)
        node.gripper_publisher = node.create_publisher(
            Float64MultiArray, args.gripper_command_topic, 1
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and node.count_subscribers(args.target_pose_topic) < 1:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.count_subscribers(args.target_pose_topic) < 1:
            raise RuntimeError(f"{args.target_pose_topic} 没有控制器订阅者")

        command_position = start_position.copy()
        command_rotation = start_rotation
        if needs_lift:
            lift_position = start_position.copy()
            lift_position[2] = args.safe_lift_z
            print(f"阶段 1/3：垂直抬升到 z={args.safe_lift_z:.3f} m")
            node.interpolate(
                command_position, command_rotation,
                lift_position, command_rotation,
                args.lift_duration,
            )
            node.hold_pose(lift_position, command_rotation, args.settle_duration)
            command_position = lift_position

        print("阶段 2/3：平滑恢复训练标准位姿")
        node.interpolate(
            command_position, command_rotation,
            target_position, target_rotation,
            args.move_duration,
        )
        command_position = target_position.copy()
        command_rotation = target_rotation
        node.hold_pose(command_position, command_rotation, args.settle_duration)

        converged = False
        rotation_tolerance = math.radians(args.rotation_tolerance_deg)
        max_rotation_correction = math.radians(args.max_rotation_correction_deg)
        for correction_index in range(args.max_corrections + 1):
            rclpy.spin_once(node, timeout_sec=0.05)
            actual_position = node.actual_position.copy()
            actual_rotation = node.actual_rotation
            position_error, rotation_error = pose_error(
                target_position, target_rotation, actual_position, actual_rotation
            )
            print_error(f"反馈检查 {correction_index}", position_error, rotation_error)
            if (
                np.linalg.norm(position_error) <= args.position_tolerance
                and rotation_error.magnitude() <= rotation_tolerance
            ):
                converged = True
                break
            if correction_index == args.max_corrections:
                break

            position_step = clipped_vector(
                args.feedback_gain * position_error,
                args.max_position_correction,
            )
            rotation_step = clipped_vector(
                args.feedback_gain * rotation_error.as_rotvec(),
                max_rotation_correction,
            )
            corrected_position = command_position + position_step
            corrected_rotation = command_rotation * Rotation.from_rotvec(rotation_step)
            validate_segment(
                command_position, command_rotation,
                corrected_position, corrected_rotation,
            )
            print(
                f"反馈补偿 {correction_index + 1}/{args.max_corrections}: "
                f"dpos={np.array2string(position_step, precision=5)}"
            )
            node.interpolate(
                command_position, command_rotation,
                corrected_position, corrected_rotation,
                args.correction_duration,
            )
            command_position = corrected_position
            command_rotation = corrected_rotation
            node.hold_pose(command_position, command_rotation, args.settle_duration)

        if not converged:
            raise RuntimeError("已达到反馈补偿次数上限，但实际位姿仍未进入标准位容差")

        print("阶段 3/3：在高位打开夹爪")
        opened = node.open_gripper()
        node.hold_pose(command_position, command_rotation, 0.5)
        print(f"完成：Franka 已回到标准位；夹爪打开命令={'已发送' if opened else '未发送'}。")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
