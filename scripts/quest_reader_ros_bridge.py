#!/usr/bin/env python3
"""Publish Piper VR QuestReader samples as ROS 2 PoseStamped and Joy topics."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to xyzw quaternion."""
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return qx, qy, qz, qw


def analog(buttons: dict, key: str) -> float:
    """Read a normalized Piper VR analog button value."""
    value = buttons.get(key, (0.0,))
    if isinstance(value, (tuple, list)) and value:
        return float(value[0])
    if isinstance(value, (bool, int, float)):
        return float(value)
    return 0.0


class QuestReaderRosBridge(Node):
    """Bridge QuestReader samples to generic ROS 2 controller topics."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("quest_reader_ros_bridge")

        repo = Path(args.piper_repo).expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from piper_vr.quest_reader import QuestReader

        self.side = args.side
        self.frame_id = args.frame_id
        self.grip_threshold = args.grip_threshold
        self.trigger_threshold = args.trigger_threshold
        self.reader = QuestReader(
            transport=args.transport,
            connection=args.connection,
            ip_address=args.quest_ip,
            simulate_on_missing=args.simulate,
        )

        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 10)
        self.joy_pub = self.create_publisher(Joy, args.joy_topic, 10)
        self.create_timer(1.0 / args.hz, self.publish_latest)

        self.get_logger().info(f"Reading Quest samples from {repo}")
        self.get_logger().info(f"Publishing PoseStamped on {args.pose_topic}")
        self.get_logger().info(f"Publishing Joy on {args.joy_topic}")

    def publish_latest(self) -> None:
        sample = self.reader.get_sample()
        if sample is None:
            return

        transform = sample.transforms_openxr.get(self.side)
        if transform is None:
            return

        transform = np.asarray(transform, dtype=float)
        stamp = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(transform[0, 3])
        pose_msg.pose.position.y = float(transform[1, 3])
        pose_msg.pose.position.z = float(transform[2, 3])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(transform[:3, :3])
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg)

        buttons = sample.buttons
        right_grip = analog(buttons, "rightGrip")
        right_trigger = analog(buttons, "rightTrig")

        joy_msg = Joy()
        joy_msg.header.stamp = stamp
        joy_msg.header.frame_id = self.frame_id
        joy_msg.buttons = [
            int(bool(buttons.get("A", False))),
            int(bool(buttons.get("B", False))),
            int(bool(buttons.get("X", False))),
            int(bool(buttons.get("Y", False))),
            int(right_grip >= self.grip_threshold),
        ]
        joy_msg.axes = [0.0] * 6
        joy_msg.axes[5] = -1.0 if right_trigger >= self.trigger_threshold else 1.0
        self.joy_pub.publish(joy_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--piper-repo", default="/home/ros/ros2_ws/src/piper-vr-teleop")
    parser.add_argument("--transport", default="adb_logcat")
    parser.add_argument("--connection", default="usb", choices=("usb", "wireless"))
    parser.add_argument("--quest-ip")
    parser.add_argument("--side", default="right", choices=("right", "left"))
    parser.add_argument("--pose-topic", default="/quest/right_controller/pose")
    parser.add_argument("--joy-topic", default="/quest/right_controller/joy")
    parser.add_argument("--frame-id", default="quest_openxr")
    parser.add_argument("--hz", "--rate", dest="hz", type=float, default=30.0)
    parser.add_argument("--grip-threshold", type=float, default=0.5)
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    parser.add_argument("--simulate", action="store_true")
    return parser.parse_args()


def main() -> int:
    rclpy.init()
    node = QuestReaderRosBridge(parse_args())
    try:
        rclpy.spin(node)
    finally:
        node.reader.stop()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
