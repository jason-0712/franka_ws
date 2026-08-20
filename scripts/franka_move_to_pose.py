#!/usr/bin/env python3
"""Smoothly interpolate the active Cartesian controller to an absolute pose."""

import argparse
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation, Slerp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--rpy", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--current-pose-topic", default="/current_pose")
    parser.add_argument("--target-pose-topic", default="/target_pose")
    parser.add_argument("--frame-id", default="base")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


class PoseInterpolator(Node):
    def __init__(self, args):
        super().__init__("franka_pose_interpolator")
        self.args = args
        self.position = None
        self.rotation = None
        self.create_subscription(PoseStamped, args.current_pose_topic, self._pose_cb, 20)
        self.publisher = None

    def _pose_cb(self, msg):
        self.position = np.array(
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
        self.rotation = Rotation.from_quat(quat)

    def wait_for_pose(self, timeout=5.0):
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline and self.position is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.position is None:
            raise TimeoutError(f"No pose received from {self.args.current_pose_topic}")

    def publish(self, position, rotation):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
        quat = rotation.as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y = map(float, quat[:2])
        msg.pose.orientation.z, msg.pose.orientation.w = map(float, quat[2:])
        self.publisher.publish(msg)


def main():
    args = parse_args()
    if args.duration <= 0 or args.rate <= 0:
        raise ValueError("--duration and --rate must be positive")

    target_position = np.asarray(args.position, dtype=float)
    if not np.all(np.isfinite(target_position)):
        raise ValueError("Target position must be finite")
    if not (0.20 <= target_position[0] <= 0.65 and -0.30 <= target_position[1] <= 0.30):
        raise ValueError(f"Target XY outside reset workspace: {target_position}")
    if not (0.10 <= target_position[2] <= 0.70):
        raise ValueError(f"Target Z outside reset workspace: {target_position[2]}")

    rclpy.init()
    node = PoseInterpolator(args)
    try:
        node.wait_for_pose()
        start_position = node.position.copy()
        start_rotation = node.rotation
        target_rotation = start_rotation if args.rpy is None else Rotation.from_euler("xyz", args.rpy)
        distance = float(np.linalg.norm(target_position - start_position))
        angle = float((target_rotation * start_rotation.inv()).magnitude())
        print(f"start_position={start_position} target_position={target_position}")
        print(f"distance={distance:.4f}m rotation={angle:.4f}rad execute={args.execute}")
        if distance > 0.40 or angle > math.pi / 2:
            raise RuntimeError("Requested reset segment is too large")
        if not args.execute:
            return
        if node.count_publishers(args.target_pose_topic) > 0:
            raise RuntimeError(f"{args.target_pose_topic} already has another publisher")
        node.publisher = node.create_publisher(PoseStamped, args.target_pose_topic, 1)
        deadline = time.time() + 3.0
        while time.time() < deadline and node.count_subscribers(args.target_pose_topic) < 1:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.count_subscribers(args.target_pose_topic) < 1:
            raise RuntimeError(f"No subscriber on {args.target_pose_topic}")

        slerp = Slerp([0.0, 1.0], Rotation.concatenate([start_rotation, target_rotation]))
        start_time = time.monotonic()
        period = 1.0 / args.rate
        while rclpy.ok():
            linear_t = min(1.0, (time.monotonic() - start_time) / args.duration)
            smooth_t = linear_t * linear_t * (3.0 - 2.0 * linear_t)
            position = start_position + smooth_t * (target_position - start_position)
            node.publish(position, slerp([smooth_t])[0])
            rclpy.spin_once(node, timeout_sec=0.0)
            if linear_t >= 1.0:
                break
            time.sleep(period)
        for _ in range(max(1, int(args.rate * 0.5))):
            node.publish(target_position, target_rotation)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
        print("finished")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
