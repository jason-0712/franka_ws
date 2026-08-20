#!/usr/bin/env python3
"""Capture one fresh frame from the Franka third-person and wrist cameras."""

import argparse
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-topic",
        default="/right/right_third_person_camera/color/image_raw",
    )
    parser.add_argument(
        "--wrist-topic",
        default="/right/right_wrist_camera/color/image_raw",
    )
    parser.add_argument("--output-dir", default="/tmp/franka_camera_frames")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


class FrameCapture(Node):
    def __init__(self, args):
        super().__init__("franka_camera_frame_capture")
        self.bridge = CvBridge()
        self.frames = {}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            # Match the live StarVLA client and RealSense image publishers.
            # TRANSIENT_LOCAL requires a transient publisher and silently
            # prevents this diagnostic subscriber from receiving live frames.
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, args.primary_topic, self._primary_cb, qos)
        self.create_subscription(Image, args.wrist_topic, self._wrist_cb, qos)

    def _primary_cb(self, msg):
        self.frames["primary"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _wrist_cb(self, msg):
        self.frames["wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def main():
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FrameCapture(args)
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and len(node.frames) < 2 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        missing = sorted({"primary", "wrist"} - set(node.frames))
        if missing:
            raise TimeoutError(f"Timed out waiting for camera frame(s): {missing}")
        for name, frame in node.frames.items():
            path = output_dir / f"{name}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Failed to write {path}")
            print(f"{name}: shape={frame.shape} path={path}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
