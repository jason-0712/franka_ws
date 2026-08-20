#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import time
import re
import numpy as np
from std_msgs.msg import Float64MultiArray


class ServoReaderNode(Node):
    def __init__(self):
        super().__init__("servo_reader_node")

        # Parameters
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("publish_rate", 50.0)  # Hz

        self.SERIAL_PORT = self.get_parameter("serial_port").value
        self.BAUDRATE = self.get_parameter("baudrate").value
        publish_rate = float(self.get_parameter("publish_rate").value)

        # Publisher
        self.pub = self.create_publisher(Float64MultiArray, "/servo_angles", 10)
        self.get_logger().info("Created publisher for /servo_angles")
        # Timer instead of rospy.Rate
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        # Serial
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
            self.get_logger().info(f"Serial port {self.SERIAL_PORT} opened")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            raise

        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7

        # State for interpolation
        self.angle_offset = [0.0] * 7          # currently published angles
        self.target_angle_offset = [0.0] * 7   # target angles
        self.num_interp = 5                    # kept for compatibility (not used directly)
        self.step_size = 1.0                   # minimum change threshold (deg)

        self._init_servos()

    def send_command(self, cmd: str) -> str:
        try:
            self.ser.write(cmd.encode("ascii"))
            time.sleep(0.008)
            return self.ser.read_all().decode("ascii", errors="ignore")
        except Exception as e:
            self.get_logger().warn(f"Serial error: {e}")
            return ""

    def pwm_to_angle(self, response_str, pwm_min=500, pwm_max=2500, angle_range=270):
        match = re.search(r"P(\d{4})", response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle

    def _init_servos(self):
        self.send_command("#000PVER!")
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f"#{i:03d}PULK!")
            response = self.send_command(f"#{i:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
        self.get_logger().info("Servo initial angle calibration completed")

    def timer_callback(self):
        # Read all servos and update targets
        for i in range(7):
            response = self.send_command(f"#{i:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip())
            if angle is not None:
                new_angle = angle - self.zero_angles[i]
                if abs(new_angle - self.target_angle_offset[i]) > self.step_size:
                    self.target_angle_offset[i] = new_angle
            else:
                self.get_logger().warn(f"Servo {i} response error: {response.strip()}")

        # Single interpolation step toward target (non-blocking)
        for i in range(7):
            delta = self.target_angle_offset[i] - self.angle_offset[i]
            self.angle_offset[i] += delta * 0.2  # smoothing factor

        # Publish
        msg = Float64MultiArray(data=self.angle_offset)
        self.pub.publish(msg)
        self.get_logger().info("Published /servo_angles")

    def destroy_node(self):
        if hasattr(self, "ser") and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("Serial port closed")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()