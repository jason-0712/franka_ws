cmake_minimum_required(VERSION 3.5)
project(leader_arm)

# 找到ament和依赖
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# 安装Python脚本为可执行文件
install(PROGRAMS
  servo_reader.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
