Depend on ROS2, see [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html) for instructions to install ROS2 Iron. Remember to install the dev tools for ROS2.

Assuming ROS2 is sourced in your terminal

To build:
```
# clone the repo
cd track_detection_ros_ws
colcon build
```

To run:
```
source install/setup.bash
ros2 run lane_detection lane_detection_node