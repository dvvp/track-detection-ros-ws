# Track Detection Using Deep Learning

by Devin Pham and Grace Chen

## Introduction

During the Winter and Spring quarters at UCSD in 2024, we were given the special opportunity to develop a track detection algorithm to be used in the [Autonomous Karting Series](https://www.autonomouskartingseries.com/) (AKS), a competition in which a number of universities came together to race autonomous electric go-karts. Due to various mechanical issues that we had with our go-kart, we were unable to test this algorithm and instead used [Donkeycar](https://github.com/Triton-AI/donkeycar). Unfortunately, the Donkeycar algorithm fell short of our expectations as the go-kart was unable to complete the entire track during the race. Because the algorithm we developed was able to make the go-kart complete the track without PID tuning on a simulator, we believe that a guide detailing the machine learning pipeline that we used would be of value to students who will be competing in an autonomous race like AKS or are simply interested in learning about deep learning and computer vision. To ensure that future teams have a solid foundation to build upon our work, we made an effort to make this guide as comprehensive as possible.

## Objectives
1. Create an algorithm that is able to detect the [Purdue Grand Prix Track](https://maps.app.goo.gl/f2X3zkNxbXbzi7fP6) using a camera mounted on the go-kart. This algorithm must be able to run directly on the camera itself or an NVIDIA Jetson module
2. Integrate algorithm with ROS driving controls to make the go-kart drive reactively (without pre-mapping)

## Resources Provided
1. [Luxonis Oak-D LR](https://shop.luxonis.com/products/oak-d-lr)

    The Luxonis Oak-D LR, with “LR'' standing for “long range”, is the main camera affixed to the front of our go-kart. It uses the left and right lenses for depth perception, while the middle lens functions as a color camera. For our purpose, we only used the color camera with less computational power.
     
    The full list of specifications for this camera can be found [here](https://docs-old.luxonis.com/projects/hardware/en/latest/pages/OAK-D-LR/). The main details to note about the color camera is that it has a 16:10 aspect ratio, 1200P resolution, and a maximum FPS of 30. In  order to run our model on this camera at the maximum FPS, we had to lower our resolution to something like 200P. We will provide more details to this augmentation later in the guide.

2. [Oak-D Pro](https://shop.luxonis.com/products/oak-d-pro-w?variant=43715946447071) / [Oak-D Lite](https://shop.luxonis.com/products/oak-d-lite-1?variant=42583102456031)

    These two cameras were used primarily for prototyping purposes for when we were unable to test using the LR. Two Oak-D Pros could potentially be pointed to the left and right of the go-kart for additional inference; This was what the [Donkeycar](https://github.com/Triton-AI/donkeycar) algorithm used during the our race.

3. [NVIDIA Jetson AGX Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/)

    We intended on deploying our model usign this computer module instead of running it directly ont he camera. The Jetson's GPU and specialized hardware accelerators allows it to make much faster inferences than running on a CPU. We did not get a chance to deploy our ROS package, but you can find instructions on how to do that [here](https://developer.nvidia.com/embedded/learn/getting-started-jetson).


4. LIDAR

    We did not utilize LIDAR in our algorithm at all, but incorporating it could potentially enhance performance, considering its widespread use in today's autonomous vehicles.

5. [OSSDC SIM](https://github.com/OSSDC/OSSDC-SIM.git)

    OSSDC SIM (Open Source Self Driving Car Simulator) is an open-source platform used to simulate the driving of autonomous cars and go-karts. Using a simulator is a cost-effective, safe, and scalable way to test autonomous vehicles that allows for rapid prototyping.

## Unsuccessful Attempts

Most of our failed attempts were from trying to use past repositories that specialized in lane detection. Most of these models used traditional computer vision methods using [OpenCV](https://opencv.org/about/), an open source computer vision and machine learning software library, which took a lot of effort to tweak as most of the code available online were outdated and/or had parameters that were difficult to adjust.

Something that we overlooked was that our model needed to differentiate between road and grass instead of lane and road. Most of the models we saw online differentiated between lane and road instead of road and grass (Note the distinction between *lane* detection and *road*/*track* detection). 
Using traditional computer vision methods also made track detections less robust in direct sunlight which is the main reason why we switched over to use deep learning instead of purely using computer vision techniques. Our race took place at around noon time so direct sunlight was a big issue.

## Introduction to Roboflow

Roboflow is a platform that allows developers to easily annotate images and deploy models for computer vision applications. If you are new to computer vision, annotation is the practice of labeling training data, which in our case, are images. Annotations can be in the form of bounding boxes (in the context of object detection), or masks (in the context of segmentation). This is usually a manual process, but it can be made faster through the use of pre-trained models and LLMs.

For track detection we will be focusing on segmentation instead of object detection. Segmentation is different from object detection in that we are not only identifying objects within an image, but we are also detecting the exact boundaries of each object at the pixel level. 

It is also important that we differentiate between the two types of segmentation: instance segmentation and semantic segmentation. Instance segmentation is able to distinguish between separate instances of the same class, while semantic segmentation does not. Although it is more computationally expensive, we will be using instance segmentation since it offers support for a lot more models.

![image](instance-vs-semantic.png)

Source: https://www.folio3.ai/blog/semantic-segmentation-vs-instance-segmentation/

Roboflow offers many options for model generation either directly through their platform or through Google Colab/Jupyter Notebooks. At the time this guide was written, the default model that Roboflow offers is called “Roboflow 3.0 Instance Segmentation”, and it is available directly through their platform. Because there are [no details provided about this model](https://discuss.roboflow.com/t/what-is-behind-the-roboflow-3-0-instance-segmentation-model/3720) or options to customize this model, we will be using the YOLO models available via their provided notebooks. YOLO (You only look once) is a state-of-the-art deep learning framework that enables real-time object detection and segmentation in images or videos. 

## Getting Started with Roboflow

## How to Annotate Images

## How to Train New Models

## ROS Integration

[Robotics Operating System](https://www.ros.org/) (ROS) needs to be integrated with your YOLO model to enable go-kart driving control.



Depending on ROS2, see [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html) for instructions to install ROS2 Iron. Remember to install the dev tools for ROS2. 

Below are instructions on how to run `track-detection-ros-ws` on OSSDC SIM:

Assuming ROS2 is sourced in your terminal

### To build:
```
# clone the repo
cd track-detection-ros-ws
colcon build
```

### To run:

In terminal 1, enter in the following commands:
```
cd ~/projects/race_common/OSSDC-SIM-ART-Linux/wise
python3 -m http.server 9090
```

In terminal 2, enter in the following commands:
```
cd ~/projects/race_common/OSSDC-SIM-ART-Linux
./start_sim_local.sh
```

In terminal 3, enter in the following commands:
```
~/projects/race_common/
tmuxP loal tools/tmux_configs/svl_iac.yaml
```

Make sure rvc in launch `autonomy_core.launch.py` is commented out

In terminal 4, enter in the following commands:
```
~/projects/race_common/
make PACKAGES=autonomy_launch
source install/setup.bash
ros2 run lane_detection lane_detection_node
```

In terminal 5, enter in the following commands:
```
source install/setup.bash
ros2 run lane_detection lane_guidance_node
```

### To stop:

Hit ctrl + c in one of the tmux windows in terminal 3.

In the same tmux window, enter in
```
tmux kill-session
```

### To make changes:

In a terminal, enter in this line if you make changes to lane detection node 
```
make PACKAGES=lane_detection
```

or this line if you make changes to the lane guidance node
```
make PACKAGES=lane_guidance
```

## PID Tuning

## Final Thoughts