Depend on ROS2, see [here](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html) for instructions to install ROS2 Iron. Remember to install the dev tools for ROS2.

Assuming ROS2 is sourced in your terminal

To build:
```
# clone the repo
cd track-detection-ros-ws
colcon build
```

To run:
```
source install/setup.bash
ros2 run lane_detection lane_detection_node

# Track Detection Using Deep Learning
*By Devin Pham and Grace Chen*

## Introduction
During the Winter and Spring quarters at UCSD in 2024, we were given the special opportunity to develop a track detection algorithm to be used in the Autonomous Karting Series (AKS), a competition in which a number of universities came together to race autonomous electric go-karts. Due to mechanical issues that we had with our go-kart, we were unable to test this algorithm and instead used Donkeycar. Unfortunately, the Donkeycar algorithm fell short of our expectations as the go-kart was unable to complete the entire track during the race. Because the algorithm we developed was able to make the go-kart complete the track without PID tuning on a simulator, we believe that a guide detailing the machine learning pipeline that we used would be of value to students who will be competing in an autonomous race like AKS or are simply interested in learning about deep learning and computer vision. To ensure that future teams have a solid foundation to build upon our work, we made an effort to make this guide as comprehensive as possible.

## Objectives 
- Create an algorithm that is able to detect the Purdue Grand Prix Track using a camera mounted on the go-kart. This algorithm must be able to run directly on the camera itself or an NVIDIA Jetson module.
- Integrate algorithm with ROS driving controls to make the go-kart drive reactively (without pre-mapping).

## Resources Provided
- **Luxonis Oak-D LR**
  - The Luxonis Oak-D LR, with “LR'' standing for “long range”, is the main camera affixed to the front of our go-kart. It uses the left and right lenses for depth perception, while the middle lens functions as a color camera. For our purpose, we only used the color camera with less computational power. 
  - The full list of specifications for this camera can be found [here](https://docs.luxonis.com/en/latest/pages/hardware/#oak-d-lr). The main details to note about the color camera is that it has a 16:10 aspect ratio, 1200P resolution, and a maximum FPS of 30. In order to run our model on this camera at the maximum FPS, we had to lower our resolution to something like 200P. We will provide more details to this augmentation later in the guide. 

- **Oak-D Pro / Oak-D Lite**
  - These two cameras were used primarily for prototyping purposes for when we were unable to test using the LR. Two Oak-D Pros could potentially be pointed to the left and right of the go-kart for additional inference; This was what the Donkeycar algorithm used during our race.

- **NVIDIA Jetson AGX Xavier**
  - We intended on deploying our model using this computer module instead of running it directly on the camera. The Jetson’s GPU and specialized hardware accelerators allows it to make much faster inferences than running on a CPU. We did not get a chance to deploy our ROS package, but you can find instructions on how to do that [here](https://github.com/ros2/ros2/wiki/Linux-Development-Setup#supported-targets).

- **LIDAR**
  - We did not utilize LIDAR in our algorithm at all, but incorporating it could potentially enhance performance, considering its widespread use in today’s autonomous vehicles.

- **OSSDC SIM**
  - OSSDC SIM (Open Source Self Driving Car Simulator) is an open-source platform used to simulate the driving of autonomous cars and go-karts. Using a simulator is a cost-effective, safe, and scalable way to test autonomous vehicles that allows for rapid prototyping.

## Unsuccessful Attempts
Most of our failed attempts were from trying to use past repositories that specialized in lane detection. Most of these models used traditional computer vision methods using OpenCV, an open-source computer vision and machine learning software library, which took a lot of effort to tweak as most of the code available online were outdated and/or had parameters that were difficult to adjust. 
Something that we overlooked was that our model needed to differentiate between road and grass instead of lane and road. Most of the models we saw online differentiated between lane and road instead of road and grass (Note the distinction between lane detection and road/track detection). 
Using traditional computer vision methods also made track detections less robust in direct sunlight which is the main reason why we switched over to use deep learning instead of purely using computer vision techniques. Our race took place at around noon time so direct sunlight was a big issue.

## Introduction to Roboflow
Roboflow is a platform that allows developers to easily annotate images and deploy models for computer vision applications. If you are new to computer vision, annotation is the practice of labeling training data, which in our case, are images. Annotations can be in the form of bounding boxes (in the context of object detection), or masks (in the context of segmentation). This is usually a manual process, but it can be made faster through the use of pre-trained models and LLMs.
For track detection, we will be focusing on segmentation instead of object detection. Segmentation is different from object detection in that we are not only identifying objects within an image, but we are also detecting the exact boundaries of each object at the pixel level. 
It is also important that we differentiate between the two types of segmentation: instance segmentation and semantic segmentation. Instance segmentation is able to distinguish between separate instances of the same class, while semantic segmentation does not. Although it is more computationally expensive, we will be using instance segmentation since it offers support for a lot more models.

![Semantic Segmentation vs. Instance Segmentation](https://www.folio3.ai/blog/wp-content/uploads/2023/05/SS.png)

**Source:** [Folio3 AI - Semantic Segmentation vs Instance Segmentation](https://www.folio3.ai/blog/semantic-segmentation-vs-instance-segmentation/)

Roboflow offers many options for model generation either directly through their platform or through Google Colab/Jupyter Notebooks. The default model that Roboflow offers at the time this guide is written is called “Roboflow 3.0 Instance Segmentation”, and it is available directly through their platform. Because there are no details provided about this model or options to customize this model, we will be using the YOLO models available via their provided notebooks. YOLO (You only look once) is a state-of-the-art deep learning algorithm that enables real-time object detection and segmentation in images or videos. 
