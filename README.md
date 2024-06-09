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

# New Project Setup

## Access Dashboard and Create New Project
- Access the Dashboard and click on **"Create New Project"**.
- Select the project type - **Instance Segmentation**.

## Upload Dataset
### Videos:
- Ensure videos are in supported format.
- Drag and drop videos or select from your local storage.
- Choose the appropriate Frame Rate (consider using lower frame rates on ambiguous frames, such as corners).

### Images:
- Ensure images are in supported format.
- Drag and drop images or select from your local storage.

### Annotated Images:
- Upload in the appropriate format.
- Assign images for labeling.

## Annotate Images
- Use Roboflow’s built-in annotation tool provided to label the regions of interest.

### Annotation Instructions:
- **Objective:** We want to annotate the track all the way up to the solid/dashed line and the cones, and exclude the pit.
- Use the **“Smart Polygon”** tool and click on the part of the track that is directly in front of the Go-Kart.
- If you are annotating the corner that includes the pit (this corner), ignore steps 2-4 and manually label using the **“Polygon Tool”** instead. Also, look at the second bullet point under **“Tips”**.
- Click only on the left and right of the track for a few clicks to get more of the track. (Do not click further to the front)
- Once you’ve finished the annotations, go to the drop-down menu on the upper left, select the **“track”** class, and hit **“enter”**.
- The vertices should now pop up. If the track has not been fully selected, drag the vertices to cover the whole track (Click on lines to create more vertices and click on existing vertices to remove them).
- When you finish a batch, @ me on Discord to let me know. **DO NOT ADD IT TO THE DATASET**.

### Reminders:
- Always select the **“track”** class from the drop-down menu instead of manually typing them to avoid making accidental new classes (Because of typos).
- Review previous annotations to make sure every frame is annotated properly (Happened to me a couple of times when my wifi connection was weak and it didn’t save).
- Make sure not to make multiple annotations on the same image (Go to the **“layers”** tab to double-check).

### Tips:
- Turn on hardware acceleration on your browser to make Roboflow smoother to navigate (If you are using Safari, I think this is automatically enabled).
- Clicking **“Repeat Previous”** and adjusting the vertices can save time (Thanks Minh).

## Generate New Versions
- Choose appropriate train/test split.
- Preprocess the Dataset.
- Roboflow offers preprocessing options such as resizing, augmenting.
- Choose the appropriate preprocessing options as needed.
- Generate a Version.
- Once preprocessing is set, click on **"Create"** to create a version of your dataset.
- This version will be used for training your models.

## Train New Models
### Train with Roboflow (limited available credits)
- Click **"Train"** and configure training settings, such as:
  - Model type (e.g., YOLOv5, Faster R-CNN).
  - Hyperparameters (e.g., number of epochs, learning rate).
  - Compute resources (e.g., number of GPUs).
- Start the training process.

### Train on YOLOv5 Model on Colab Notebook
- Follow instructions of the following link to train and download weights (.pt file) of the model trained: [YOLOv5 Training on Colab Notebook](https://colab.research.google.com/drive/1JTz7kpmHsg-5qwVz2d2IH3AaenI1tv0N?usp=sharing#scrollTo=Y6DFBei-tpoJ).
