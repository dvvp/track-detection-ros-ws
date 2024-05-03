import json
import cv2
import depthai as dai
import numpy as np
import time

from yolo_api import Segment

with open("helpers/config.json", "r") as config:
    model_data = json.load(config)

preview_img_width = model_data["input_width"]
preview_img_height = model_data["input_height"]
input_shape = [1, 3, preview_img_height, preview_img_width]

output0_shape = model_data["shapes"]["output0"]
output1_shape = model_data["shapes"]["output1"]

path_to_yolo_blob = "helpers/models/best320x192v2.blob"


def main():
    pipeline = dai.Pipeline()

    # Init pipeline's output queue
    xout_rgb = pipeline.createXLinkOut()
    xout_yolo_nn = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    xout_yolo_nn.setStreamName("yolo_nn")

    # Neural network pipeline properties
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(path_to_yolo_blob)
    nn.out.link(xout_yolo_nn.input)

    # Color cam properties (Cam_A/RGB)
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(preview_img_width, preview_img_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    cam_rgb.preview.link(nn.input)
    cam_rgb.preview.link(xout_rgb.input)
    print("Color cam resolution: ", cam_rgb.getResolutionSize())

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        yolo_nn_queue = device.getOutputQueue("yolo_nn", maxSize=4, blocking=False)
        
        fps = 0
        start_time = time.time()

        while True:
            rgb_queue_msg = rgb_queue.get()
            yolo_nn_queue_msg = yolo_nn_queue.get()

            if rgb_queue_msg is not None:
                frame = rgb_queue_msg.getCvFrame()

                if yolo_nn_queue_msg is not None:
                    output0 = np.reshape(
                        yolo_nn_queue_msg.getLayerFp16("output0"),
                        newshape=(output0_shape),
                    )
                    output1 = np.reshape(
                        yolo_nn_queue_msg.getLayerFp16("output1"),
                        newshape=(output1_shape),
                    )
                    if len(output0) > 0 and len(output1) > 0:
                        yoloseg = Segment(
                            input_shape=input_shape,
                            input_height=preview_img_height,
                            input_width=preview_img_width,
                            conf_thres=0.2,
                            iou_thres=0.5,
                        )
                        yoloseg.prepare_input_for_oakd(frame.shape[:2])
                        yoloseg.segment_objects_from_oakd(output0, output1)
                        frame, _ = yoloseg.draw_masks(frame.copy())

                cv2.imshow("Output", frame)
                
                # Calculate FPS
                fps += 1
                if time.time() - start_time >= 1:
                    print("FPS:", fps)
                    fps = 0
                    start_time = time.time()

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
