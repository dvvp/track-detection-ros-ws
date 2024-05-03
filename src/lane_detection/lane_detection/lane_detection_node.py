import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32, Int32MultiArray
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from ament_index_python import get_package_share_directory
from ultralytics import YOLO
import base64

# Nodes in this program
NODE_NAME = 'lane_detection_node'

# Topics subcribed/published to in this program
CAMERA_TOPIC_NAME = '/oak/rgb/image_raw'
CENTROID_TOPIC_NAME = '/centroid'

MODEL_PATH = os.path.join(get_package_share_directory("lane_detection"), "models/best320x192v2.pt")


class LaneDetection(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.centroid_error_publisher = self.create_publisher(Float32, CENTROID_TOPIC_NAME, 10)
        self.centroid_error_publisher
        self.centroid_error = Float32()
        self.camera_subscriber = self.create_subscription(Image, CAMERA_TOPIC_NAME, self.locate_centroid, 10)
        self.camera_subscriber
        self.bridge = CvBridge()

        # Initialize model
        self.model = YOLO(MODEL_PATH)


    def locate_centroid(self, data):
        # Image processing from rosparams
        frame = self.bridge.imgmsg_to_cv2(data)

        # Find contour with the highest confidence
        results = self.model.predict(frame, conf=0.9)
        conf = results[0].boxes.conf.tolist()
        if len(conf) != 0:
            max_conf_index = np.argmax(conf)

            # Find centroid
            coords_xy = results[max_conf_index].masks.xy
            coords_x = coords_xy[0][:, 0]
            centroid_x = np.mean(coords_x)

            # Find horizontal center of image
            _, width, _ = frame.shape
            center_x = width // 2

            # Compute centroid error and publish
            error_x = centroid_x - center_x
            self.centroid_error.data = error_x
            self.centroid_error_publisher.publish(self.centroid_error)

def main(args=None):
    rclpy.init(args=args)
    centroid_publisher = LaneDetection()
    try:
        rclpy.spin(centroid_publisher)
        centroid_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        centroid_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')

        # Kill cv2 windows and node
        cv2.destroyAllWindows()
        centroid_publisher.destroy_node()
        rclpy.shutdown()
        centroid_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()