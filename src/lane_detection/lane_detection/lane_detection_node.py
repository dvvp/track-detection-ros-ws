import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge
import os
from ament_index_python import get_package_share_directory
from ultralytics import YOLO
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import split

# Nodes in this program
NODE_NAME = 'lane_detection_node'

# Topics subscribed/published to in this program
CAMERA_TOPIC_NAME = '/oak/rgb/image_raw'
SIM_CAMERA_TOPIC_NAME = '/camera_link/image/compressed'
CENTROID_TOPIC_NAME = '/centroid'

MODEL_PATH = os.path.join(get_package_share_directory("lane_detection"), "models/best.pt")


class LaneDetection(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.centroid_error_publisher = self.create_publisher(Float32, CENTROID_TOPIC_NAME, 10)
        self.camera_subscriber = self.create_subscription(Image, CAMERA_TOPIC_NAME, self.locate_centroid, 10)
        self.bridge = CvBridge()

        # Initialize model
        self.model = YOLO(MODEL_PATH)

    def locate_centroid(self, data):
        # Image processing from rosparams
        frame = self.bridge.imgmsg_to_cv2(data)

        # Find contour with the highest confidence
        results = self.model.predict(frame, conf=0.7)
        conf = results[0].boxes.conf.tolist()

        if len(conf) != 0:
            max_conf_index = np.argmax(conf)
            height, width, _ = frame.shape

            # Find center line of image
            coords_xy = results[max_conf_index].masks.xy[0]
            center_x = width // 2
            center_line = LineString([(center_x, 0), (center_x, height)])

            # Split polygon down the middle
            polygon = Polygon(coords_xy)
            left_polygon, right_polygon = split(polygon, center_line).geoms

            # Determine which polygon is on the left and which is on the right
            if left_polygon.centroid.x > right_polygon.centroid.x:
                left_polygon, right_polygon = right_polygon, left_polygon

            # Publish difference between area
            centroid_error = left_polygon.area - right_polygon.area
            self.centroid_error_publisher.publish(centroid_error)

            print("Error:", centroid_error)


def main(args=None):
    rclpy.init(args=args)
    centroid_publisher = LaneDetection()
    try:
        rclpy.spin(centroid_publisher)
    except KeyboardInterrupt:
        centroid_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')

        # Kill cv2 windows and node
        cv2.destroyAllWindows()
        centroid_publisher.destroy_node()
        rclpy.shutdown()
        centroid_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()
