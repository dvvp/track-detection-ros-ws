import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Int32, Int32MultiArray
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from ament_index_python import get_package_share_directory
from ultralytics import YOLO
import base64
from shapely.geometry import Polygon

# Nodes in this program
NODE_NAME = 'lane_detection_node'

# Topics subcribed/published to in this program
CAMERA_TOPIC_NAME = '/oak/rgb/image_raw'
SIM_CAMERA_TOPIC_NAME = '/camera_link/image/compressed'
CENTROID_TOPIC_NAME = '/centroid'

MODEL_PATH = os.path.join(get_package_share_directory("lane_detection"), "models/best.pt")


class LaneDetection(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.centroid_error_publisher = self.create_publisher(Float32, CENTROID_TOPIC_NAME, 10)
        self.centroid_error_publisher
        self.centroid_error = Float32()
        self.camera_subscriber = self.create_subscription(Image, CAMERA_TOPIC_NAME, self.locate_centroid, 10)
        self.camera_subscriber
        self.bridge = CvBridge()
        self.sim_camera_subs = self.create_subscription(CompressedImage, SIM_CAMERA_TOPIC_NAME, self.sim_locate_centroid, 10)
        self.sim_camera_subs
        self.test_frame_pub = self.create_publisher(Image, "/test_frame_received", 10)
        self.test_frame_pub


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

            # Find centroid
            coords_xy = results[max_conf_index].masks.xy
            coords_x = coords_xy[0][:, 0]
            centroid_x = (np.min(coords_x) + np.max(coords_x)) / 2

            _, width, _ = frame.shape

            # Find horizontal center of image
            center_x = width // 2

            # Compute centroid error and publish
            error_x = centroid_x - center_x
            self.centroid_error.data = error_x
            self.centroid_error_publisher.publish(self.centroid_error)

    def sim_locate_centroid(self, data):
        # Image processing from rosparams
        frame = self.bridge.compressed_imgmsg_to_cv2(data)
        # self.test_frame_pub.publish(self.bridge.cv2_to_imgmsg(frame))

        # Find contour with the highest confidence
        results = self.model.predict(frame, conf=0.1, show=True)
        conf = results[0].boxes.conf.tolist()

        if len(conf) != 0:
            max_conf_index = np.argmax(conf)
            height, width, _ = frame.shape

            # Find centroid
            coords_xy = results[max_conf_index].masks.xy
            coords_x = coords_xy[0][:, 0]
            coords_y = coords_xy[0][:, 1]
            # filter out the points on the % bottom of the image
            filter_y_threshold_pct = 0.25
            filter_y_threshold = int((1 - filter_y_threshold_pct) * height)
            index = coords_y < filter_y_threshold
            coords_x = coords_x[index].reshape((-1, 1))
            coords_y = coords_y[index].reshape((-1, 1))
            filtered_xy = np.hstack((coords_x, coords_y))

            moments = cv2.moments(filtered_xy)
            centroid_x = moments['m10'] / (moments['m00'] + 1e-5)
            centroid_y = moments['m01'] / (moments['m00'] + 1e-5)

            # centroid_x = (np.min(coords_x) + np.max(coords_x)) / 2
            # polygon = Polygon(filtered_xy.tolist())
            # centroid = polygon.centroid
            # centroid_x = centroid.x
            # centroid_y = centroid.y

            # visulization
            cv2.circle(frame, (int(centroid_x), int(centroid_y)), 1, (255,255,0), -1)
            cv2.line(frame, (0,filter_y_threshold), (width - 1, filter_y_threshold), (255,0,255), 4)
            # cv2.polylines(frame, filtered_xy.astype(np.int32), True, (255,0,0), 1)
            self.test_frame_pub.publish(self.bridge.cv2_to_imgmsg(frame))

            # Find horizontal center of image
            center_x = width // 2

            # Compute centroid error and publish
            error_x = centroid_x - center_x
            self.centroid_error.data = error_x
            print('error: ', self.centroid_error.data, ', x: ', center_x)
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
