import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32, Int32MultiArray
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from roboflow import Roboflow
import base64

# Nodes in this program
NODE_NAME = 'lane_detection_node'

# Topics subcribed/published to in this program
CAMERA_TOPIC_NAME = '/oak/rgb/image_raw'
CENTROID_TOPIC_NAME = '/centroid'


class LaneDetection(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.centroid_error_publisher = self.create_publisher(Float32, CENTROID_TOPIC_NAME, 10)
        self.centroid_error_publisher
        self.centroid_error = Float32()
        self.camera_subscriber = self.create_subscription(Image, CAMERA_TOPIC_NAME, self.locate_centroid, 10)
        self.camera_subscriber

        # Initialize Roboflow
        self.rf = Roboflow(api_key="XG3i4cX7XdFeVFrfNqy5")
        self.project = self.rf.workspace().project("dsc190-road-detection")
        self.model = self.project.version("11").model

    def locate_centroid(self, data):
        # Image processing from rosparams
        frame = self.bridge.imgmsg_to_cv2(data)
        _, jpg_image = cv2.imencode('.jpg', frame)
        jpg_bytes = jpg_image.tobytes()

        # Load mask
        prediction = self.model.predict(jpg_bytes)
        prediction_json = prediction.json()
        mask = prediction_json['predictions'][0]['segmentation_mask']

        # Decode base64 string to binary
        mask_bytes = base64.b64decode(mask)
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)

        # Decode the mask image
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

        # Convert to binary mask
        binary_mask = (mask[:, :, 0] == 1).astype(np.uint8)

        # Compute moments to find centroid
        moments = cv2.moments(binary_mask)
        centroid_x = int(moments['m10'] / moments['m00'])

        # Find image center
        _, width, _ = jpg_image.shape
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