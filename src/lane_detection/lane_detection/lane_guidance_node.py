import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

NODE_NAME = 'lane_guidance_node'
CENTROID_TOPIC_NAME = '/centroid'
ACTUATOR_TOPIC_NAME = '/auto/raw_command'


class PathPlanner(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.twist_publisher = self.create_publisher(Twist, ACTUATOR_TOPIC_NAME, 10)
        self.twist_cmd = Twist()
        self.centroid_subscriber = self.create_subscription(Float32, CENTROID_TOPIC_NAME, self.controller, 10)
        self.centroid_subscriber

        # Default actuator values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('Kp_steering', 1),
                ('Ki_steering', 0),
                ('Kd_steering', 0),
                ('error_threshold', 0.15),
                ('zero_throttle',0.0),
                ('max_throttle', 0.2),
                ('min_throttle', 0.1),
                ('max_right_steering', 1.0),
                ('max_left_steering', -1.0)
            ])
        self.Kp = self.get_parameter('Kp_steering').value  # between [0,1]
        self.Ki = self.get_parameter('Ki_steering').value  # between [0,1]
        self.Kd = self.get_parameter('Kd_steering').value  # between [0,1]
        self.error_threshold = self.get_parameter('error_threshold').value  # between [0,1]
        self.zero_throttle = self.get_parameter('zero_throttle').value  # between [-1,1] but should be around 0
        self.max_throttle = self.get_parameter('max_throttle').value  # between [-1,1]
        self.min_throttle = self.get_parameter('min_throttle').value  # between [-1,1]
        self.max_right_steering = self.get_parameter('max_right_steering').value  # between [-1,1]
        self.max_left_steering = self.get_parameter('max_left_steering').value  # between [-1,1]

        # initializing PID control
        self.Ts = float(1 / 20)
        self.ek = 0  # current error
        self.ek_1 = 0  # previous error
        self.proportional_error = 0  # proportional error term for steering
        self.derivative_error = 0  # derivative error term for steering
        self.integral_error = 0  # integral error term for steering
        self.integral_max = 1E-8

        self.get_logger().info(
            f'\nKp_steering: {self.Kp}'
            f'\nKi_steering: {self.Ki}'
            f'\nKd_steering: {self.Kd}'
            f'\nerror_threshold: {self.error_threshold}'
            f'\nzero_throttle: {self.zero_throttle}'
            f'\nmax_throttle: {self.max_throttle}'
            f'\nmin_throttle: {self.min_throttle}'
            f'\nmax_right_steering: {self.max_right_steering}'
            f'\nmax_left_steering: {self.max_left_steering}'
        )

    def controller(self, data):
        # setting up PID control
        self.ek = data.data

        # Throttle gain scheduling (function of error)
        throttle_float = self.zero_throttle

        # Steering based on error
        steering_float = self.ek * (self.max_left_steering - self.max_right_steering)

        # Publish values
        try:
            # publish control signals
            self.twist_cmd.angular.z = steering_float
            self.twist_cmd.linear.x = throttle_float
            self.twist_publisher.publish(self.twist_cmd)

            # shift current time and error values to previous values
            self.ek_1 = self.ek

        except KeyboardInterrupt:
            self.twist_cmd.linear.x = self.zero_throttle
            self.twist_publisher.publish(self.twist_cmd)

    def clamp(self, value, upper_bound, lower_bound=None):
        if lower_bound == None:
            lower_bound = -upper_bound  # making lower bound symmetric about zero
        if value < lower_bound:
            value_c = lower_bound
        elif value > upper_bound:
            value_c = upper_bound
        else:
            value_c = value
        return value_c


def main(args=None):
    rclpy.init(args=args)
    path_planner_publisher = PathPlanner()
    try:
        rclpy.spin(path_planner_publisher)
        path_planner_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        path_planner_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
        path_planner_publisher.twist_cmd.linear.x = path_planner_publisher.zero_throttle
        path_planner_publisher.twist_publisher.publish(path_planner_publisher.twist_cmd)
        time.sleep(1)
        path_planner_publisher.destroy_node()
        rclpy.shutdown()
        path_planner_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()