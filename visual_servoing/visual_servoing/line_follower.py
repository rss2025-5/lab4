#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
import math

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class LineFollower(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("line_follower")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone",
            self.relative_cone_callback, 1)

        self.following_distance = 0.1
        self.relative_x = 0
        self.relative_y = 0
        self.distance_error = 0
        self.wheelbase = 0.33


        self.get_logger().info("Line Follower Initialized")


    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos - 0.02
        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)

        angle_to_line= -np.arctan2(-self.relative_y, self.relative_x)
        drive_cmd = AckermannDriveStamped()

        #################################
        self.distance_error = distance - self.following_distance

        self.get_logger().info("distance error: %s" %self.distance_error)
        self.get_logger().info("angle to cone: %s" %angle_to_line)

        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(angle_to_line), distance)
        velocity = min(self.distance_error*2, 0.8)

        drive_cmd.drive.steering_angle = steering_angle
        drive_cmd.drive.speed = velocity  

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = self.distance_error
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        #################################

        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = LineFollower()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
