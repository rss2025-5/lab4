#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
import math

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone",
            self.relative_cone_callback, 1)

        self.parking_distance = 1 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.distance_error = 0


        self.get_logger().info("Parking Controller Initialized")


    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        distance = np.sqrt(self.relative_x**2 + self.relative_y**2)

        angle_to_cone = -np.arctan2(-self.relative_y, self.relative_x)
        drive_cmd = AckermannDriveStamped()

        #################################
        self.distance_error = distance - self.parking_distance

        self.get_logger().info("distance error: %s" %self.distance_error)
        self.get_logger().info("angle to cone: %s" %angle_to_cone)

        # helper func
        def turn(dir):
            # do the three point turn
            if dir == 'left':
                drive_cmd.drive.steering_angle = -np.pi/2
                drive_cmd.drive.speed = -0.75
            if dir == 'right':
                drive_cmd.drive.steering_angle = np.pi/2
                drive_cmd.drive.speed = -0.75

        if self.distance_error > 0.1: # if too far from cone
            # less than 90 deg turn, so turn directly towards cone
            if angle_to_cone < abs(1.5):
                drive_cmd.drive.steering_angle = angle_to_cone
                velocity = min(self.distance_error * 0.5, 0.8)
                drive_cmd.drive.speed = velocity
            else:
                if self.relative_y > 0: # on the left
                    turn('left')
                else:
                    turn('right')
        elif self.distance_error < -0.1: # if too close to clone
            if self.relative_x < 0: # don't crash into cone backwards!
                drive_cmd.drive.steering_angle = np.pi/2
                velocity = -self.distance_error * 0.5 + 0.5
                drive_cmd.drive.speed = velocity
            else:
                velocity = max(self.distance_error * 0.5, -0.5)
                drive_cmd.drive.speed = velocity - 0.3 # move back extra so now case 1
        else:
            if abs(angle_to_cone) > 0.1:
                if self.relative_y > 0: # on the left
                    turn('left')
                else:
                    turn('right')
            else: drive_cmd.drive.speed = 0.0



        #else:
            # distance is good, do:
            # Angle Check


            # Get the closest wall point, do a 90 deg right or left turn away from it

            # Move forward


            # turn the opposite direction of the above but 45 deg


            # move back


            # turn 45 deg opposite to above


            # move forward


        # drive_cmd.drive.steering_angle = angle_to_cone
        # drive_cmd.drive.speed = velocity
        #################################

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
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
