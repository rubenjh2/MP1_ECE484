import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time

import matplotlib.pyplot as plt   # ADDED

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True   # CHANGE

        # ADDED STUFF
        self.accelerations = []
        self.prev_time = 0
        self.time_tracker = 0
        self.time_set = []


    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
      
        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y

        vel_x = currentPose.twist.linear.x
        vel_y = currentPose.twist.linear.y

        vel = np.sqrt(vel_x**2 + vel_y**2)

        orientation_quaternion = currentPose.pose.orientation

        _, _, yaw = quaternion_to_euler(orientation_quaternion.x, orientation_quaternion.y, orientation_quaternion.z, orientation_quaternion.w)

        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################

        # get location components of next target
        if len(future_unreached_waypoints) > 1:
            target_x1, target_y1 = future_unreached_waypoints[0]
            target_x2, target_y2 = future_unreached_waypoints[1]
        else:
            target_x1, target_y1 = future_unreached_waypoints[0]
            target_x2, target_y2 = target_x1, target_y1


        # rename for simplicity
        x1 = curr_x
        y1 = curr_y
        x2 = target_x1
        y2 = target_y1
        x3 = target_x2
        y3 = target_y2

        # compute the center of the curve's circle (x_c, y_c)
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)   # midpoint from 1 to 2
        mid2 = ((x2 + x3) / 2, (y2 + y3) / 2)   # midpoint from 2 to 3

        dx1, dy1 = x2 - x1, y2 - y1   # slope of 1 to 2
        dx2, dy2 = x3 - x2, y3 - y2   # slope of 2 to 3

        if dx1 == 0:  # Vertical line case
            perp_slope1 = 0  # Perpendicular line is horizontal
        else:
            perp_slope1 = -dx1 / dy1 if dy1 != 0 else np.inf  # Perpendicular slope

        if dx2 == 0:  # Vertical line case
            perp_slope2 = 0
        else:
            perp_slope2 = -dx2 / dy2 if dy2 != 0 else np.inf

        if np.isinf(perp_slope1):  # First bisector is vertical
            x_c = mid1[0]
            y_c = perp_slope2 * (x_c - mid2[0]) + mid2[1]
        elif np.isinf(perp_slope2):  # Second bisector is vertical
            x_c = mid2[0]
            y_c = perp_slope1 * (x_c - mid1[0]) + mid1[1]
        else:  # Normal case: Solve for intersection of the two perpendicular bisectors
            A = np.array([[-perp_slope1, 1], [-perp_slope2, 1]])
            b = np.array([mid1[1] - perp_slope1 * mid1[0], mid2[1] - perp_slope2 * mid2[0]])
            x_c, y_c = np.linalg.solve(A, b)

        R = np.sqrt((x1 - x_c) ** 2 + (y1 - y_c) ** 2)

        kappa = 1/R

        # compute target velocity with root-proportionality
        v_max = 20   # maximum velocity in any case

        slow_param = 50   # TUNE 50 works
        velocity_curve = v_max * np.exp(-kappa * slow_param)

        target_vel = np.clip(velocity_curve, 8.0, v_max)

        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_vel # target_vel


    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
                
        # get location components of next target
        if len(future_unreached_waypoints) > 1:
            target_x1, target_y1 = target_point
            target_x2, target_y2 = future_unreached_waypoints[1]
        else:
            target_x1, target_y1 = target_point
            target_x2, target_y2 = target_x1, target_y1


        # rename for simplicity
        x1 = curr_x
        y1 = curr_y
        x2 = target_x1
        y2 = target_y1
        x3 = target_x2
        y3 = target_y2

        # compute the center of the curve's circle (x_c, y_c)
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)   # midpoint from 1 to 2
        mid2 = ((x2 + x3) / 2, (y2 + y3) / 2)   # midpoint from 2 to 3

        dx1, dy1 = x2 - x1, y2 - y1   # slope of 1 to 2
        dx2, dy2 = x3 - x2, y3 - y2   # slope of 2 to 3

        if dx1 == 0:  # Vertical line case
            perp_slope1 = 0  # Perpendicular line is horizontal
        else:
            perp_slope1 = -dx1 / dy1 if dy1 != 0 else np.inf  # Perpendicular slope

        if dx2 == 0:  # Vertical line case
            perp_slope2 = 0
        else:
            perp_slope2 = -dx2 / dy2 if dy2 != 0 else np.inf

        if np.isinf(perp_slope1):  # First bisector is vertical
            x_c = mid1[0]
            y_c = perp_slope2 * (x_c - mid2[0]) + mid2[1]
        elif np.isinf(perp_slope2):  # Second bisector is vertical
            x_c = mid2[0]
            y_c = perp_slope1 * (x_c - mid1[0]) + mid1[1]
        else:  # Normal case: Solve for intersection of the two perpendicular bisectors
            A = np.array([[-perp_slope1, 1], [-perp_slope2, 1]])
            b = np.array([mid1[1] - perp_slope1 * mid1[0], mid2[1] - perp_slope2 * mid2[0]])
            x_c, y_c = np.linalg.solve(A, b)

        R = np.sqrt((x1 - x_c) ** 2 + (y1 - y_c) ** 2)

        sign_theta = np.sign((x2 - x1)*(y3 - y2) - (y2 - y1)*(x3 - x2))

        ld = 10   # m --- TUNE THIS
        theta = sign_theta * ld / R   # arc angle using arc length formula --- rads

        # compute alpha
        circ_to_car_x = (x1 - x_c)
        circ_to_car_y = (y1 - y_c)
        
        x_ld = circ_to_car_x * np.cos(theta) - circ_to_car_y * np.sin(theta)
        y_ld = circ_to_car_x * np.sin(theta) + circ_to_car_y * np.cos(theta)

        x_ld_world = x_ld + x_c
        y_ld_world = y_ld + y_c

        heading_des_x = x_ld_world - x1
        heading_des_y = y_ld_world - y1

        heading_world = np.arctan2(heading_des_y, heading_des_x)

        alpha = heading_world - curr_yaw

        # compute steering angle (delta)
        delta = np.arctan2(2*self.L*np.sin(alpha), ld)
        

        ####################### TODO: Your TASK 3 code starts Here #######################
        return delta


    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)

        # Acceleration and Time Tracking (ADDED)
        self.accelerations.append(acceleration)

        curr_time = (rospy.Time.now()).to_sec()
        dt = curr_time - self.prev_time
        self.time_tracker += dt
        self.time_set.append(self.time_tracker)
        self.prev_time = curr_time

        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

        self.prev_vel = curr_vel

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)


        # Plot code (ADDED)
        plt.figure(figsize=(8, 5))
        plt.plot(self.time_set, self.accelerations, marker="o", linestyle="-", linewidth=2, label="Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.title("Acceleration vs. Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("accel_plot.png")
