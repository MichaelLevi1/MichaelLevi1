import time
import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import rps.utilities.misc as misc

# RobotariumABC: This is an interface for the Robotarium class that
# ensures the simulator and the robots match up properly.  

# THIS FILE SHOULD NEVER BE MODIFIED OR SUBMITTED!

class RobotariumABC(ABC):

    def __init__(self, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):

        # Check user input types
        assert isinstance(number_of_robots,
                          int), "The number of robots used argument (number_of_robots) provided to create the Robotarium object must be an integer type. Recieved type %r." % type(
            number_of_robots).__name__
        assert isinstance(initial_conditions,
                          np.ndarray), "The initial conditions array argument (initial_conditions) provided to create the Robotarium object must be a numpy ndarray. Recieved type %r." % type(
            initial_conditions).__name__
        assert isinstance(show_figure,
                          bool), "The display figure window argument (show_figure) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(
            show_figure).__name__
        assert isinstance(sim_in_real_time,
                          bool), "The simulation running at 0.033s per loop (sim_real_time) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(
            show_figure).__name__

        # Check user input ranges/sizes
        assert (
                    number_of_robots >= 0 and number_of_robots <= 50), "Requested %r robots to be used when creating the Robotarium object. The deployed number of robots must be between 0 and 50." % number_of_robots
        if (initial_conditions.size > 0):
            assert initial_conditions.shape == (3,
                                                number_of_robots), "Initial conditions provided when creating the Robotarium object must of size 3xN, where N is the number of robots used. Expected a 3 x %r array but recieved a %r x %r array." % (
            number_of_robots, initial_conditions.shape[0], initial_conditions.shape[1])

        self.number_of_robots = number_of_robots
        self.show_figure = show_figure
        self.initial_conditions = initial_conditions*0.8

        # Boundary stuff -> lower left point / width / height
        self.boundaries = [-1.75, -1.75, 3.5, 3.5]

        self.file_path = None
        self.current_file_size = 0

        ##### Beginning of Modifications by Michael Leventeris 01/02/22 #####

        # Aircraft Type Variable for Display (1 for fixed-wing, 2 for Quadcopter)
        self.Aircraft_Type = 1

        # Constants
  
        self.time_step = 0.033
        self.robot_diameter = 0.15
        self.wheel_radius = 0.016
        self.base_length = 0.105
        self.robot_radius = self.robot_diameter/2

        self.max_linear_velocity = 5

        if self.Aircraft_Type == 1:
            self.max_angular_velocity = 0.01 * (self.wheel_radius / self.robot_diameter) * (self.max_linear_velocity / self.wheel_radius)
        elif self.Aircraft_Type == 2:
            self.max_angular_velocity = 100 * (self.wheel_radius / self.robot_diameter) * (self.max_linear_velocity / self.wheel_radius)

        self.max_wheel_velocity = self.max_linear_velocity / self.wheel_radius

        self.velocities = np.zeros((2, number_of_robots))
        self.poses = self.initial_conditions
        if self.initial_conditions.size == 0:
            self.poses = misc.generate_initial_conditions(self.number_of_robots, spacing=0.2, width=1.5, height=1.5)

        # Visualization
        self.figure = []
        self.axes = []

        # Chassic Patch for both
        self.chassis_patches = []

        # Fixed-Wing Wing Patches
        self.rect_r_patches = []
        self.rect_l_patches = []
        self.taper_r_patches=[]
        self.taper_l_patches=[]

        # Quadcopter Patches
        self.top_left1_patches=[]
        self.top_right1_patches=[]
        self.bottom_left1_patches=[]
        self.bottom_right1_patches=[]

        self.top_left2_patches=[]
        self.top_right2_patches=[]
        self.bottom_left2_patches=[]
        self.bottom_right2_patches=[]

        self.prop_top_left_patches = []
        self.prop_top_right_patches = []
        self.prop_bottom_left_patches = []
        self.prop_bottom_right_patches = []

        # Constants
        self.rect_height = self.robot_radius
        self.rect_width = self.robot_radius*0.3

        if (self.show_figure):
            self.figure, self.axes = plt.subplots()
            self.axes.set_axis_off()
            for i in range(number_of_robots):
                if self.Aircraft_Type == 1:  # Fixed-Wing 

                    # Creating and displaying shapes for Fixed-Wing 
                    p = patches.Ellipse(self.poses[:2, i], self.rect_height, self.rect_width, math.degrees(self.poses[2,i]), facecolor='#2ECC71')
                    rect_l = patches.Rectangle(self.poses[:2, i], self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), facecolor='#2ECC71')
                    rect_r = patches.Rectangle(self.poses[:2, i], self.rect_width, -self.rect_height, \
                                            math.degrees(self.poses[2,i]), facecolor='#2ECC71')
                    taper_l = patches.Rectangle(self.poses[:2, i], -self.rect_width*0.7, self.rect_height*0.7, \
                                            math.degrees(self.poses[2,i]), facecolor='#2ECC71')
                    taper_r = patches.Rectangle(self.poses[:2, i], -self.rect_width*0.7, -self.rect_height*0.7, \
                                            math.degrees(self.poses[2,i]), facecolor='#2ECC71')

                    self.chassis_patches.append(p)
                    self.rect_l_patches.append(rect_l)
                    self.rect_r_patches.append(rect_r)
                    self.taper_r_patches.append(taper_r)
                    self.taper_l_patches.append(taper_l)

                    self.axes.add_patch(p)
                    self.axes.add_patch(rect_l)
                    self.axes.add_patch(rect_r) 
                    self.axes.add_patch(taper_l)
                    self.axes.add_patch(taper_r)

                elif self.Aircraft_Type == 2: # Quadcopter

                    # Creating and displaying shapes for Quadcopter
                    p = patches.RegularPolygon(self.poses[:2, i], 4, 0.5*self.robot_radius, self.poses[2,i] + math.pi/4, facecolor='#1a1a1a')

                    # 'Arms' of Quadcopter
                    top_left_arm_1 = patches.Rectangle(self.poses[:2, i], 0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#3366ff')
                    top_left_arm_2 = patches.Rectangle(self.poses[:2, i], -0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#3366ff')

                    top_right_arm_1 = patches.Rectangle(self.poses[:2, i], 0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#3366ff')
                    top_right_arm_2 = patches.Rectangle(self.poses[:2, i], -0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#3366ff')

                    bot_left_arm_1 = patches.Rectangle(self.poses[:2, i], 0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#1a1a1a')
                    bot_left_arm_2 = patches.Rectangle(self.poses[:2, i], -0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#1a1a1a')

                    bot_right_arm_1 = patches.Rectangle(self.poses[:2, i], 0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#1a1a1a')
                    bot_right_arm_2 = patches.Rectangle(self.poses[:2, i], -0.5*self.rect_width, self.rect_height, \
                                            math.degrees(self.poses[2,i]), color='#1a1a1a')

                    # Propellers
                    prop_top_left = patches.Ellipse(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))\
                                                    -0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))), 0.5*self.rect_width, self.rect_height, 0,  facecolor='k')
                    prop_top_right = patches.Ellipse(self.poses[:2, i]-self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))\
                                                    +0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))), 0.5*self.rect_width, self.rect_height, 0,  facecolor='k')
                    prop_bot_right = patches.Ellipse(self.poses[:2, i]-self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))\
                                                    +0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))), 0.5*self.rect_width, self.rect_height, 0,  facecolor='k')
                    prop_bot_left = patches.Ellipse(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))\
                                                    -0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))), 0.5*self.rect_width, self.rect_height, 0,  facecolor='k')

                    self.chassis_patches.append(p)
                    self.top_left1_patches.append(top_left_arm_1)
                    self.top_left2_patches.append(top_left_arm_2)
                    self.top_right1_patches.append(top_right_arm_1)
                    self.top_right2_patches.append(top_right_arm_2)

                    self.bottom_left1_patches.append(bot_left_arm_1)
                    self.bottom_left2_patches.append(bot_left_arm_2)
                    self.bottom_right1_patches.append(bot_right_arm_1)
                    self.bottom_right2_patches.append(bot_right_arm_2)
                                
                    self.prop_top_left_patches.append(prop_top_left)
                    self.prop_bottom_right_patches.append(prop_bot_right)
                    self.prop_top_right_patches.append(prop_top_right)
                    self.prop_bottom_left_patches.append(prop_bot_left)


                    self.axes.add_patch(top_left_arm_1)
                    self.axes.add_patch(top_left_arm_2)
                    self.axes.add_patch(top_right_arm_1)
                    self.axes.add_patch(top_right_arm_2)

                    self.axes.add_patch(bot_left_arm_1)
                    self.axes.add_patch(bot_left_arm_2)
                    self.axes.add_patch(bot_right_arm_1)
                    self.axes.add_patch(bot_right_arm_2)
                    self.axes.add_patch(p)

                    self.axes.add_patch(prop_top_left)
                    self.axes.add_patch(prop_bot_right)
                    self.axes.add_patch(prop_top_right)
                    self.axes.add_patch(prop_bot_left)

        ##### Ending of Modifications by Michael Leventeris 01/02/22 #####

            # Draw arena
            self.boundary_patch = self.axes.add_patch(
                patches.Rectangle(self.boundaries[:2], self.boundaries[2], self.boundaries[3], fill=False))

            self.axes.set_xlim(self.boundaries[0] - 0.1, self.boundaries[0] + self.boundaries[2] + 0.1)
            self.axes.set_ylim(self.boundaries[1] - 0.1, self.boundaries[1] + self.boundaries[3] + 0.1)

            plt.ion()
            plt.show()

            plt.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=1.03, wspace=0, hspace=0)

    def set_velocities(self, ids, velocities):

        # Threshold linear velocities
        idxs = np.where(np.abs(velocities[0, :]) > self.max_linear_velocity)
        velocities[0, idxs] = self.max_linear_velocity * np.sign(velocities[0, idxs])

        # Threshold angular velocities
        idxs = np.where(np.abs(velocities[1, :]) > self.max_angular_velocity)
        velocities[1, idxs] = self.max_angular_velocity * np.sign(velocities[1, idxs])
        self.velocities = velocities

    @abstractmethod
    def get_poses(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    # Protected Functions
    def _threshold(self, dxu):
        dxdd = self._uni_to_diff(dxu)

        to_thresh = np.absolute(dxdd) > self.max_wheel_velocity
        dxdd[to_thresh] = self.max_wheel_velocity * np.sign(dxdd[to_thresh])

        dxu = self._diff_to_uni(dxdd)

    def _uni_to_diff(self, dxu):
        r = self.wheel_radius
        l = self.base_length
        dxdd = np.vstack((1 / (2 * r) * (2 * dxu[0, :] - l * dxu[1, :]), 1 / (2 * r) * (2 * dxu[0, :] + l * dxu[1, :])))

        return dxdd

    def _diff_to_uni(self, dxdd):
        r = self.wheel_radius
        l = self.base_length
        dxu = np.vstack((r / (2) * (dxdd[0, :] + dxdd[1, :]), r / l * (dxdd[1, :] - dxdd[0, :])))

        return dxu

    def _validate(self, errors={}):
        # This is meant to be called on every iteration of step.
        # Checks to make sure robots are operating within the bounds of reality.

        p = self.poses
        b = self.boundaries
        N = self.number_of_robots

        for i in range(N):
            x = p[0, i]
            y = p[1, i]

            if (x < b[0] or x > (b[0] + b[2]) or y < b[1] or y > (b[1] + b[3])):
                if "boundary" in errors:
                    errors["boundary"] += 1
                else:
                    errors["boundary"] = 1
                    errors["boundary_string"] = "iteration(s) robots were outside the boundaries."

        for j in range(N - 1):
            for k in range(j + 1, N):
                if (np.linalg.norm(p[:2, j] - p[:2, k]) <= self.robot_diameter):
                    if "collision" in errors:
                        errors["collision"] += 1
                    else:
                        errors["collision"] = 1
                        errors["collision_string"] = "iteration(s) where robots collided."

        dxdd = self._uni_to_diff(self.velocities)
        exceeding = np.absolute(dxdd) > self.max_wheel_velocity
        if (np.any(exceeding)):
            if "actuator" in errors:
                errors["actuator"] += 1
            else:
                errors["actuator"] = 1
                errors["actuator_string"] = "iteration(s) where the actuator limits were exceeded."

        return errors
