'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3] Page 158, Table 6.1 for sensor model
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # Four distributions are mixed by a weighted average, defined by
        # z_hit, z_short, z_max, z_rand, with z_hit + z_short + z_max + z_rand = 1
        self._z_hit = 0.6
        self._z_short = 0.15
        self._z_max = 0.05
        self._z_rand = 0.2

        # sigma_hit is an intrinsic noise parameter of the sensor model for measurement noise
        self._sigma_hit = 2
        # lambda_short is an intrinsic parameter of the sensor model, for exponential noise
        self._lambda_short = 0.03

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 2   # ratio of down sampling

        self._offset = 25  # The laser on the robot is 25 cm offset forward from center of the robot
        self._occupancy_map = occupancy_map
        self._resolution = 10  # each cell has a 10cm resolution in x,y axes

        # self._norm_wts = 1.0

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # z_t, z_t_transform = self.ray_frame_transform(z_t1_arr, x_t1)
        z_t, zstar_t = self.ray_casting(z_t1_arr, x_t1)

        prob_zt1 = 1.0
        for i in range(len(z_t)):
            p = self._z_hit * self.get_p_hit(z_t[i], zstar_t[i]) + \
                self._z_short * self.get_p_short(z_t[i], zstar_t[i]) + \
                self._z_max * self.get_p_max(z_t[i]) + \
                self._z_rand * self.get_p_rand(z_t[i])
            prob_zt1 *= p
            # if prob_zt1 == 0:
            #     prob_zt1 = 1e-20

        return prob_zt1

    # def ray_frame_transform(self, z_t1_arr, x_t1):
    #     z_t = np.empty(shape=[len(z_t1_arr) // self._subsampling])
    #     z_t_transform = np.empty_like(z_t)
    #     theta_robot = x_t1[2]
    #     laser_origin = [x_t1[0] + self._offset * math.cos(theta_robot), x_t1[1] + self._offset * math.sin(theta_robot)]
    #
    #     for k in range(0, len(z_t1_arr), self._subsampling):
    #         # laser beam orientation in world frame
    #         theta_laser = -np.pi/2 + k
    #         # laser reading along x-axis in world frame
    #         zx_world = laser_origin[0] + z_t1_arr[k] * math.cos(theta_robot + theta_laser)
    #         # laser reading along y-axis in world frame
    #         zy_world = laser_origin[1] + z_t1_arr[k] * math.sin(theta_robot + theta_laser)
    #         # laser reading in world frame
    #         z_t_transform[k // self._subsampling] = math.sqrt(zx_world**2 + zy_world**2)
    #         # Rest laser reading after downsample
    #         z_t[k // self._subsampling] = z_t1_arr[k]
    #
    #     return z_t, z_t_transform

    def ray_casting(self, z_t1_arr, x_t1):
        z_t = np.empty(shape=[len(z_t1_arr) // self._subsampling])
        zstar_t = np.empty_like(z_t)
        theta_robot = x_t1[2]
        laser_origin = [x_t1[0] + self._offset * math.cos(theta_robot), x_t1[1] + self._offset * math.sin(theta_robot)]
        dist_step = np.linspace(0, self._max_range, 500)

        for i in range(len(zstar_t)):
            # Down-sample the laser reading
            z_t[i] = z_t1_arr[i * self._subsampling]

            theta_laser = -np.pi/2 + self._subsampling * i
            zx_world = laser_origin[0] + dist_step * math.cos(theta_robot + theta_laser)
            zy_world = laser_origin[1] + dist_step * math.sin(theta_robot + theta_laser)

            zx = (zx_world / self._resolution).astype(int)   # TODO: Should do round down or round to nearest?
            zy = (zy_world / self._resolution).astype(int)

            for j in range(len(zx)):
                # Check if zx and zy are inside the map
                if 0 <= zx[j] < self._occupancy_map.shape[1] and 0 <= zy[j] < self._occupancy_map.shape[0]:
                    if self._occupancy_map[zy[j]][zx[j]] >= self._min_probability:
                        zstar_t[i] = math.sqrt((zx_world[j] - laser_origin[0]) ** 2 + (zy_world[j] - laser_origin[1]) ** 2)

        return z_t, zstar_t

    def get_p_hit(self, z_t, zstar_t):
        p_hit = 0
        if 0 <= z_t <= self._max_range:
            p_hit = -0.5 * math.log(2 * math.pi * (self._sigma_hit ** 2)) - 0.5 * ((z_t - zstar_t) ** 2 / (
                        self._sigma_hit ** 2))
            p_hit = math.exp(p_hit)

        return p_hit

    def get_p_short(self, z_t, zstar_t):
        p_short = 0
        if 0 <= z_t <= zstar_t:
            eta = 1 / (1 - math.exp(-self._lambda_short * zstar_t))
            p_short = eta * self._lambda_short * math.exp(-self._lambda_short * z_t)

        return p_short

    def get_p_max(self, z_t):
        if z_t == self._max_range:
            return 1
        else:
            return 0

    def get_p_rand(self, z_t):
        p_rand = 0
        if 0 <= z_t < self._max_range:
            p_rand = 1 / self._max_range
        return p_rand



