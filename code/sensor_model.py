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
from tqdm import tqdm

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
        # z_hit, z_short, z_max, z_rand
        self._z_hit = 1  # 0.1 ~ 10
        self._z_short = 0.12  # 0.01 ~ 1
        self._z_max = 0.05  # 0.01 ~ 1
        self._z_rand = 800  # 10 ~ 1000

        # sigma_hit is an intrinsic noise parameter of the sensor model for measurement noise
        self._sigma_hit = 100
        # lambda_short is an intrinsic parameter of the sensor model, for exponential noise
        self._lambda_short = 0.1

        self._max_range = 8183
        self._min_probability = 0.35
        self._subsampling = 2   # ratio of down sampling

        self._offset = 25  # The laser on the robot is 25 cm offset forward from center of the robot
        self._occupancy_map = occupancy_map
        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        self._interpolation_num = 250  # The number of points interpolated during ray casting

    def beam_range_finder_model(self, z_t1_arr, x_t1, raycast_map):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[in] raycast_map : look up map for the true laser range readings zstar_t
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # Down-sample the laser reading
        z_t = np.array([z_t1_arr[i] for i in range(0, 180, self._subsampling)])
        # Get the true laser reading
        zstar_t = self.ray_casting(x_t1, raycast_map)

        prob_zt1 = self._z_hit * self.get_p_hit(z_t, zstar_t) + \
                   self._z_short * self.get_p_short(z_t, zstar_t) + \
                   self._z_max * self.get_p_max(z_t) + \
                   self._z_rand * self.get_p_rand(z_t)

        prob_zt1 = np.delete(prob_zt1, np.where(prob_zt1 == 0.0))
        prob_zt1 = np.sum(np.log(prob_zt1))

        return np.exp(prob_zt1)

    def ray_casting(self, x_t1, raycast_map):
        theta_robot = x_t1[2]
        origin_laser_x = int((x_t1[0] + self._offset * math.cos(theta_robot))//self._resolution)
        origin_laser_y = int((x_t1[1] + self._offset * math.sin(theta_robot))//self._resolution)
        # Get the down-sampled angles in radian
        theta_laser = [(theta_robot - np.pi/2 + theta * np.pi / 180) for theta in range(0, 180, self._subsampling)]
        theta_laser = (np.degrees(theta_laser) % 360).astype(int)    # convert from radian to degree
        zstar_t = raycast_map[origin_laser_y, origin_laser_x, theta_laser]
        return zstar_t

    def precompute_raycast(self):
        height, width = self._occupancy_map.shape
        raycast_map = np.zeros((height, width, 360))

        for i in tqdm(range(height * width)):
            x = i % width
            y = i // width
            # Make sure the initial pose is unoccupied
            if self._occupancy_map[y][x] != 0:
                continue

            x_map = x * self._resolution
            y_map = y * self._resolution

            zstar_t = self.ray_casting_all((x_map, y_map))

            for theta, z in zip(range(360), zstar_t):
                raycast_map[y, x, theta] = z

        return raycast_map

    def ray_casting_all(self, origin_laser):
        zstar_t = np.ones(360) * self._max_range
        dist_step = np.linspace(0, self._max_range, self._interpolation_num)

        for i in range(len(zstar_t)):
            theta_laser = i * np.pi/180
            zx_world = origin_laser[0] + dist_step * math.cos(theta_laser)
            zy_world = origin_laser[1] + dist_step * math.sin(theta_laser)

            zx = (zx_world / self._resolution).astype(int)
            zy = (zy_world / self._resolution).astype(int)

            for j in range(len(zx)):
                # Check if zx and zy are inside the map
                if 0 <= zx[j] < self._occupancy_map.shape[1] and 0 <= zy[j] < self._occupancy_map.shape[0]:
                    # Reached an obstacle
                    if self._occupancy_map[zy[j]][zx[j]] >= self._min_probability or self._occupancy_map[zy[j]][zx[j]] == -1:
                        zstar_t[i] = math.sqrt(
                            (zx_world[j] - origin_laser[0]) ** 2 + (zy_world[j] - origin_laser[1]) ** 2)
                        break
                else:
                    break

        return zstar_t

    def get_p_hit(self, z_t, zstar_t):
        eta = norm.cdf(self._max_range, loc=zstar_t, scale=self._sigma_hit) - norm.cdf(0, loc=zstar_t, scale=self._sigma_hit)
        p_hit = norm.pdf(z_t, loc=zstar_t, scale=self._sigma_hit) / eta
        p_hit[z_t > self._max_range] = 0
        p_hit[z_t < 0] = 0
        return p_hit

    def get_p_short(self, z_t, zstar_t):
        eta = np.zeros_like(z_t)
        eta[zstar_t != 0] = 1 / (1 - np.exp(-self._lambda_short * zstar_t[zstar_t != 0]))
        p_short = eta * self._lambda_short * np.exp(-self._lambda_short * z_t)
        p_short[np.where((z_t < 0) & (z_t > zstar_t))] = 0
        return p_short

    def get_p_max(self, z_t):
        return z_t == self._max_range

    def get_p_rand(self, z_t):
        p_rand = np.zeros_like(z_t)
        p_rand[np.where((z_t >= 0) & (z_t < self._max_range))] = 1 / self._max_range
        return p_rand
