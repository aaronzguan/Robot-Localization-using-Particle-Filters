'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


def sample(mean, std):
    return np.random.normal(mean, std)


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5] Page 136, Table 5.6 for motion model
    """

    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.1
        self._alpha2 = 0.1
        self._alpha3 = 0.3
        self._alpha4 = 0.3

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        delta_rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = math.sqrt((u_t0[0] - u_t1[0]) ** 2 + (u_t0[1] - u_t1[1]) ** 2)
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        true_rot1 = delta_rot1 - sample(0, math.sqrt(self._alpha1 * delta_rot1 ** 2 + self._alpha2 * delta_trans ** 2))
        true_trans = delta_trans - sample(0, math.sqrt(
            self._alpha3 * delta_trans ** 2 + self._alpha4 * delta_rot1 ** 2 + self._alpha4 * delta_rot2 ** 2))
        true_rot2 = delta_rot2 - sample(0, math.sqrt(self._alpha1 * delta_rot2 ** 2 + self._alpha1 * delta_trans ** 2))

        x_t1 = np.zeros(x_t0.shape)
        x_t1[0] = x_t0[0] + true_trans * math.cos(x_t0[2] + true_rot1)
        x_t1[1] = x_t0[1] + true_trans * math.sin(x_t0[2] + true_rot1)
        x_t1[2] = x_t0[2] + true_rot1 + true_rot2

        return x_t1
