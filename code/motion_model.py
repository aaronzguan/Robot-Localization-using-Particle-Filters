'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math
from map_reader import MapReader
from matplotlib import pyplot as plt


def sample(mean, std):
    return np.random.normal(mean, std)


def wrap(angle):
    """
    Covert angle to lie in [-pi, pi]
    param[in] angle : angle value in range [-inf, inf]
    param[out] angle_rad : angle value in range [-pi, pi]
    Ref: https://stackoverflow.com/questions/27093704/converge-values-to-range-pi-pi-in-matlab-not-using-wraptopi
    """
    angle_rad = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
    return angle_rad


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
        self._alpha1 = 0.001
        self._alpha2 = 0.001
        self._alpha3 = 0.001
        self._alpha4 = 0.001

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

        true_rot1 = delta_rot1 - sample(0, self._alpha1 * delta_rot1 ** 2 + self._alpha2 * delta_trans ** 2)
        true_trans = delta_trans - sample(0, self._alpha3 * delta_trans ** 2 + self._alpha4 * delta_rot1 ** 2 + self._alpha4 * delta_rot2 ** 2)
        true_rot2 = delta_rot2 - sample(0, self._alpha1 * delta_rot2 ** 2 + self._alpha1 * delta_trans ** 2)

        x_t1 = np.zeros(x_t0.shape)
        x_t1[0] = x_t0[0] + true_trans * math.cos(x_t0[2] + true_rot1)
        x_t1[1] = x_t0[1] + true_trans * math.sin(x_t0[2] + true_rot1)
        x_t1[2] = wrap(x_t0[2] + true_rot1 + true_rot2)

        return x_t1

"""
Test the motion model only using one particle
Check if the particle follows a sensible trajectory
The odometry would cause the particle position to drift over time globally,
but locally the motion should still make sense
"""

if __name__ == "__main__":
    src_path_map = '../data/map/wean.dat'
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    src_path_log = '../data/log/robotdata1.log'
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    num_particles = 1
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))
    X_bar = np.hstack((x0_vals, y0_vals, theta0_vals)).transpose()
    print('Initial position: {}'.format(X_bar))
    xt, yt, Xt, Yt = [], [], [], []
    first_time_idx = True
    for time_idx, line in enumerate(logfile):
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
        odometry_robot = meas_vals[0:3]

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        u_t1 = odometry_robot
        X_bar = motion_model.update(u_t0, u_t1, X_bar)
        u_t0 = u_t1

        xt.append(X_bar[0])
        yt.append(X_bar[1])
        Xt.append(u_t0[0])
        Yt.append(u_t0[1])

    plt.subplot(1, 2, 1)
    plt.plot(xt, yt)
    plt.scatter(xt[0], yt[0], c='r')
    plt.subplot(1, 2, 2)
    plt.plot(Xt, Yt)
    plt.scatter(Xt[0], Yt[0], c='r')
    plt.show()