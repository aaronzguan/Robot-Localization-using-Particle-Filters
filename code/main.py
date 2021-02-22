'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import random


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    This version converges faster than init_particles_random
    """
    X_bar_init = np.empty((num_particles, 4))

    for i in range(num_particles):
        while 1:
            y0_vals = np.random.uniform(0, 7000)
            x0_vals = np.random.uniform(3000, 7000)
            theta0_vals = np.random.uniform(-np.pi, np.pi)
            w0_vals = 1 / num_particles

            y_map = int(y0_vals // 10)
            x_map = int(x0_vals // 10)

            if occupancy_map[y_map][x_map] == 0:
                X_bar_init[i] = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
                break

    return X_bar_init


def visualize_raycast(xt, raycast_map):
    """
    Visualize raycast on the occupancy_map given a pose xt and the precomputed raycast_map
    """
    x, y, theta_robot, _ = xt
    origin_laser_x = x + np.cos(theta_robot) * 25
    origin_laser_y = y + np.sin(theta_robot) * 25
    theta_laser = [(theta_robot - np.pi / 2 + n * np.pi/180) for n in range(180)]

    x_laser_map = int(origin_laser_x//10)
    y_laser_map = int(origin_laser_y//10)

    z_cast_x, z_cast_y = [], []
    for i, theta in enumerate(theta_laser):
        theta_deg = int(np.degrees(theta) % 360)
        z_cast = raycast_map[y_laser_map, x_laser_map, theta_deg]
        x = origin_laser_x + z_cast * np.cos(theta)
        y = origin_laser_y + z_cast * np.sin(theta)

        z_cast_x.append(x // 10)
        z_cast_y.append(y // 10)

    plt.scatter(x_laser_map, y_laser_map, c='r', marker='o')
    for i in range(len(theta_laser)):
        plt.plot([x_laser_map, z_cast_x[i]], [y_laser_map, z_cast_y[i]], color='g', linewidth=0.8)
    plt.show()

if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=1000, type=int)
    parser.add_argument('--visualize', action='store_false')
    parser.add_argument('--path_to_raycast_map', default='raycast_map.npy')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    random.seed(2)
    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if not os.path.exists(args.path_to_raycast_map):
        print("Start pre-computing the ray cast map")
        raycast_map = sensor_model.precompute_raycast()
        np.save(args.path_to_raycast_map, raycast_map)
        print('Pre-compute of ray casting done!')
    else:
        raycast_map = np.load(args.path_to_raycast_map)

    if args.visualize:
        visualize_map(occupancy_map)
        # visualize_raycast(X_bar[1], raycast_map)

    start = time.time()
    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        s = time.time()
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1, raycast_map)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)

        print("Processing each data line in {:.2f}s".format(time.time() - s))

    print("Total time: {}s".format(time.time() - start))
