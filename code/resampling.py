'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    Page 110, Table 4.4 for Low Variance Sampling
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled = []
        n = len(X_bar)

        weight = X_bar[:, -1] / sum(X_bar[:, -1])
        outcome = np.random.multinomial(n, weight)

        for i, m in enumerate(outcome):
            X_bar_resampled += [X_bar[i]] * m

        X_bar_resampled = np.array(X_bar_resampled)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled = np.zeros_like(X_bar)
        n = len(X_bar)
        r = np.random.uniform(0, 1/n)
        i, j = 0, 0

        # Normalized the weight s.t their sum is 1
        weight = X_bar[:, -1] / sum(X_bar[:, -1])
        c = weight[0]

        for m in range(n):
            U = r + m * (1/n)
            while U > c:
                i += 1
                c += weight[i]
            X_bar_resampled[j] = X_bar[i]
            j += 1

        return X_bar_resampled
