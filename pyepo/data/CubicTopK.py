# budget=2  # number of items that can be picked
# num_items=50,  # number of targets to consider
# num_data = 1000

# X = torch.rand(num_data, num_items, 1) - 1
# Y = 10 * (X.pow(3) - 0.65 * X).squeeze()

#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for Shortest path problem
"""
import numpy as np


def genData(num_data,  num_items, noise_width= 0.25, seed=135):
    """
    A function to generate synthetic data and features for shortest path

    Args:
        num_data (int): number of data points
        num_items (int): number of total items
        seed (int): random seed

    Returns:
       tuple: data features (np.ndarray), costs (np.ndarray)
    """

    # set seed
    rnd = np.random.RandomState(seed)
    # numbrnda points
    n = num_data

    # x = rnd.rand(num_data, num_items, 1) - 1
    
    x = 2*rnd.rand(num_data, num_items, 1) -1
    # x_h = np.ones((num_data, 1, 1))*(-0.5)
    # x = np.concatenate ((x, x_h), axis=1)
    c = 10 * (x**3 - 0.65 * x).squeeze()


    epislon = rnd.uniform( - noise_width, noise_width, (num_data, num_items) )

    # # dimension of features
    # p = num_features
    # # dimension of the cost vector
    # d = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
    # # random matrix parameter B
    # B = rnd.binomial(1, 0.5, (d, p))
    # # feature vectors
    # x = rnd.normal(0, 1, (n, p))
    # # cost vectors
    # c = np.zeros((n, d))
    # for i in range(n):
    #     # cost without noise
    #     ci = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
    #     # rescale
    #     ci /= 3.5 ** deg
    #     # noise
    #     epislon = rnd.uniform(1 - noise_width, 1 + noise_width, d)
    #     ci *= epislon
    #     c[i, :] = ci

    return  x, c + epislon
