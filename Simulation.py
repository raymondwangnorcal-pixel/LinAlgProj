import numpy as np
import matplotlib.pyplot as plt

def get_payoff_matrix(resource_level):
    value = 6 + 4 * resource_level
    cost = 8 - 2 * resource_level

    CC = value / 2              # cooperator vc cooperator
    CCh = 1                     # cooperator vs cheater
    ChC = value                 # cheater vs cooperator
    ChCh = (value - cost) / 2   # cheater vs cheater

    return np.array([
        [CC, CCh],
        [ChC, ChCh]
    ], dtype=float)
