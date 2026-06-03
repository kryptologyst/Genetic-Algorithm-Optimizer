import numpy as np
from loguru import logger


def rastrigin_function(x):
    return - (10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x))


def sphere_function(x):
    return -sum(xi**2 for xi in x)


def get_optimization_problem(name="rastrigin"):
    if name == "rastrigin":
        bounds = [(-5.12, 5.12)] * 5
        return rastrigin_function, bounds
    else:
        bounds = [(-10, 10)] * 5
        return sphere_function, bounds
