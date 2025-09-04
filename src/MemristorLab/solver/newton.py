"""
Filename: newton.py
Author: zhisan
Contact: 762598802@qq.com
"""
import numpy as np


def newton(eps, max_iter, init_x, left_items, left_derivative, **kwargs):
    eps = np.abs(eps)
    f_x = 1 + eps
    iteration = 0
    x = init_x
    while np.abs(f_x) > eps and iteration < max_iter:
        f_x = left_items(x=x, **kwargs)
        if np.abs(f_x) < eps:
            break
        derivative = left_derivative(x=x, **kwargs)
        x = x - f_x / derivative
        iteration += 1
    return x
