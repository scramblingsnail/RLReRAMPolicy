"""
Filename: differential_equation_tools.py
Author: zhisan
Contact: 762598802@qq.com
"""


difference_schemes = ('forward_euler', 'second_runge_kutta', 'fourth_runge_kutta')


def forward_euler(differential_function, x, dt, **kwargs):
    dx_dt = differential_function(x, **kwargs)
    dx = dt * dx_dt
    # next_x = x + dt * dx_dt
    return dx


def second_runge_kutta(differential_function, x, dt, **kwargs):
    k1 = dt * differential_function(x=x, **kwargs)
    k2 = dt * differential_function(x=x+k1, **kwargs)
    dx = (k1 + k2) / 2
    # print('k1: {}; k2: {}'.format(k1, k2))
    return dx


def fourth_runge_kutta(differential_function, x, dt, **kwargs):
    k1 = dt * differential_function(x=x, **kwargs)
    k2 = dt * differential_function(x=x+k1/2, **kwargs)
    k3 = dt * differential_function(x=x+k2/2, **kwargs)
    k4 = dt * differential_function(x=x+k3, **kwargs)
    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return dx
