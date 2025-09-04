"""
Filename: team.py
Author: zhisan
Contact: 762598802@qq.com
"""
import numpy as np


class TEAM:
    r"""
    Threshold adaptive memristor model.
    A numerical model of memristor simplified from the Simmons tunneling barrier model.
    refer to
        DOI: 10.1109/TCSI.2012.2215714
        DOI: 10.1063/1.3236506

    The memristor model is governed by an internal state x. Physically, it indicates the tunneling barrier width,
    or in other words, the undoped region width.
    dynamic equation:
        dx/dt =
                k_reset * [i(t)/i_reset - 1]**alpha_reset * f_reset(x)                          (i(t) < i_reset < 0)
                0                                                                               (i_reset < i(t) < i_set)
                k_set * [i(t)/i_set - 1]**alpha_set * f_set(x)                                  (0 < i_set < i(t))
    window function:
        f_reset(x) = exp(-exp((x - x_off) / w_c))
        f_set(x) = exp(-exp(-(x - x_on) / w_c))
    i-v relationship:
        exponential
            v = i * r_on * exp(lambda / (x_off - x_on) * (x - x_on))
        linear
            v = i * r_on * exp(lambda * (x - x_on) / (x_off - x_on))

    Additions:
        by zhi-san. 762598802@qq.com.
                Due to the positive feedback mechanism, the set process is too sensitive to the current, causing a sharp
            response to external stimulus. By contrast, practically used devices are much gentler.
                To this end, an exponential decay scale depending upon state x is introduced to the derivative equation
            of set process, giving an updated equation:
                flatten_scale = exp(- delta_set * lambda * (x_off - x) / (x_off - self.x_on))
                dx/dt = k_set * [i(t)*flatten_scale/i_set - 1]**alpha_set * f_set(x)              (0 < i_set < i(t))
            in which the parameter delta_set controls the degree of flattening.
    parameters:
        k_reset, k_set: nm/s;               (k_reset > 0, k_set < 0)
        alpha_reset, alpha_set, delta_set
        i_reset, i_set: uA                  (i_reset < 0, i_set > 0);
        x_off, x_on: nm
        w_c: nm
        r_on, r_off: kohm
        lamb: ln(r_off / r_on)
        init_r: kohm
        dt: s
    """
    def __init__(self, k_reset, k_set, alpha_reset, alpha_set, delta_set, i_reset, i_set, x_off, x_on, w_c, r_on, r_off,
                 init_r):
        assert k_set < 0 < k_reset
        assert i_reset < 0 < i_set
        assert r_off > r_on
        self.k_reset, self.k_set = k_reset, k_set
        self.alpha_reset, self.alpha_set, self.delta_set = alpha_reset, alpha_set, delta_set
        self.i_reset, self.i_set = i_reset, i_set
        self.x_off, self.x_on = x_off, x_on
        self.w_c = w_c
        self.r_on, self.r_off = r_on, r_off
        self.lamb = np.log(r_off / r_on)
        self.x = self.memsistance_to_state(init_r)

    def f_reset(self, x):
        return np.exp(-np.exp((self.x_off - x) / self.w_c))

    def f_set(self, x):
        return np.exp(-np.exp(-(self.x_on - x) / self.w_c))

    def state_derivative(self, x, i):
        if i < self.i_reset:
            # print('i: ', i)
            derivative = self.k_reset * np.power(i / self.i_reset - 1, self.alpha_reset) * self.f_reset(x)
            # print('derivative: ', derivative)
        elif i > self.i_set:
            # raw derivative
            # derivative = self.k_set * np.power(i / self.i_set - 1, self.alpha_set) * self.f_set(x)

            # flattened derivative
            flatten_scale = np.exp(-self.delta_set * self.lamb * (self.x_off - x) / (self.x_off - self.x_on))
            derivative = self.k_set * np.power(i * flatten_scale / self.i_set - 1, self.alpha_set) * self.f_set(x)
        else:
            derivative = 0
        return derivative

    @property
    def r(self):
        return self.state_to_memristance(self.x)

    def set_r(self, resistance):
        r"""
        set the memristance of the memristance.

        Args:
            resistance: kOhm
        """
        assert self.r_on <= resistance <= self.r_off
        self.x = self.memsistance_to_state(resistance)

    def memsistance_to_state(self, m):
        # exponential
        x = (self.x_off - self.x_on) / self.lamb * np.log(m / self.r_on) + self.x_on
        # linear
        # x = (m - self.r_on) * (self.x_off - self.x_on) / (self.r_off - self.r_on) + self.x_on
        return x

    def state_to_memristance(self, x):
        r"""
        v: mV
        i: uA
        """
        # exponential
        m = self.r_on * np.exp(self.lamb * (x - self.x_on) / (self.x_off - self.x_on))
        # linear
        # m = self.r_on + (x - self.x_on) / (self.x_off - self.x_on) * (self.r_off - self.r_on)
        return m
