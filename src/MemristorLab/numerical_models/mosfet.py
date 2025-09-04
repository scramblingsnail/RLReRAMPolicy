"""
Filename: mosfet.py
Author: zhisan
Contact: 762598802@qq.com
"""
import numpy as np
from ..solver import newton


class NMOSFET:
    r"""
    A numerical model for NMOSFET, in which the source is connected to the bulk, resulting a PN junction between source
    and drain.
    Second-order effect, specifically, channel-length modulation is taken into consideration.

    IV relationship:
        denote (u_n * C_ox * W / L) as a single parameter k.
        I = k * [(Vgs - Vth) * Vds - Vds**2 / 2]                                                (0 <= Vds <= Vgs - Vth)
        I = 1/2 * k * (Vgs - Vth)**2 * (1 + lambda * Vds) - 1/2 * k * lambda * (Vgs - Vth)**3   (Vds > Vgs - Vth)
        I = k * [(Vgs - Vth) * Vds + Vds**2 / 2]                                                (0 < - Vds <= Vgs - Vth)
        I = 1/2 * k * (Vgs - Vth)**2 * (-1 + lambda * Vds) + 1/2 * k * lambda * (Vgs - Vth)**3  (- Vds > Vgs - Vth)
        for simplicity:
        I = 0                 (Vgs < Vth)


    PN junction:
        I = - i_s * (exp(e * Vsd / (kT)) - 1)               (Vds < 0)
        I = 0                                               (Vds > 0)
    denote (e/kT) as a single parameter z, z = 0.03868174 e3.

    parameters:
        k: uA/(mV)**2
        v_th: mV
        lamb: 1/mV
        i_s: uA
    """
    def __init__(self, k, v_th, lamb, i_s):
        self.k = k
        self.v_th = v_th
        self.lamb = lamb
        self.i_s = i_s
        self.z = 0.03868174
        self.eps = 1e-8

    def vi(self, v_d, v_s, v_g):
        r"""
        v_d, v_s, v_g: mV
        return:
            i: uA; positive direction: from drain to source
        """
        v_ds = v_d - v_s
        v_gs = v_g - v_s
        if v_gs < self.v_th:
            i_ds = 0
        elif 0 < - v_ds <= v_gs - self.v_th:
            i_ds = self.k * ((v_gs - self.v_th) * v_ds + np.power(v_ds, 2) / 2)
        elif - v_ds > v_gs - self.v_th:
            i_ds = self.k * np.power((v_gs - self.v_th), 2) * (
                        -1 + self.lamb * v_ds) / 2 + self.k * self.lamb * np.power((v_gs - self.v_th), 3) / 2
        elif 0 <= v_ds <= v_gs - self.v_th:
            i_ds = self.k * ((v_gs - self.v_th) * v_ds - np.power(v_ds, 2) / 2)
        else:
            i_ds = self.k * np.power((v_gs - self.v_th), 2) * (1 + self.lamb * v_ds) / 2 - self.k * self.lamb * np.power((v_gs - self.v_th), 3) / 2

        if v_ds < 0:
            i_pn = - self.i_s * (np.exp(-self.z * v_ds) - 1)
            # i_pn = 0
        else:
            i_pn = 0
        return i_ds + i_pn

    def di_dvd(self, v_d, v_s, v_g):
        r"""
        v_d, v_s, v_g: mV
        return:
            di/dv_d: uA / mV
        """
        v_ds = v_d - v_s
        v_gs = v_g - v_s
        if v_gs < self.v_th:
            di_dv_d = 0
        elif 0 < - v_ds <= v_gs - self.v_th:
            di_dv_d = self.i_s * self.z * np.exp(- self.z * v_ds) + self.k * (v_gs - self.v_th + v_ds)
        elif - v_ds > v_gs - self.v_th:
            di_dv_d = self.i_s * self.z * np.exp(- self.z * v_ds) + 1/2 * self.k * self.lamb * np.power(v_gs - self.v_th, 2)
        elif 0 <= v_ds <= v_gs - self.v_th:
            di_dv_d = self.k * (v_gs - self.v_th - v_ds)
        else:
            di_dv_d = 1/2 * self.k * self.lamb * np.power(v_gs - self.v_th, 2)
        return di_dv_d


class NMOSFETResistor(NMOSFET):
    r"""
    A NMOSFET series-connected with a resistor.
    schematic:
                    gate
        source -----    ----- drain ----- resistor ----- res_node

    Given Vg, Vs, Vr, R, calculate the current i.
        Vg: voltage of gate
        Vs: voltage of source
        Vr: voltage of res_node
        R: resistance of the resistor

    The class is used to calculate one linear discrete time step during analyzing the dynamic response of 1T1R
    to the external stimulus.
        Solving the equation:
            I_MOS(Vd, Vg, Vs) - I_R(Vd, Vr, R) = f(Vd) = 0
        Preparations before Newton method:
            f(Vd) = I_MOS(Vd, Vg, Vs) + (Vd - Vr) / R
            derivative: d_I_MOS / d_Vd + 1 / R

    Parameters:
        k: uA/(mV)**2
        v_th: mV
        lamb: 1/mV
        i_s: uA
        max_iteration
        eps: uA/mV
    """
    def __init__(self, k, v_th, lamb, i_s, max_iteration, eps):
        super().__init__(k, v_th, lamb, i_s)
        self.max_iter = max_iteration
        self.eps = eps

    def left_items(self, x, v_g, v_s, v_r, r, **kwargs):
        r"""
        v_d, v_g, v_s, v_r: mV
        r: kohm
        """
        left = self.vi(v_d=x, v_s=v_s, v_g=v_g) + (x - v_r) / (r + self.eps)
        return left

    def left_derivative(self, x, v_g, v_s, r, **kwargs):
        r"""
        v_d, v_g, v_s, v_r: mV
        r: kohm
        """
        derivative = self.di_dvd(v_d=x, v_s=v_s, v_g=v_g) + 1 / (r + self.eps)
        return derivative

    def initiate_vd(self, v_g, v_s, v_r, r):
        r"""
        initiate vd to the cross point of the piecewise linear approximation of MOSFET's i-v_ds curve
        and the Resistor's i-v_ds curve.
        """
        v_gs = v_g - v_s
        override_v = v_gs - self.v_th
        coe = 1/2 * self.k * r * np.power(override_v, 2)
        v_ds1 = (v_r + v_s) / (1/2 * self.k * r * override_v + 1)
        v_ds2 = (v_r + v_s + coe * (override_v * self.lamb - 1)) / (coe * self.lamb + 1)
        init_vd = np.max((v_ds1, v_ds2)) + v_s
        return init_vd

    def calculate_i(self, v_g, v_s, v_r, r):
        v_d = self.initiate_vd(v_g, v_s, v_r, r)
        v_d = newton(eps=self.eps, max_iter=self.max_iter, init_x=v_d, left_items=self.left_items,
                     left_derivative=self.left_derivative, v_g=v_g, v_s=v_s, v_r=v_r, r=r)
        current = (v_r - v_d) / (r + self.eps)
        return v_d, current
