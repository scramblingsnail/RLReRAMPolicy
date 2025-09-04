"""
Filename: simulator.py
Author: zhisan
Contact: 762598802@qq.com
"""
import numpy as np
from ..numerical_models import TEAM, NMOSFETResistor
from ..solver import differential_equation_tools, difference_schemes


class BaseSimulator:
    r"""
    A basic simulator for memristive circuit.
    The positive terminal of memristor is connected to node r, while another terminal's connection depends on your idea.
    In this basic simulator, the negative terminal is connected to the node s. You can insert any other circuit elements
    between the node s and the negative terminal by overrding the function "probe", refer to the class OneTOneRSimulator
    as an example.
    """
    def __init__(self, k_reset, k_set, alpha_reset, alpha_set, delta_set, i_reset, i_set,
                 x_off, x_on, w_c, r_on, r_off, init_r,
                 max_dt, probe_dt, difference_scheme='second_runge_kutta', relative_noise_std=0):
        self.memristor = TEAM(k_reset=k_reset, k_set=k_set, alpha_reset=alpha_reset, alpha_set=alpha_set,
                              delta_set=delta_set, i_reset=i_reset, i_set=i_set, x_off=x_off, x_on=x_on, w_c=w_c,
                              r_on=r_on, r_off=r_off, init_r=init_r)
        self.max_dt = max_dt
        if probe_dt < max_dt:
            raise ValueError('Please set probe_dt no less than max_dt')
        self.probe_dt = probe_dt
        if difference_scheme not in difference_schemes:
            raise ValueError('Supported difference schemes: {}'.format(difference_schemes))
        self.calculate_dx = getattr(differential_equation_tools, difference_scheme)
        self.relative_noise_std = relative_noise_std

    def set_memristance(self, r):
        r = np.clip(r, self.memristor.r_on, self.memristor.r_off)
        self.memristor.set_r(r)

    def probe(self, x, v_s, v_r, **kwargs):
        r"""
        Here, the positive direction of current is from v_r to v_s.
        """
        r = self.memristor.state_to_memristance(x)
        i = (v_r - v_s) / r
        return i

    def state_derivative(self, x, v_s, v_r, **kwargs):
        # calculate current
        i = self.probe(x=x, v_s=v_s, v_r=v_r, **kwargs)
        # derivative
        dx_dt = self.memristor.state_derivative(x=x, i=i)
        return dx_dt

    def single_step(self, dt, v_s, v_r, probe: bool, **kwargs):
        dx = self.calculate_dx(differential_function=self.state_derivative, x=self.memristor.x,
                               dt=dt, v_s=v_s, v_r=v_r, **kwargs)
        next_x = self.memristor.x + dx
        self.memristor.x = np.clip(next_x, self.memristor.x_on, self.memristor.x_off)
        if probe:
            i = self.probe(x=self.memristor.x, v_s=v_s, v_r=v_r, **kwargs)
            return i

    def evolution(self, t, v_s, v_r, **kwargs):
        r"""
        Under constant voltages v_s, v_r, evolution for time t
        return:
            probe time, probe current.
        """
        steps = np.ceil(t / self.max_dt)
        dt = t / steps
        probe_t, probe = 0, False
        probe_t_list, probe_i_list = [], []
        for idx in range(int(steps)):
            next_t = (idx + 1) * dt
            if next_t > probe_t:
                probe_t_list.append(next_t)
                probe_t += self.probe_dt
                probe = True
            current = self.single_step(dt=dt, v_s=v_s, v_r=v_r, probe=probe, **kwargs)
            if probe:
                probe_i_list.append(current)
            probe = False
        # add noise for each writing
        noise = np.random.normal(scale=np.abs(self.relative_noise_std))
        noisy_r = self.memristor.r * (1 + noise)
        # deal with nan
        if np.isnan(noisy_r):
            print(noisy_r)
            raise ValueError('r is nan.')
        if not np.isfinite(noisy_r):
            print(noisy_r)
            raise ValueError('r is infinite.')
        noisy_r = np.clip(noisy_r, self.memristor.r_on, self.memristor.r_off)
        self.memristor.set_r(noisy_r)
        return probe_t_list, probe_i_list


class OneTOneRSimulator(BaseSimulator):
    r"""
    A simulator for 1T1R circuit.
    A NMOSFET is inserted between node s and the negative terminal of memristor.
    """
    def __init__(self,
                 k, v_th, lamb, i_s, max_iteration, eps,
                 k_reset, k_set, alpha_reset, alpha_set, delta_set, i_reset, i_set, x_off, x_on, w_c, r_on, r_off, init_r,
                 max_dt, probe_dt, difference_scheme='second_runge_kutta', relative_noise_std=0
                 ):
        super().__init__(k_reset=k_reset, k_set=k_set, alpha_reset=alpha_reset, alpha_set=alpha_set, delta_set=delta_set,
                         i_reset=i_reset, i_set=i_set, x_off=x_off, x_on=x_on, w_c=w_c, r_on=r_on, r_off=r_off, init_r=init_r,
                         max_dt=max_dt, probe_dt=probe_dt, difference_scheme=difference_scheme,
                         relative_noise_std=relative_noise_std)
        self.transistor = NMOSFETResistor(k=k, v_th=v_th, lamb=lamb, i_s=i_s, max_iteration=max_iteration, eps=eps)

    def probe(self, x, v_s, v_r, **kwargs):
        v_g = kwargs['v_g']
        r = self.memristor.state_to_memristance(x)
        v_d, i = self.transistor.calculate_i(v_g=v_g, v_s=v_s, v_r=v_r, r=r)
        return i
