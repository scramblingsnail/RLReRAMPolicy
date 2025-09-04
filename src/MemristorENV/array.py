import numpy as np
from typing import Tuple, List, Union
from ..MemristorLab import OneTOneRSimulator


def random_onetoner_cell(array_para_dict, memristor_para_dict, transistor_para_dict, simulate_para_dict):
    r"""
    memristor_relative_noise: relative noise std for params: k_reset, k_set, r_on, r_off
    transistor_relative_noise: relative noise std for params: k, v_th
    """
    # memristor params
    k_reset, k_set = memristor_para_dict['k_reset'], memristor_para_dict['k_set']
    alpha_reset, alpha_set = memristor_para_dict['alpha_reset'], memristor_para_dict['alpha_set']
    delta_set = memristor_para_dict['delta_set']
    i_reset, i_set = memristor_para_dict['i_reset'], memristor_para_dict['i_set']
    x_off, x_on = memristor_para_dict['x_off'], memristor_para_dict['x_on']
    w_c, r_on, r_off = memristor_para_dict['w_c'], memristor_para_dict['r_on'], memristor_para_dict['r_off']

    while True:
        mem_noises = np.random.normal(scale=np.abs(array_para_dict['memristor_relative_noise']), size=4)
        k_reset = k_reset * (1 + mem_noises[0])
        k_set = k_set * (1 + mem_noises[1])
        if k_set < 0 < k_reset:
            break
    # r_on = r_on * (1 + mem_noises[2])
    # r_off = r_off * (1 + mem_noises[3])
    init_g = np.random.uniform(low=array_para_dict['program_conductance_range'][0],
                               high=array_para_dict['program_conductance_range'][1])
    init_r = 1 / init_g

    # transistor params
    k, v_th, lamb = transistor_para_dict['k'], transistor_para_dict['v_th'], transistor_para_dict['lamb']
    i_s = transistor_para_dict['i_s']
    transistor_noises = np.random.normal(scale=np.abs(array_para_dict['transistor_relative_noise']), size=2)
    k = k * (1 + transistor_noises[0])
    v_th = v_th * (1 + transistor_noises[1])

    # simulate params
    max_dt, probe_dt = simulate_para_dict['max_dt'], simulate_para_dict['probe_dt']
    max_iteration, eps = simulate_para_dict['max_iteration'], simulate_para_dict['eps']
    difference_scheme = simulate_para_dict['difference_scheme']
    relative_noise_std = simulate_para_dict['relative_noise_std']

    cell = OneTOneRSimulator(k=k, v_th=v_th, lamb=lamb, i_s=i_s, max_iteration=max_iteration, eps=eps,
                             k_reset=k_reset, k_set=k_set, alpha_reset=alpha_reset, alpha_set=alpha_set,
                             delta_set=delta_set, i_reset=i_reset, i_set=i_set, x_off=x_off, x_on=x_on, w_c=w_c,
                             r_on=r_on, r_off=r_off, init_r=init_r, max_dt=max_dt, probe_dt=probe_dt,
                             difference_scheme=difference_scheme, relative_noise_std=relative_noise_std)
    return cell


class OneTOneRArray:
    r"""
    A 1T1R array.

    Args:
        array_para_dict (dict): the array-level parameter settings.
        memristor_para_dict (dict): the setting dict of memristor model.
        transistor_para_dict (dict): the setting dict of transistor model.
        simulate_para_dict (dict): the setting dict of simulation.

    refer to the docstring of class `OneTOneREnv` for detailed descriptions of these settings.
    """
    def __init__(self, array_para_dict: dict, memristor_para_dict: dict, transistor_para_dict: dict,
                 simulate_para_dict: dict):
        self.array_size = array_para_dict['array_size']
        self.array = []
        for row in range(self.array_size[0]):
            row_cells = []
            for col in range(self.array_size[1]):
                each_cell = random_onetoner_cell(array_para_dict=array_para_dict,
                                                 memristor_para_dict=memristor_para_dict,
                                                 transistor_para_dict=transistor_para_dict,
                                                 simulate_para_dict=simulate_para_dict)
                row_cells.append(each_cell)
            self.array.append(row_cells)
