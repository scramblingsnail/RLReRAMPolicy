import numpy as np
import os.path as path
import yaml
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import io
import cv2

from .array import OneTOneRArray
from .memristor_env import OneTOneREnv


def load_1t1r_config(memristor_config_path=None, transistor_config_path=None,
                    simulate_config_path=None, array_config_path=None):
    current_dir = path.dirname(path.abspath(__file__))
    if memristor_config_path is None:
        memristor_config_path = r'{}/default_memristor_cfg.yaml'.format(current_dir)
    if transistor_config_path is None:
        transistor_config_path = r'{}/default_transistor_cfg.yaml'.format(current_dir)
    if simulate_config_path is None:
        simulate_config_path = r'{}/default_simulate_cfg.yaml'.format(current_dir)
    if array_config_path is None:
        array_config_path = r'{}/default_array_cfg.yaml'.format(current_dir)

    with open(memristor_config_path, 'r') as m_f:
        memristor_config = yaml.safe_load(m_f)
    with open(transistor_config_path, 'r') as t_f:
        transistor_config = yaml.safe_load(t_f)
    with open(simulate_config_path, 'r') as s_f:
        simulate_config = yaml.safe_load(s_f)
    with open(array_config_path, 'r') as a_f:
        array_config = yaml.safe_load(a_f)

    print(array_config)
    return memristor_config, transistor_config, simulate_config, array_config


def init_1t1r_array(memristor_config_path=None, transistor_config_path=None,
                    simulate_config_path=None, array_config_path=None):
    memristor_config, transistor_config, simulate_config, array_config = load_1t1r_config(
        memristor_config_path=memristor_config_path,
        transistor_config_path=transistor_config_path,
        simulate_config_path=simulate_config_path,
        array_config_path=array_config_path)
    array = OneTOneRArray(array_para_dict=array_config, memristor_para_dict=memristor_config,
                          transistor_para_dict=transistor_config, simulate_para_dict=simulate_config)
    # print(memristor_config)
    # print(transistor_config)
    # print(simulate_config)
    # print(array_config)
    return array


def build_1t1r_env(memristor_config=None, transistor_config=None, simulate_config=None, array_config=None):
    if None in [memristor_config, transistor_config, simulate_config, array_config]:
        print('Using default configs')
        memristor_config, transistor_config, simulate_config, array_config = load_1t1r_config()
    env = OneTOneREnv(array_para_dict=array_config, memristor_para_dict=memristor_config,
                      transistor_para_dict=transistor_config, simulate_para_dict=simulate_config)
    return env


