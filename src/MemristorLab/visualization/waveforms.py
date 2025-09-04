"""
Filename: wavefroms.py
Author: zhisan
Contact: 762598802@qq.com
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def show_waveform(ax, durations: List[float], voltages: List[float]):
    r"""
    voltages
    durations: duration of corresponding voltage
    """
    edge_time = np.min(np.array(durations)) / 1e3
    time_list = [0]
    v_list = [0]
    for idx, duration in enumerate(durations):
        t_points = [time_list[-1] + edge_time, time_list[-1] + duration]
        v_points = [voltages[idx], voltages[idx]]
        time_list += t_points
        v_list += v_points
    ax.plot(time_list, v_list)
