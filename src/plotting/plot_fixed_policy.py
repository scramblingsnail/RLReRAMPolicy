import json
import os.path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Union


root = Path(__file__)
for idx in range(3):
    root = root.parent


def plot_3d_array(
        array_3d: np.ndarray,
        delta_vg_list: np.ndarray,
        delta_reset_vds_list: np.ndarray,
        delta_set_vds_list: np.ndarray,
        val_max: float,
        val_min: float,
        image_path: str
):
    r"""


    Args:
        array_3d (np.ndarray): dims - (vg, reset_vds, set_vds); val: reward or success_rate or program_step
        delta_vg_list (np.ndarray):
        delta_reset_vds_list (np.ndarray):
        delta_set_vds_list (np.ndarray):
        val_max (float):
        val_min (float):
        image_path (str):

    Returns:

    """
    blue_array = np.array([19, 164, 192, 255]) / 255
    red_array = np.array([234, 96, 142, 255]) / 255

    clist = [
        (0, red_array),
        (0.5, np.array([255, 255, 255, 255]) / 255),
        (1, blue_array),
    ]
    newcmp = LinearSegmentedColormap.from_list('slight_pair', clist)

    fig = plt.figure(figsize=(9, 6))
    ax3d = fig.add_axes(Axes3D(fig))
    ax3d.xaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": "--", "color": "k"})
    ax3d.yaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": "--", "color": "k"})
    ax3d.zaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": "--", "color": "k"})

    ax3d.xaxis.set_pane_color((0, 0, 0, 0))
    ax3d.yaxis.set_pane_color((0, 0, 0, 0))
    ax3d.zaxis.set_pane_color((0, 0, 0, 0))

    ax3d.set_xlim(np.min(delta_vg_list), np.max(delta_vg_list))
    ax3d.set_ylim(np.min(delta_reset_vds_list), np.max(delta_reset_vds_list))
    ax3d.set_zlim3d(np.min(delta_set_vds_list), np.max(delta_set_vds_list))

    ax3d.set_xlabel("Delta Vg (V)")
    ax3d.set_ylabel("Delta Vds_reset (V)")
    ax3d.set_zlabel("Delta Vds_set (V)")

    x_data, y_data, z_data, val_data = [], [], [], []

    for vg_idx in range(delta_vg_list.shape[0]):
        for reset_vds_idx in range(delta_reset_vds_list.shape[0]):
            for set_vds_idx in range(delta_set_vds_list.shape[0]):
                x_data.append(delta_vg_list[vg_idx])
                y_data.append(delta_reset_vds_list[reset_vds_idx])
                z_data.append(delta_set_vds_list[set_vds_idx])
                val_data.append(array_3d[vg_idx][reset_vds_idx][set_vds_idx])

    sizes = 20

    scatter_img = ax3d.scatter(
        xs=x_data, ys=y_data, zs=z_data, s=sizes, c=val_data, vmin=val_min, vmax=val_max, cmap=newcmp, rasterized=True,
    )
    ax3d.view_init(elev=30, azim=135)

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])
    cb = plt.colorbar(scatter_img, cax=cax)
    # plt.show()
    plt.savefig(image_path, dpi=300)


def _load_fixed_policy_data(bit_width: int, noise: int):
    data_dir = str(root / r"extra_data")

    vg_data_path = fr"{data_dir}/delta_vg_bit-{bit_width}_noise-{noise}.npy"
    set_vds_data_path = fr"{data_dir}/delta_set_vds_bit-{bit_width}_noise-{noise}.npy"
    reset_vds_data_path = fr"{data_dir}/delta_reset_vds_bit-{bit_width}_noise-{noise}.npy"
    reward_data_path = fr"{data_dir}/average_rewards_bit-{bit_width}_noise-{noise}.npy"
    step_data_path = fr"{data_dir}/average_steps_bit-{bit_width}_noise-{noise}.npy"
    success_step_data_path = fr"{data_dir}/average_success_steps_bit-{bit_width}_noise-{noise}.npy"
    success_rate_data_path = fr"{data_dir}/success_rates_bit-{bit_width}_noise-{noise}.npy"

    delta_vg_list = np.load(vg_data_path)
    delta_set_vds_list = np.load(set_vds_data_path)
    delta_reset_vds_list = np.load(reset_vds_data_path)
    avg_rewards_array = np.load(reward_data_path)
    avg_steps_array = np.load(step_data_path)
    avg_success_steps_array = np.load(success_step_data_path)
    success_rate_array = np.load(success_rate_data_path)

    return (
        delta_vg_list, delta_set_vds_list, delta_reset_vds_list,
        avg_rewards_array, avg_steps_array, avg_success_steps_array, success_rate_array
    )


def _find_max_reward_step_success_rate(
        bit_width: int,
        noise: int,
):
    (
        delta_vg_list, delta_set_vds_list, delta_reset_vds_list,
        avg_rewards_array, avg_steps_array, avg_success_steps_array, success_rate_array
    ) = _load_fixed_policy_data(bit_width=bit_width, noise=noise)

    print(delta_vg_list)
    print(delta_set_vds_list)
    print(delta_reset_vds_list)

    max_r = -1e3
    max_x, max_y, max_z = 0, 0 , 0
    for i in range(avg_rewards_array.shape[0]):
        for j in range(avg_rewards_array.shape[1]):
            for k in range(avg_rewards_array.shape[2]):
                if avg_rewards_array[i][j][k] > max_r:
                    max_x, max_y, max_z = i, j, k
                max_r = max(max_r, avg_rewards_array[i][j][k])

    max_reward = avg_rewards_array[max_x][max_y][max_z]
    min_step = avg_steps_array[max_x][max_y][max_z]
    max_success_rate = success_rate_array[max_x][max_y][max_z]
    return max_reward, min_step, max_success_rate


def _get_performance_in_varying_noise(
        bit_width: int,
        noise_list: tuple = tuple(range(10, 60, 10))
):
    max_reward_list = []
    min_step_list = []
    max_success_rate_list = []
    for noise in noise_list:
        max_reward, min_step, max_success_rate = _find_max_reward_step_success_rate(bit_width=bit_width, noise=noise)
        max_reward_list.append(max_reward)
        min_step_list.append(min_step)
        max_success_rate_list.append(max_success_rate)
    return max_reward_list, min_step_list, max_success_rate_list


def plot_fixed_policy_reward_3d(bit_width: int, noise: int):
    fig_dir = str(root / fr"figs/grid_search")
    data_dir = str(root / r"extra_data")

    vg_data_path = fr"{data_dir}/delta_vg_bit-{bit_width}_noise-{noise}.npy"
    set_vds_data_path = fr"{data_dir}/delta_set_vds_bit-{bit_width}_noise-{noise}.npy"
    reset_vds_data_path = fr"{data_dir}/delta_reset_vds_bit-{bit_width}_noise-{noise}.npy"
    reward_data_path = fr"{data_dir}/average_rewards_bit-{bit_width}_noise-{noise}.npy"
    step_data_path = fr"{data_dir}/average_steps_bit-{bit_width}_noise-{noise}.npy"
    success_step_data_path = fr"{data_dir}/average_success_steps_bit-{bit_width}_noise-{noise}.npy"
    success_rate_data_path = fr"{data_dir}/success_rates_bit-{bit_width}_noise-{noise}.npy"

    delta_vg_list = np.load(vg_data_path)
    delta_set_vds_list = np.load(set_vds_data_path)
    delta_reset_vds_list = np.load(reset_vds_data_path)
    avg_rewards_array = np.load(reward_data_path)
    avg_steps_array = np.load(step_data_path)
    avg_success_steps_array = np.load(success_step_data_path)
    success_rate_array = np.load(success_rate_data_path)

    # reward, step, success_rate
    reward_image_path = fr"{fig_dir}/average_reward_bit-{bit_width}_noise-{noise}.svg"
    min_r = min([-100, np.min(avg_rewards_array)])
    plot_3d_array(
        array_3d=avg_rewards_array,
        delta_vg_list=delta_vg_list,
        delta_reset_vds_list=delta_reset_vds_list,
        delta_set_vds_list=delta_set_vds_list,
        val_max=100,
        val_min=min_r,
        image_path=reward_image_path,
    )


def plot_fixed_policy_reward_in_varying_noise():
    noise_list = list(range(10, 60, 10))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig_dir = str(root / fr"figs/grid_search")
    # x_axis: noise; y_axis: reward, step, success_rate
    image_path = fr"{fig_dir}/fixed_policy_performance_varying_noise.svg"

    for img_idx, bit_width in enumerate(range(4, 7)):
        max_reward_list, min_step_list, max_success_rate_list = _get_performance_in_varying_noise(bit_width=bit_width)
        axes[0].plot(noise_list, max_reward_list, label=f"{bit_width} bit", marker="o", markersize=4)
        axes[1].plot(noise_list, min_step_list, label=f"{bit_width} bit", marker="o", markersize=4)
        axes[2].plot(noise_list, max_success_rate_list, label=f"{bit_width} bit", marker="o", markersize=4)

    axes[0].set_yticks(np.arange(60, 81, 5))
    axes[0].set_ylim(58, 82)
    axes[1].set_yticks(np.arange(0, 41, 5))
    axes[1].set_ylim(-2, 42)
    axes[2].set_yticks(np.arange(0.9, 1.01, 0.02))
    axes[2].set_ylim(0.89, 1.01)

    axes[0].set_xlabel("noise_std")
    axes[1].set_xlabel("noise_std")
    axes[2].set_xlabel("noise_std")

    axes[0].set_ylabel("maximum_reward")
    axes[1].set_ylabel("step")
    axes[2].set_ylabel("success_rate")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.savefig(image_path, dpi=300)


if __name__ == "__main__":
    # for bit in range(4, 7):
    #     plot_fixed_policy_reward_3d(bit_width=bit, noise=50)
    plot_fixed_policy_reward_in_varying_noise()
