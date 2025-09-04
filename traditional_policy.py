import os
import h5py
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

from src.MemristorENV import build_1t1r_env, OneTOneREnv
from src.MemristorENV.utils import load_1t1r_config
from src.Algorithm.baseline import FeedBackWriter


def eval_fixed_strategy(supervise_cfg):
    mem_env = build_1t1r_env()
    neg_vr_step = supervise_cfg['neg_vr_step']
    pos_vg_step = supervise_cfg['pos_vg_step']
    pos_vr_step = supervise_cfg['pos_vr_step']
    feed = FeedBackWriter(mem_env, neg_vr_step=neg_vr_step, pos_vg_step=pos_vg_step, pos_vr_step=pos_vr_step)

    episodic_rewards, episodic_steps = feed.evaluate(max_steps=5000)
    print("Episodic rewards: \n", episodic_rewards)
    print("Episodic steps: \n", episodic_steps)


def load_grid_results(delta_vr_list = None, delta_vg_list = None):
    if delta_vr_list is None:
        delta_vr_list = np.arange(0.05, 0.35, 0.05)
    if delta_vg_list is None:
        delta_vg_list = np.arange(0.005, 0.025, 0.005)

    avg_rewards_array = np.zeros((delta_vg_list.shape[0], delta_vr_list.shape[0]))
    avg_steps_array = np.zeros((delta_vg_list.shape[0], delta_vr_list.shape[0]))
    success_rate_array = np.zeros((delta_vg_list.shape[0], delta_vr_list.shape[0]))

    for vg_idx, delta_vg in enumerate(delta_vg_list):
        for vr_idx, delta_vr in enumerate(delta_vr_list):
            reward_path = "./supervise_data/grid_search/reward_dvg_{:.3f}_dvr_{:.3f}.npy".format(delta_vg, delta_vr)
            step_path = "./supervise_data/grid_search/step_dvg_{:.3f}_dvr_{:.3f}.npy".format(delta_vg, delta_vr)
            rewards = np.load(reward_path)
            steps = np.load(step_path)

            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps)
            success_rate = steps.shape[0] / rewards.shape[0]
            print(f"AVG reward {avg_reward}, AVG steps {avg_steps}, Success rate {success_rate}")
            avg_rewards_array[vg_idx, vr_idx] = avg_reward
            avg_steps_array[vg_idx, vr_idx] = avg_steps
            success_rate_array[vg_idx, vr_idx] = success_rate
    return avg_rewards_array, avg_steps_array, success_rate_array


def evaluate(eval_envs: OneTOneREnv, max_steps, incremental_action: np.ndarray):
    r"""
    evaluate the trained network's performance: until terminal or exceeding max steps.

    Args:
        eval_envs (OneTOneREnv): envs for evaluating
        max_steps (int): max evaluate steps
        eval_episodic_num (int): the number of eval episodic that needs to be obtained.
        data_save_path (str): path for saving total evaluate data
        sub_data_save_path (str): path for saving current evaluate data.
            (It is generally a temporary path for converting to gifs)
        save_data (bool): whether to save data.

    Returns:
        avg_reward (float): average episodic reward
        avg_step (float): average episodic step
    """
    # TODO: with no grad
    state = eval_envs.reset()[0]
    rewards = np.zeros((eval_envs.num_envs, max_steps), dtype=float)
    all_truncated = np.zeros((eval_envs.num_envs, max_steps), dtype=bool)
    terminals = np.zeros((eval_envs.num_envs, max_steps), dtype=bool)
    accumulate_rewards = np.zeros(eval_envs.num_envs, dtype=float)
    start_indices = np.zeros(eval_envs.num_envs, dtype=int)

    episodic_rewards = []
    success_episodic_steps = []
    episodic_steps = []

    episodic_num = 0

    # record failed states:
    failed_start_states = []
    # failed_state_path = rf"{log_dir}/eval_{self.total_steps}_failed.txt"
    failed_state = np.zeros_like(state)

    print('Evaluating multi envs ...')
    for step_idx in range(max_steps):
        # interact with envs
        state, reward, terminal, truncated, _ = eval_envs.step(incremental_action)
        rewards[:, step_idx] = reward
        terminals[:, step_idx] = terminal
        all_truncated[:, step_idx] = truncated

        # record failed state
        if truncated:
            failed_start_states.append(failed_state)
            failed_state = state

        success = np.bitwise_and(np.bitwise_not(truncated), terminal)

        # TODO: align with gym 0.24.0: truncate also mean terminals
        terminal = np.bitwise_or(terminal, truncated)

        episodic_num += np.sum(terminal)
        # for envs that terminate:
        # add the final rewards.
        accumulate_rewards += reward
        episodic_rewards += list(accumulate_rewards[terminal])
        episodic_steps += [max(step_idx - start_idx, 1) for start_idx in start_indices[terminal]]
        success_episodic_steps += [max(step_idx - start_idx, 1) for start_idx in start_indices[success]]
        # Reset record
        accumulate_rewards[terminal] = 0
        start_indices[terminal] = step_idx

    if len(episodic_rewards) > 0:
        avg_reward = np.mean(episodic_rewards)
        avg_step = np.mean(episodic_steps)
        if success_episodic_steps:
            avg_success_step = np.mean(success_episodic_steps)
        else:
            avg_success_step = max_steps
        success_rate = len(success_episodic_steps) / len(episodic_rewards)
    else:
        avg_reward, avg_step = -200, max_steps
        avg_success_step = max_steps
        success_rate = 0

    torch.set_grad_enabled(True)
    return avg_reward, avg_step, avg_success_step, success_rate


def plot_array(array, delta_vg_list, delta_vr_list, title: str):
    eval_dir = r"./supervise_data"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    fig, axes = plt.subplots(figsize=(array.shape[1] + 1, array.shape[0]))

    ax = axes
    plt.cla()
    img = ax.imshow(array)
    # ax.set_xlim(0, array.shape[1])
    # ax.set_ylim(0, array.shape[0])
    x_ticks = list(range(delta_vr_list.shape[0]))
    y_ticks = list(range(delta_vg_list.shape[0]))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["{:.3f}".format(x) for x in delta_vr_list])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["{:.3f}".format(y) for y in delta_vg_list])
    ax.set_xlabel("Delta Vds / V")
    ax.set_ylabel("Delta Vg / V")
    cb_ax = fig.add_axes([0.9, 0.115, 0.025, 0.765])
    cb = fig.colorbar(img, cax=cb_ax)
    for x_idx in range(array.shape[0]):
        for y_idx in range(array.shape[1]):
            ax.text(x=y_idx, y=x_idx, s="{:.2f}".format(array[x_idx, y_idx]), ha='center', va='center', color='black')

    ax.set_title(title)
    plt.savefig(f"./supervise_data/{title}.png", dpi=300)


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
    ax3d = Axes3D(fig)
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

    ax3d.scatter(
        xs=x_data, ys=y_data, zs=z_data, s=sizes, c=val_data, vmin=val_min, vmax=val_max,cmap=newcmp, rasterized=True,
    )
    ax3d.view_init(elev=15, azim=135)
    plt.savefig(image_path, dpi=300)


def search_incremental_action(config_dict: dict):
    # TODO: noise
    noise_std = config_dict["noise_std"]
    bit_width = config_dict["bit_width"]

    memristor_config, transistor_config, simulate_config, array_config = load_1t1r_config()
    array_config["memristor_relative_noise"] = noise_std
    array_config["bit_width"] = bit_width

    mem_env = OneTOneREnv(
        array_para_dict=array_config,
        memristor_para_dict=memristor_config,
        transistor_para_dict=transistor_config,
        simulate_para_dict=simulate_config,
    )
    eval_steps = 500

    # delta_set_vds_list = np.arange(0.05, 1.51, 0.05)
    # delta_reset_vds_list = np.arange(-1.5, -0.04, 0.05)
    # delta_vg_list = np.arange(0.002, 0.051, 0.002)

    delta_set_vds_list = np.arange(0.1, 1.51, 0.05)
    delta_reset_vds_list = np.arange(-1.5, -0.09, 0.05)
    delta_vg_list = np.arange(0.001, 0.004, 0.002)

    set_vds_num = delta_set_vds_list.shape[0]
    reset_vds_num = delta_reset_vds_list.shape[0]
    vg_num = delta_vg_list.shape[0]

    avg_rewards_array = np.zeros((vg_num, reset_vds_num, set_vds_num))
    avg_steps_array = np.zeros((vg_num, reset_vds_num, set_vds_num))
    avg_success_steps_array = np.zeros((vg_num, reset_vds_num, set_vds_num))
    success_rate_array = np.zeros((vg_num, reset_vds_num, set_vds_num))

    for vg_idx, delta_vg in enumerate(delta_vg_list):
        for reset_vds_idx, delta_reset_vds in enumerate(delta_reset_vds_list):
            for set_vds_idx, delta_set_vds in enumerate(delta_set_vds_list):
                # d_vg_set, d_vds_set, d_vg_reset, d_vds_reset
                print(
                    f"Processing -- bit {bit_width} -- std {noise_std} -- Vg {delta_vg} "
                    f"-- Reset_Vds {delta_reset_vds} -- Set_Vds {delta_set_vds}"
                )
                fixed_action = np.array([delta_vg, delta_set_vds, delta_vg, delta_reset_vds])
                fixed_action = fixed_action[np.newaxis, :]
                avg_reward, avg_step, avg_success_step, success_rate = evaluate(
                    eval_envs=mem_env, max_steps=eval_steps, incremental_action=fixed_action
                )
                print("avg_reward: ", avg_reward)
                print("avg_step: ", avg_step)
                print("avg_success_step: ", avg_success_step)
                print("success_rate: ", success_rate)

                avg_rewards_array[vg_idx, reset_vds_idx, set_vds_idx] = avg_reward
                avg_steps_array[vg_idx, reset_vds_idx, set_vds_idx] = avg_step
                success_rate_array[vg_idx, reset_vds_idx, set_vds_idx] = success_rate
                avg_success_steps_array[vg_idx, reset_vds_idx, set_vds_idx] = avg_success_step

    fig_dir = r"./figs/grid_search"
    data_dir = r"./extra_data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    vg_data_path = fr"{data_dir}/delta_vg_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"
    set_vds_data_path = fr"{data_dir}/delta_set_vds_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"
    reset_vds_data_path = fr"{data_dir}/delta_reset_vds_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"
    reward_data_path = fr"{data_dir}/average_rewards_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"
    step_data_path = fr"{data_dir}/average_steps_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"
    success_step_data_path = fr"{data_dir}/average_success_steps_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"
    success_rate_data_path = fr"{data_dir}/success_rates_bit-{bit_width}_noise-{int(noise_std*100)}_rest.npy"

    np.save(vg_data_path, delta_vg_list)
    np.save(set_vds_data_path, delta_set_vds_list)
    np.save(reset_vds_data_path, delta_reset_vds_list)
    np.save(reward_data_path, avg_rewards_array)
    np.save(step_data_path, avg_steps_array)
    np.save(success_step_data_path, avg_success_steps_array)
    np.save(success_rate_data_path, success_rate_array)


    # reward_image_path = fr"{fig_dir}/average_reward.svg"
    #
    # print("max r: {}; min r: {}".format(np.max(avg_rewards_array), np.min(avg_rewards_array)))
    #
    # min_r = min([-100, np.min(avg_rewards_array)])
    #
    # plot_3d_array(
    #     array_3d=avg_rewards_array,
    #     delta_vg_list=delta_vg_list,
    #     delta_reset_vds_list=delta_reset_vds_list,
    #     delta_set_vds_list=delta_set_vds_list,
    #     val_max=100,
    #     val_min=min_r,
    #     image_path=reward_image_path,
    # )

    # plot_array(array=avg_rewards_array, title="Avg_reward", delta_vg_list=delta_vg_list, delta_vr_list=delta_vr_list)
    # plot_array(
    #     array=avg_steps_array,
    #     title="Average_steps", delta_vg_list=delta_vg_list, delta_vr_list=delta_vr_list,
    # )
    # plot_array(array=success_rate_array, title="Success_rate", delta_vg_list=delta_vg_list, delta_vr_list=delta_vr_list)
    #
    # plot_array(
    #     array=avg_success_steps_array,
    #     title="Average_success_steps",
    #     delta_vg_list=delta_vg_list,
    #     delta_vr_list=delta_vr_list,
    # )

def plot_fixed_policy_performance():
    fig_dir = r"./figs/grid_search"
    data_dir = r"./extra_data"

    vg_data_path = fr"{data_dir}/delta_vg.npy"
    set_vds_data_path = fr"{data_dir}/delta_set_vds.npy"
    reset_vds_data_path = fr"{data_dir}/delta_reset_vds.npy"
    reward_data_path = fr"{data_dir}/average_rewards.npy"
    step_data_path = fr"{data_dir}/average_steps.npy"
    success_step_data_path = fr"{data_dir}/average_success_steps.npy"
    success_rate_data_path = fr"{data_dir}/success_rates.npy"

    delta_vg_list = np.load(vg_data_path)
    delta_set_vds_list = np.load(set_vds_data_path)
    delta_reset_vds_list = np.load(reset_vds_data_path)
    avg_rewards_array = np.load(reward_data_path)
    avg_steps_array = np.load(step_data_path)
    avg_success_steps_array = np.load(success_step_data_path)
    success_rate_array = np.load(success_rate_data_path)

    # reward, step, success_rate
    reward_image_path = fr"{fig_dir}/average_reward.svg"
    min_r = min([-100, np.min(avg_rewards_array)])

    max_r = -1e3
    max_x, max_y, max_z = 0, 0 , 0
    for i in range(avg_rewards_array.shape[0]):
        for j in range(avg_rewards_array.shape[1]):
            for k in range(avg_rewards_array.shape[2]):
                if avg_rewards_array[i][j][k] > max_r:
                    max_x, max_y, max_z = i, j ,k
                max_r = max(max_r, avg_rewards_array[i][j][k])

    print("max indices: ", max_x, max_y, max_z)
    print("min steps: ", avg_steps_array[max_x][max_y][max_z])
    print("max success rate: ", success_rate_array[max_x][max_y][max_z])
    print("max reward: ", avg_rewards_array[max_x][max_y][max_z])

    max_reward = np.max(avg_rewards_array)
    print(max_reward)

    plot_3d_array(
        array_3d=avg_rewards_array,
        delta_vg_list=delta_vg_list,
        delta_reset_vds_list=delta_reset_vds_list,
        delta_set_vds_list=delta_set_vds_list,
        val_max=100,
        val_min=min_r,
        image_path=reward_image_path,
    )


def search_action():
    mem_env = build_1t1r_env()

    delta_vr_list = np.arange(0.05, 0.35, 0.05)
    delta_vg_list = np.arange(0.001, 0.006, 0.002)

    eval_steps = 100

    avg_rewards_array = np.zeros((delta_vg_list.shape[0], delta_vr_list.shape[0]))
    avg_steps_array = np.zeros((delta_vg_list.shape[0], delta_vr_list.shape[0]))
    success_rate_array = np.zeros((delta_vg_list.shape[0], delta_vr_list.shape[0]))

    failed_step = -10
    no_valid_episodic_reward = -500

    for vg_idx, delta_vg in enumerate(delta_vg_list):
        for vr_idx, delta_vr in enumerate(delta_vr_list):
            print(f"Processing -- Vg {delta_vg} -- Vr {delta_vr}")
            feed = FeedBackWriter(mem_env, neg_vr_step=delta_vr, pos_vg_step=delta_vg, pos_vr_step=delta_vr)
            episodic_rewards, episodic_steps = feed.evaluate(max_steps=eval_steps)
            valid_steps = [s for s in episodic_steps if not isinstance(s, tuple)]
            success_rate = len(valid_steps) / max(1, len(episodic_steps))
            if episodic_rewards:
                avg_reward = np.mean(episodic_rewards)
            else:
                avg_reward = no_valid_episodic_reward

            print("Valid steps: ", len(valid_steps))
            print("Success rate: ", success_rate)
            print("Episodic steps: ", len(episodic_steps))

            if len(valid_steps) == 0:
                avg_step = failed_step
            else:
                avg_step = np.mean(valid_steps)

            reward_path = "./supervise_data/grid_search/reward_dvg_{:.3f}_dvr_{:.3f}.npy".format(delta_vg, delta_vr)
            step_path = "./supervise_data/grid_search/step_dvg_{:.3f}_dvr_{:.3f}.npy".format(delta_vg, delta_vr)
            np.save(reward_path, np.array(episodic_rewards))
            np.save(step_path, np.array(valid_steps))

            avg_rewards_array[vg_idx, vr_idx] = avg_reward
            avg_steps_array[vg_idx, vr_idx] = avg_step
            success_rate_array[vg_idx, vr_idx] = success_rate

    avg_rewards_array, avg_steps_array, success_rate_array = load_grid_results(
        delta_vr_list=delta_vr_list,
        delta_vg_list=delta_vg_list,
    )

    plot_array(array=success_rate_array, title="Success Rate")
    plot_array(array=avg_steps_array, title="Average Steps (successful)")
    plot_array(array=avg_rewards_array, title="Average Rewards")


def plot_eval_distribution():
    dvg = 0.01
    dvr = 0.1
    reward_path = r"./supervise_data/grid_search/reward_dvg_{:.3f}_dvr_{:.3f}.npy".format(dvg, dvr)
    step_path = r"./supervise_data/grid_search/step_dvg_{:.3f}_dvr_{:.3f}.npy".format(dvg, dvr)

    rewards = np.load(reward_path)
    steps = np.load(step_path)

    br_rewards = np.broadcast_to(rewards[:, np.newaxis], (rewards.shape[0], 10))
    br_steps = np.broadcast_to(steps[:, np.newaxis], (steps.shape[0], 10))

    reward25 = np.percentile(br_rewards, 25, axis=0)
    reward50 = np.percentile(br_rewards, 50, axis=0)
    reward75 = np.percentile(br_rewards, 75, axis=0)

    step25 = np.percentile(br_steps, 25, axis=0)
    step50 = np.percentile(br_steps, 50, axis=0)
    step75 = np.percentile(br_steps, 75, axis=0)

    plt.plot(reward25, color="tomato")
    plt.plot(reward75, color="tomato")
    plt.plot(reward50, color="red")
    plt.ylim(0, 200)
    plt.ylabel("Reward")
    plt.fill_between(list(range(reward50.shape[0])), reward25, reward75, alpha=0.5, color="tomato")
    plt.show()

    plt.plot(step25, color="tomato")
    plt.plot(step75, color="tomato")
    plt.plot(step50, color="red")
    plt.ylim(0, 200)
    plt.ylabel("Step num")
    plt.fill_between(list(range(step50.shape[0])), step25, step75, alpha=0.5, color="tomato")
    plt.show()

    plt.boxplot(rewards)
    plt.ylabel("Reward")
    plt.show()

    plt.boxplot(steps)
    plt.ylabel("Step num")
    plt.show()


def concat_rest_result(bit_width: int, noise_std: float):
    data_dir = r"./extra_data"

    init_vg_data_path = fr"{data_dir}/delta_vg_bit-{bit_width}_noise-{int(noise_std*100)}.npy"
    init_reward_data_path = fr"{data_dir}/average_rewards_bit-{bit_width}_noise-{int(noise_std*100)}.npy"
    init_step_data_path = fr"{data_dir}/average_steps_bit-{bit_width}_noise-{int(noise_std*100)}.npy"
    init_success_step_data_path = fr"{data_dir}/average_success_steps_bit-{bit_width}_noise-{int(noise_std*100)}.npy"
    init_success_rate_data_path = fr"{data_dir}/success_rates_bit-{bit_width}_noise-{int(noise_std*100)}.npy"

    rest_vg_data_path = fr"{data_dir}/delta_vg_bit-{bit_width}_noise-{int(noise_std * 100)}_rest.npy"
    rest_reward_data_path = fr"{data_dir}/average_rewards_bit-{bit_width}_noise-{int(noise_std * 100)}_rest.npy"
    rest_step_data_path = fr"{data_dir}/average_steps_bit-{bit_width}_noise-{int(noise_std * 100)}_rest.npy"
    rest_success_step_data_path = fr"{data_dir}/average_success_steps_bit-{bit_width}_noise-{int(noise_std * 100)}_rest.npy"
    rest_success_rate_data_path = fr"{data_dir}/success_rates_bit-{bit_width}_noise-{int(noise_std * 100)}_rest.npy"

    init_vg_data = np.load(init_vg_data_path)
    init_reward_data = np.load(init_reward_data_path)
    init_step_data = np.load(init_step_data_path)
    init_success_step_data = np.load(init_success_step_data_path)
    init_success_rate_data = np.load(init_success_rate_data_path)

    rest_vg_data = np.load(rest_vg_data_path)
    rest_reward_data = np.load(rest_reward_data_path)
    rest_step_data = np.load(rest_step_data_path)
    rest_success_step_data = np.load(rest_success_step_data_path)
    rest_success_rate_data = np.load(rest_success_rate_data_path)

    new_vg_data = np.concatenate((init_vg_data, rest_vg_data), axis=0)
    new_reward_data = np.concatenate((init_reward_data, rest_reward_data), axis=0)
    new_step_data = np.concatenate((init_step_data, rest_step_data), axis=0)
    new_success_step_data = np.concatenate((init_success_step_data, rest_success_step_data), axis=0)
    new_success_rate_data = np.concatenate((init_success_rate_data, rest_success_rate_data), axis=0)

    # np.save(init_vg_data_path, new_vg_data)
    # np.save(init_reward_data_path, new_reward_data)
    # np.save(init_step_data_path, new_step_data)
    # np.save(init_success_step_data_path, new_success_step_data)
    # np.save(init_success_rate_data_path, new_success_rate_data)


def multi_core_search():
    memristor_relative_noise_range = [0.3, ]  # np.arange(0.1, 0.51, 0.1)
    bit_width_range = [5, ]  # list(range(4, 7))
    config_list = []
    for bit_width in bit_width_range:
        for noise in memristor_relative_noise_range:
            noise_percent = int(noise * 100)
            config_dict = {"bit_width": bit_width, "noise_std": noise}
            config_list.append(config_dict)

    with mp.Pool(processes=1) as pool:
        pool.map(search_incremental_action, config_list)

if __name__ == "__main__":
    # config_path = "./src/Algorithm/baseline/supervise_config.yaml"
    # with open(config_path, 'r') as config_f:
    #     supervise_cfg_dict = yaml.safe_load(config_f)
    #
    # eval_fixed_strategy(supervise_cfg=supervise_cfg_dict)

    # search_action()

    # (5, 30)

    # multi_core_search()

    plot_fixed_policy_performance()

    # load_grid_results()

    # plot_eval_distribution()