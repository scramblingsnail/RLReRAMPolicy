import json
import os.path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import Axes
from pathlib import Path
from typing import List, Union


root = Path(__file__)
for idx in range(3):
    root = root.parent


blue_array = np.array([19, 164, 192, 255]) / 255
red_array = np.array([234, 96, 142, 255]) / 255


TD3_COLOR = np.array([19, 164, 192, 255]) / 255
TD3_FILL_COLOR = np.array([210, 233, 240, 255]) / 255
SAC_COLOR = np.array([246, 173, 60, 255]) / 255
SAC_FILL_COLOR = np.array([250, 223, 194, 255]) / 255
PPO_COLOR = np.array([234, 96, 142, 255]) / 255
PPO_FILL_COLOR = np.array([246, 208, 217, 255]) / 255



def read_log_data(
        log_data_path: str,
        max_steps: int = int(4e6),
        upper_percent: float = 0.75,
        mid_percent: float = 0.5,
        lower_percent: float = 0.25,

):
    r"""

    Args:
        log_data_path (str):
        max_steps (int):
        upper_percent (float):
        mid_percent (float):
        lower_percent (float):

    Returns:
        eval_step_list: np.ndarray (eval_times_dim, )
        step_lower_list: np.ndarray (eval_times_dim, )
        step_mid_list: np.ndarray (eval_times_dim, )
        step_upper_list: np.ndarray (eval_times_dim, )
        reward_lower_list: np.ndarray (eval_times_dim, )
        reward_mid_list: np.ndarray (eval_times_dim, )
        reward_upper_list: np.ndarray (eval_times_dim, )
        success_rate_list: np.ndarray (eval_times_dim, )

    """
    with open(log_data_path, "r") as log_f:
        data_dict = json.load(log_f)

    eval_step_list = np.array(data_dict["eval_positions"])

    valid_eval_step_list = []
    step_lower_list, step_mid_list, step_upper_list = [], [], []
    reward_lower_list, reward_mid_list, reward_upper_list = [], [], []
    success_rate_list = []

    for eval_step_idx in eval_step_list:
        if eval_step_idx > max_steps:
            break

        valid_eval_step_list.append(eval_step_idx)
        eval_data_key = f"step_{eval_step_idx}"
        eval_data_dict = data_dict[eval_data_key]
        eval_episodic_steps = eval_data_dict["episode_steps"]
        eval_episodic_success_steps = eval_data_dict["episode_success_steps"]
        eval_episodic_rewards = eval_data_dict["episode_rewards"]

        step_lower, step_mid, step_upper = (
            np.quantile(eval_episodic_steps, q=lower_percent),
            np.quantile(eval_episodic_steps, q=mid_percent),
            np.quantile(eval_episodic_steps, q=upper_percent),
        )
        step_lower_list.append(step_lower)
        step_mid_list.append(step_mid)
        step_upper_list.append(step_upper)

        reward_lower, reward_mid, reward_upper = (
            np.quantile(eval_episodic_rewards, q=lower_percent),
            np.quantile(eval_episodic_rewards, q=mid_percent),
            np.quantile(eval_episodic_rewards, q=upper_percent),
        )
        reward_lower_list.append(reward_lower)
        reward_mid_list.append(reward_mid)
        reward_upper_list.append(reward_upper)

        success_rate = len(eval_episodic_success_steps) / len(eval_episodic_steps)
        success_rate_list.append(success_rate)

    print(
        len(valid_eval_step_list), len(step_lower_list), len(step_mid_list), len(step_upper_list),
        len(reward_lower_list), len(reward_mid_list), len(reward_upper_list), len(success_rate_list),
    )

    return (
        valid_eval_step_list,
        step_lower_list, step_mid_list, step_upper_list,
        reward_lower_list, reward_mid_list, reward_upper_list,
        success_rate_list,
    )


def _plot_lower_avg_upper(
        ax: Axes,
        x_data: list,
        data_lower: list,
        data_avg: list,
        data_upper: list,
        data_label: str,
        line_color: Union[str, np.ndarray],
        fill_color: Union[str, np.ndarray],
):
    ax.plot(x_data, data_avg, color=line_color, label=data_label, linewidth=0.5)
    ax.fill_between(x_data, data_lower, data_upper, facecolor=fill_color, alpha=0.7)


def _plot_algorithms_rewards(
        ax: Axes,
        eval_step_list: List[int],
        td3_reward_lower_list: List[float],
        td3_reward_mid_list: List[float],
        td3_reward_upper_list: List[float],
        ppo_reward_lower_list: List[float],
        ppo_reward_mid_list: List[float],
        ppo_reward_upper_list: List[float],
        sac_reward_lower_list: List[float],
        sac_reward_mid_list: List[float],
        sac_reward_upper_list: List[float],
):
    _plot_lower_avg_upper(
        ax=ax,
        x_data=eval_step_list,
        data_lower=sac_reward_lower_list,
        data_avg=sac_reward_mid_list,
        data_upper=sac_reward_upper_list,
        data_label="SAC",
        line_color=SAC_COLOR,
        fill_color=SAC_FILL_COLOR,
    )
    _plot_lower_avg_upper(
        ax=ax,
        x_data=eval_step_list,
        data_lower=td3_reward_lower_list,
        data_avg=td3_reward_mid_list,
        data_upper=td3_reward_upper_list,
        data_label="TD3",
        line_color=TD3_COLOR,
        fill_color=TD3_FILL_COLOR,
    )
    _plot_lower_avg_upper(
        ax=ax,
        x_data=eval_step_list,
        data_lower=ppo_reward_lower_list,
        data_avg=ppo_reward_mid_list,
        data_upper=ppo_reward_upper_list,
        data_label="PPO",
        line_color=PPO_COLOR,
        fill_color=PPO_FILL_COLOR,
    )
    ax.legend()
    ax.set_yticks(np.arange(-100, 101, 25))
    ax.set_ylim(-110, 110)
    ax.tick_params(axis="both", direction="in", width=0.5)
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)


def _plot_algorithms_steps(
        ax: Axes,
        eval_step_list: List[int],
        td3_step_lower_list: List[float],
        td3_step_mid_list: List[float],
        td3_step_upper_list: List[float],
        ppo_step_lower_list: List[float],
        ppo_step_mid_list: List[float],
        ppo_step_upper_list: List[float],
        sac_step_lower_list: List[float],
        sac_step_mid_list: List[float],
        sac_step_upper_list: List[float],
):
    _plot_lower_avg_upper(
        ax=ax,
        x_data=eval_step_list,
        data_lower=sac_step_lower_list,
        data_avg=sac_step_mid_list,
        data_upper=sac_step_upper_list,
        data_label="SAC",
        line_color=SAC_COLOR,
        fill_color=SAC_FILL_COLOR,
    )
    _plot_lower_avg_upper(
        ax=ax,
        x_data=eval_step_list,
        data_lower=td3_step_lower_list,
        data_avg=td3_step_mid_list,
        data_upper=td3_step_upper_list,
        data_label="TD3",
        line_color=TD3_COLOR,
        fill_color=TD3_FILL_COLOR,
    )
    _plot_lower_avg_upper(
        ax=ax,
        x_data=eval_step_list,
        data_lower=ppo_step_lower_list,
        data_avg=ppo_step_mid_list,
        data_upper=ppo_step_upper_list,
        data_label="PPO",
        line_color=PPO_COLOR,
        fill_color=PPO_FILL_COLOR,
    )
    ax.legend()
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(-5, 105)
    ax.tick_params(axis="both", direction="in", width=0.5)
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)


def _plot_algorithms_success_rates(
        ax: Axes,
        eval_step_list: List[int],
        td3_success_rate_list: List[float],
        ppo_success_rate_list: List[float],
        sac_success_rate_list: List[float],
):
    ax.plot(eval_step_list, sac_success_rate_list, color=SAC_COLOR, label="SAC", linewidth=0.5)
    ax.plot(eval_step_list, td3_success_rate_list, color=TD3_COLOR, label="TD3", linewidth=0.5)
    ax.plot(eval_step_list, ppo_success_rate_list, color=PPO_COLOR, label="PPO", linewidth=0.5)
    ax.legend()
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", direction="in", width=0.5)
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)


def plot_rl_bit_performance(bit_width: int, noise_list: List[float]):
    log_dir = root / r"logs"

    image_dir = root / r"figs/rl_variations"
    if not os.path.exists(str(image_dir)):
        os.makedirs(str(image_dir))

    image_path = image_dir / r"rl_performance_bit-{}.svg".format(bit_width)

    row_num, col_num = len(noise_list), 3
    fig, axes = plt.subplots(row_num, col_num, figsize=(3 * 5, len(noise_list) * 5))

    for row_idx in range(row_num):
        noise_std = noise_list[row_idx]
        sac_log_path = log_dir / fr"SAC_bit-{int(bit_width)}_noise-{int(noise_std * 100)}/train_data.json"
        ppo_log_path = log_dir / fr"PPO_bit-{int(bit_width)}_noise-{int(noise_std * 100)}/train_data.json"
        td3_log_path = log_dir / fr"TD3_bit-{int(bit_width)}_noise-{int(noise_std * 100)}/train_data.json"
        (
            eval_step_list,
            sac_step_lower_list, sac_step_mid_list, sac_step_upper_list,
            sac_reward_lower_list, sac_reward_mid_list, sac_reward_upper_list,
            sac_success_rate_list,
        ) = read_log_data(log_data_path=sac_log_path)

        (
            eval_step_list,
            ppo_step_lower_list, ppo_step_mid_list, ppo_step_upper_list,
            ppo_reward_lower_list, ppo_reward_mid_list, ppo_reward_upper_list,
            ppo_success_rate_list,
        ) = read_log_data(log_data_path=ppo_log_path)

        (
            eval_step_list,
            td3_step_lower_list, td3_step_mid_list, td3_step_upper_list,
            td3_reward_lower_list, td3_reward_mid_list, td3_reward_upper_list,
            td3_success_rate_list,
        ) = read_log_data(log_data_path=td3_log_path)

        axes[row_idx][0].set_xlabel("Steps")
        axes[row_idx][0].set_ylabel("Average Episodic Reward")
        axes[row_idx][0].set_title(f"noise: {noise_std*100}; bit: {bit_width}")

        axes[row_idx][1].set_xlabel("Steps")
        axes[row_idx][1].set_ylabel("Average Program Count")
        axes[row_idx][1].set_title(f"noise: {noise_std * 100}; bit: {bit_width}")

        axes[row_idx][2].set_xlabel("Steps")
        axes[row_idx][2].set_ylabel("Average Success Rate")
        axes[row_idx][2].set_title(f"noise: {noise_std * 100}; bit: {bit_width}")

        _plot_algorithms_rewards(
            ax=axes[row_idx][0],
            eval_step_list=eval_step_list,
            td3_reward_lower_list=td3_reward_lower_list,
            td3_reward_mid_list=td3_reward_mid_list,
            td3_reward_upper_list=td3_reward_upper_list,
            ppo_reward_lower_list=ppo_reward_lower_list,
            ppo_reward_mid_list=ppo_reward_mid_list,
            ppo_reward_upper_list=ppo_reward_upper_list,
            sac_reward_lower_list=sac_reward_lower_list,
            sac_reward_mid_list=sac_reward_mid_list,
            sac_reward_upper_list=sac_reward_upper_list,
        )

        _plot_algorithms_steps(
            ax=axes[row_idx][1],
            eval_step_list=eval_step_list,
            td3_step_lower_list=td3_step_lower_list,
            td3_step_mid_list=td3_step_mid_list,
            td3_step_upper_list=td3_step_upper_list,
            ppo_step_lower_list=ppo_step_lower_list,
            ppo_step_mid_list=ppo_step_mid_list,
            ppo_step_upper_list=ppo_step_upper_list,
            sac_step_lower_list=sac_step_lower_list,
            sac_step_mid_list=sac_step_mid_list,
            sac_step_upper_list=sac_step_upper_list,
        )

        _plot_algorithms_success_rates(
            ax=axes[row_idx][2],
            eval_step_list=eval_step_list,
            td3_success_rate_list=td3_success_rate_list,
            sac_success_rate_list=sac_success_rate_list,
            ppo_success_rate_list=ppo_success_rate_list,
        )
    plt.savefig(str(image_path))


if __name__ == "__main__":
    plot_rl_bit_performance(bit_width=5, noise_list=[0.1, 0.2, 0.3, 0.4, 0.5])
