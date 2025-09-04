import logging
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List


def set_logger():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename='{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S')),
                        filemode='w')


def update_persist_train_data(
        log_dir: str,
        data_dict: dict,
        step_count: int,
        episode_steps: List[int],
        episode_success_steps: List[int],
        episode_rewards: List[float],
):
    eval_position_key = f"eval_positions"
    data_key = f"step_{step_count}"

    eval_positions = data_dict.get(eval_position_key, [])
    eval_positions.append(step_count)

    update_items = {
        eval_position_key: eval_positions,
        data_key: {
            "episode_steps": episode_steps,
            "episode_success_steps": episode_success_steps,
            "episode_rewards": episode_rewards,
        }
    }
    data_dict.update(update_items)
    data_path = fr"{log_dir}/train_data.json"
    with open(data_path, "w") as data_f:
        json.dump(data_dict, data_f, indent=4)


def _plot_lower_avg_upper(
        image_path: str,
        x_data: list,
        data_lower: list,
        data_avg: list,
        data_upper: list,
        line_color: str,
        fill_color: str,
        x_label: str,
        y_label: str,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_data, data_avg, color=line_color)
    ax.fill_between(x_data, data_lower, data_upper, facecolor=fill_color, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(image_path)


def _plot_line(
        image_path: str,
        x_data: list,
        line_data: list,
        x_label: str,
        y_label: str,
        line_color: str,

):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_data, line_data, color=line_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(image_path)


def plot_status_1(
        image_path: str,
        steps: list,
        rewards: list,
        write_steps: list,
        success_steps: list,
        success_rates: list,
):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes[0][0].plot(steps, rewards)
    axes[0][0].set_xlabel("Steps")
    axes[0][0].set_ylabel("Rewards")

    axes[0][1].plot(steps, write_steps)
    axes[0][1].set_xlabel("Steps")
    axes[0][1].set_ylabel("Avg write count")

    axes[1][0].plot(steps, success_steps)
    axes[1][0].set_xlabel("Steps")
    axes[1][0].set_ylabel("Avg success write count")

    axes[1][1].plot(steps, success_rates)
    axes[1][1].set_xlabel("Steps")
    axes[1][1].set_ylabel("Success rate")
    plt.savefig(image_path)
    # np.savetxt(train_data_path, train_data)


def plot_status(
        log_dir: str,
        extra_prefix: str = "",
):
    data_path = fr"{log_dir}/train_data.json"
    step_path = fr"{log_dir}/steps_{extra_prefix}.png"
    rewards_path = fr"{log_dir}/rewards_{extra_prefix}.png"
    success_step_path = fr"{log_dir}/success_steps_{extra_prefix}.png"
    success_rate_path = fr"{log_dir}/success_rate_{extra_prefix}.png"
    status_path = fr"{log_dir}/train_status_{extra_prefix}.png"


    with open(data_path, "r") as data_f:
        data_dict = json.load(data_f)

    eval_positions = data_dict["eval_positions"]
    avg_step_list, avg_reward_list, avg_success_step_list, success_rate_list = [], [], [], []

    step_mid_list, reward_mid_list, success_step_mid_list = [], [], []
    step_lower_list, reward_lower_list, success_step_lower_list = [], [], []
    step_upper_list, reward_upper_list, success_step_upper_list = [], [], []

    failed_step_num = 500
    failed_r = -300

    for eval_idx in eval_positions:
        data_key = f"step_{eval_idx}"
        each_dict = data_dict[data_key]
        episode_steps = each_dict["episode_steps"]
        episode_success_steps = each_dict["episode_success_steps"]
        episode_rewards = each_dict["episode_rewards"]
        if episode_rewards:
            avg_step, std_step = (
                np.mean(episode_steps),
                np.std(episode_steps),
            )
            # step_lower, step_upper = avg_step - std_step, avg_step + std_step
            step_lower, step_mid, step_upper = (
                np.quantile(episode_steps, q=0.1), np.quantile(episode_steps, q=0.5), np.quantile(episode_steps, q=0.9),
            )

            avg_r, std_r = (
                np.mean(episode_rewards),
                np.std(episode_rewards),
            )
            # r_lower, r_upper = avg_r - std_r, avg_r + std_r
            r_lower, r_mid, r_upper = (
                np.quantile(episode_rewards, q=0.1), np.quantile(episode_rewards, q=0.5), np.quantile(episode_rewards, q=0.9),
            )

            avg_success_step, std_success_step = (
                np.mean(episode_success_steps),
                np.std(episode_success_steps),
            )
            # success_step_lower, success_step_upper = avg_success_step - std_success_step, avg_success_step + std_success_step
            success_step_lower, success_step_mid, success_step_upper = (
                np.quantile(episode_success_steps, q=0.1), np.quantile(episode_success_steps, q=0.5),
                np.quantile(episode_success_steps, q=0.9),
            )

            success_rate = len(episode_success_steps) / len(episode_steps)
        else:
            avg_step, step_lower, step_mid, step_upper = failed_step_num, failed_step_num, failed_step_num, failed_step_num
            avg_r, r_lower, r_mid, r_upper = failed_r, failed_r, failed_r, failed_r
            avg_success_step, success_step_lower, success_step_mid, success_step_upper = (
                failed_step_num, failed_step_num, failed_step_num, failed_step_num
            )
            success_rate = 0

        avg_step_list.append(avg_step)
        step_lower_list.append(step_lower)
        step_mid_list.append(step_mid)
        step_upper_list.append(step_upper)

        avg_reward_list.append(avg_r)
        reward_lower_list.append(r_lower)
        reward_mid_list.append(r_mid)
        reward_upper_list.append(r_upper)

        avg_success_step_list.append(avg_success_step)
        success_step_lower_list.append(success_step_lower)
        success_step_mid_list.append(success_step_mid)
        success_step_upper_list.append(success_step_upper)
        success_rate_list.append(success_rate)

    _plot_line(
        image_path=success_rate_path, line_data=success_rate_list, x_data=eval_positions, line_color="red",
        x_label="Steps", y_label="Success rate",
    )

    _plot_lower_avg_upper(
        image_path=success_step_path, x_data=eval_positions, data_avg=success_step_mid_list,
        data_lower=success_step_lower_list, data_upper=success_step_upper_list, line_color="red", fill_color="pink",
        x_label="Steps", y_label="Successful write count"
    )

    _plot_lower_avg_upper(
        image_path=step_path, x_data=eval_positions, data_avg=step_mid_list,
        data_lower=step_lower_list, data_upper=step_upper_list, line_color="red", fill_color="pink",
        x_label="Steps", y_label="Write count"
    )

    _plot_lower_avg_upper(
        image_path=rewards_path, x_data=eval_positions, data_avg=reward_mid_list,
        data_lower=reward_lower_list, data_upper=reward_upper_list, line_color="red", fill_color="pink",
        x_label="Steps", y_label="Episode rewards"
    )

    plot_status_1(
        image_path=status_path, steps=eval_positions, rewards=avg_reward_list, write_steps=avg_step_list,
        success_steps=avg_success_step_list, success_rates=success_rate_list,
    )