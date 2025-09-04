import torch
import time
import os
import numpy as np
import multiprocessing as mp
import json

from pathlib import Path
from typing import List

from src.Algorithm.rl import PPO
from src.Algorithm.net import PPOActor, PPOCritic
from src.utils import set_logger, plot_status, update_persist_train_data
from src.MemristorENV import build_1t1r_env, load_1t1r_config


set_logger()
torch.manual_seed(2025)
np.random.seed(2025)

root = Path(__file__).parent
current_time = time.strftime('%m-%d-%H-%M')


r""" setting """
batch_size = 128


def build_envs(**array_kwargs):
    memristor_config, transistor_config, simulate_config, array_config = load_1t1r_config()

    for key in array_kwargs.keys():
        if key not in array_config.keys():
            raise ValueError(f"Invalid array arg: {key}")

    array_config.update(array_kwargs)

    mem_envs = build_1t1r_env(
        memristor_config=memristor_config,
        transistor_config=transistor_config,
        simulate_config=simulate_config,
        array_config=array_config,
    )
    array_config['sampler_num'] = 1
    eval_mem_envs = build_1t1r_env(
        memristor_config=memristor_config,
        transistor_config=transistor_config,
        simulate_config=simulate_config,
        array_config=array_config,
    )
    return mem_envs, eval_mem_envs

# envs = build_1t1r_env()
# state_dim, action_dim = envs.observation_dim, envs.action_dim
#
# memristor_config, transistor_config, simulate_config, array_config = load_1t1r_config()
# array_config['sampler_num'] = 1
# eval_envs = build_1t1r_env(
#     memristor_config=memristor_config,
#     transistor_config=transistor_config,
#     simulate_config=simulate_config,
#     array_config=array_config,
# )


def train(config_dict: dict):
    train_flag, array_kwargs = config_dict["train_flag"], config_dict["array_kwargs"]

    envs, eval_envs = build_envs(**array_kwargs)
    state_dim, action_dim = envs.observation_dim, envs.action_dim
    train_data_dict = dict()
    actor = PPOActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 32],
    )
    critic = PPOCritic(
        state_dim=state_dim,
        hidden_dims=[64, 32],
    )
    agent = PPO(
        actor=actor,
        critic=critic,
        envs=envs,
        gamma=0.99,
        lambda_gae=0.95,
        lr=6e-5,
        batch_size=batch_size,
        ratio_clip=0.25,
        lambda_entropy=0.01,
    )

    r""" setting 2 """
    explore_steps = 2048
    repeat_times = 8
    total_train_steps = 5e6
    eval_max_step = 5000
    eval_per_step = int(2e4)

    eval_counter = 0
    max_rewards = -100
    min_steps = eval_max_step

    log_dir = str(root / fr"logs/{current_time}_{train_flag}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_dir = str(root / fr"checkpoints/{current_time}_{train_flag}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    agent.reset_total_steps()

    while True:
        with torch.no_grad():
            agent.interact_with_envs(interact_steps=explore_steps)

        update_times = int(agent.buffer_size * repeat_times / batch_size)
        critic_loss, actor_loss = 0, 0
        for idx in range(update_times):
            c_loss, a_loss = agent.update_network()
            critic_loss += c_loss / update_times
            actor_loss += a_loss / update_times
        print('\t>>> avg critic loss: {}; avg actor loss: {}.'.format(critic_loss, actor_loss))
        if agent.total_steps > eval_counter * eval_per_step:
            episode_rewards, episode_steps, episode_success_steps, avg_reward, avg_step = agent.evaluate(
                eval_envs=eval_envs,
                max_steps=eval_max_step,
                log_dir=log_dir,
            )
            update_persist_train_data(
                log_dir=log_dir,
                data_dict=train_data_dict,
                step_count=eval_counter * eval_per_step,
                episode_rewards=episode_rewards,
                episode_steps=episode_steps,
                episode_success_steps=episode_success_steps,
            )
            plot_status(
                log_dir=log_dir,
                extra_prefix=f"ppo_bit_{envs.bit_width}"
            )
            eval_counter += 1

            if avg_reward > max_rewards or avg_step < min_steps:
                max_rewards = avg_reward
                min_steps = avg_step
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    "actor_step{:d}_r{:.0f}_step{:.1f}.pkl".format(agent.total_steps, avg_reward, avg_step)
                )
                torch.save(agent.actor, checkpoint_path)
        if agent.total_steps > total_train_steps:
            break


def multi_core_train():
    memristor_relative_noise_range = np.arange(0.5, 0.51, 0.05)
    bit_width_range = list(range(6, 7))
    config_list = []

    bit_noise = [(4, 0.3), (4, 0.4), (4, 0.5), (5, 0.3), (5, 0.4), (5, 0.5), (6, 0.3), (6, 0.4)]

    # for bit_width in bit_width_range:
    #     for noise in memristor_relative_noise_range:

    for bit_width, noise in bit_noise:
        noise_percent = int(noise * 100)
        train_flag = f"bit-{bit_width}_noise-{noise_percent}"
        array_kwargs = {
            "bit_width": bit_width,
            "memristor_relative_noise": noise,
        }
        config_dict = {"train_flag": train_flag, "array_kwargs": array_kwargs}
        config_list.append(config_dict)

    with mp.Pool(processes=4) as pool:
        pool.map(train, config_list)

if __name__ == "__main__":
    multi_core_train()
