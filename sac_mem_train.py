import torch
import h5py
import multiprocessing as mp
import logging
import time
import gym
import os

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from multiprocessing import Queue, Process
from src.utils import set_logger, make_dirs, plot_status, update_persist_train_data
from src.gym_envs.vector_env import make_envs
from src.Algorithm.rl import SAC, SACActor, TwinCritic
from src.Algorithm.replay_buffer import ReplayBuffer
from src.MemristorENV import build_1t1r_env, load_1t1r_config


set_logger()
torch.manual_seed(2025)
np.random.seed(2025)

root = Path(__file__).parent
current_time = time.strftime('%m-%d-%H-%M')

r""" setting 1 """
batch_size = 64
buffer_size = int(2e5)
env_max_step = 12358


def build_mlp(dims: [int], activation=None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


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
# observation_dim, action_dim = envs.observation_dim, envs.action_dim
# state_dim = observation_dim
#
# memristor_config, transistor_config, simulate_config, array_config = load_1t1r_config()
# array_config['sampler_num'] = 1
# eval_env = build_1t1r_env(memristor_config=memristor_config, transistor_config=transistor_config,
#                           simulate_config=simulate_config, array_config=array_config)
# eval_envs = build_1t1r_env(memristor_config=memristor_config, transistor_config=transistor_config,
#                            simulate_config=simulate_config, array_config=array_config)
#
# init_states = envs.reset()[0]
# print('state_dim: {}; action_dim: {}'.format(state_dim, action_dim))


def train(config_dict: dict):
    train_flag, array_kwargs = config_dict["train_flag"], config_dict["array_kwargs"]

    envs, eval_envs = build_envs(**array_kwargs)
    init_states = envs.reset()[0]
    state_dim, action_dim = envs.observation_dim, envs.action_dim
    train_data_dict = dict()

    actor_net = build_mlp([state_dim, 64, 32, action_dim * 2])

    critic_net = build_mlp([action_dim + state_dim, 64, 32, 2])
    actor = SACActor(action_dim=action_dim, state_dim=state_dim, actor_net=actor_net, random_init=False)
    critic = TwinCritic(q_net=critic_net, action_dim=action_dim, state_dim=state_dim)
    buffer = ReplayBuffer(num_samplers=envs.sampler_num, buffer_size=buffer_size, batch_size=batch_size,
                          observation_dim=state_dim, action_dim=action_dim, padding_action=np.zeros(action_dim),
                          init_states=init_states, max_length=1)

    agent = SAC(actor=actor, critic=critic, buffer=buffer, envs=envs, update_momentum=5e-3, gamma=0.99, lr=6e-5,
                batch_norm_momentum=0.1)

    r""" setting 2 """
    explore_steps = 512
    repeat_times = 1
    random_explore_times = 16
    total_train_steps = 5e6  # int(8e4)

    eval_max_step = 5000
    eval_per_step = int(2e4)

    eval_counter = 0
    max_rewards = -100
    min_steps = eval_max_step
    for random_idx in range(random_explore_times):
        # random explore
        with torch.no_grad():
            interact_results = agent.interact_with_envs(interact_steps=explore_steps, random_mode=True)

    log_dir = str(root / fr"logs/{current_time}_{train_flag}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_dir = str(root / fr"checkpoints/{current_time}_{train_flag}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # reset step record
    agent.reset_total_steps()

    while True:
        # explore
        with torch.no_grad():
            infos = agent.interact_with_envs(interact_steps=explore_steps)
        # update
        update_times = buffer.add_data_num * repeat_times

        critic_loss, actor_loss, alpha_loss = 0, 0, 0
        for idx in range(update_times):
            c_loss, a_loss, al_loss = agent.update_network()
            critic_loss += c_loss / update_times
            actor_loss += a_loss / update_times
            alpha_loss += al_loss / update_times
        print('\t>>> avg critic loss: {}; avg actor loss: {}; avg alpha loss : {}.'.format(critic_loss, actor_loss,
                                                                                           alpha_loss))
        # evaluate and save
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
                extra_prefix=f"sac_bit_{envs.bit_width}_var_005"
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

        # judge break
        if agent.total_steps > total_train_steps:
            break


def multi_core_train():
    memristor_relative_noise_range = np.arange(0.3, 0.51, 0.1)
    bit_width_range = list(range(4, 5))
    config_list = []
    for bit_width in bit_width_range:
        for noise in memristor_relative_noise_range:
            noise_percent = int(noise * 100)
            train_flag = f"bit-{bit_width}_noise-{noise_percent}"
            array_kwargs = {
                "bit_width": bit_width,
                "memristor_relative_noise": noise,
            }
            config_dict = {"train_flag": train_flag, "array_kwargs": array_kwargs}
            config_list.append(config_dict)

    with mp.Pool(processes=3) as pool:
        pool.map(train, config_list)


if __name__ == "__main__":
    multi_core_train()
