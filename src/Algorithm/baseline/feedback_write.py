import torch
import h5py
import os
import numpy as np

from typing import Tuple, List, Union
from src.MemristorENV import OneTOneREnv


class FeedBackWriter:
    r"""
    A writer of feedback writing.

    Args:
        env (OneTOneREnv): the memristor env.
        neg_vr_step (float): the change step of negative vr.
            it is supposed to be negative.
        pos_vg_step (float): the change step of positive vg.
            it is supposed to be positive.
        pos_vr_step (float): the change step of positive vr.
            it is supposed to be positive.
    """
    def __init__(self, env: OneTOneREnv, neg_vr_step: float, pos_vg_step: float, pos_vr_step: float):
        self.env = env
        self.state_bot = self.env.observation_space.__dict__['low'][0]
        self.state_top = self.env.observation_space.__dict__['high'][0]
        self.action_bot = self.env.action_space.__dict__['low'][0]
        self.action_top = self.env.action_space.__dict__['high'][0]
        self.sampler_num = self.env.sampler_num
        self.action_dim = self.env.action_dim
        self.neg_vr_step = neg_vr_step
        self.pos_vg_step = pos_vg_step
        self.pos_vr_step = pos_vr_step
        self.default_neg_vr = -0.5
        self.default_neg_vg = self.env.actual_vg_reset_range[1]
        self.default_pos_vr = 2
        self.default_pos_vg = self.env.actual_vg_set_range[0]
        self.max_pos_vg = 0.75
        self.max_pos_vr = self.env.actual_vds_set_range[1]
        self.min_neg_vr = self.env.actual_vds_reset_range[0]
        self.current_neg_vr = np.ones(self.sampler_num) * self.default_neg_vr
        self.current_neg_vg = np.ones(self.sampler_num) * self.default_neg_vg
        self.current_pos_vr = np.ones(self.sampler_num) * self.default_pos_vr
        self.current_pos_vg = np.ones(self.sampler_num) * self.default_pos_vg
        self.current_observations = self.env.read_all_samplers()
        self.last_observations = self.current_observations
        self.current_actions = np.zeros((self.sampler_num, self.action_dim))
        self.truncated = np.zeros(self.sampler_num)
        print('state_bot:', self.state_bot)
        print('state_top:', self.state_top)
        print('action_bot:', self.action_bot)
        print('action_top:', self.action_top)
        print('default_neg_vg:', self.default_neg_vg)
        print('default_pos_vg:', self.default_pos_vg)
        print('max_pos_vg:', self.max_pos_vg)
        print('max_pos_vr:', self.max_pos_vr)
        print('min_neg_vr:', self.min_neg_vr)
        self.start_write = False
        self.collected_data = None

    def select_vg(self, sample_idx, current_g, target_g, no_move_count):
        r"""
        The vg strategy.

        """
        if current_g < target_g:
            # set
            vg = self.current_pos_vg[sample_idx] + self.pos_vg_step * (no_move_count > 0)

            vg = min(vg, self.max_pos_vg)
            self.current_pos_vg[sample_idx] = vg
        else:
            # reset
            vg = self.current_neg_vg[sample_idx]
            # set current pos vg value to default pos vg.
            self.current_pos_vg[sample_idx] = self.default_pos_vg
        return vg

    def select_vr(self, sample_idx, current_g, target_g, no_move_count):
        r"""
        The vr strategy.

        """
        if current_g < target_g:
            # set
            if current_g < 1.2e-2:
                vr = self.max_pos_vr
            else:
                vr = self.current_pos_vr[sample_idx] + self.pos_vr_step * (no_move_count > 0)

            vr = min(vr, self.max_pos_vr)
            self.current_pos_vr[sample_idx] = vr
            # set current neg vr value to default neg vr.
            self.current_neg_vr[sample_idx] = self.default_neg_vr
        else:
            # reset
            vr = self.current_neg_vr[sample_idx] + self.neg_vr_step * (no_move_count > 0)
            vr = max(vr, self.min_neg_vr)
            self.current_neg_vr[sample_idx] = vr
            # set current pos vr value to default pos vr.
            self.current_pos_vr[sample_idx] = self.default_pos_vr
        return vr

    def select_action(self):
        r"""
        Select action according to the observations.
        The format of current observations: (current_conductance, target_conductance, no_move_count)
        """
        for sample_idx in range(self.sampler_num):
            current_g, target_g, no_move_count = tuple(self.current_observations[sample_idx])
            vg = self.select_vg(sample_idx=sample_idx, current_g=current_g, target_g=target_g,
                                no_move_count=no_move_count)
            vr = self.select_vr(sample_idx=sample_idx, current_g=current_g, target_g=target_g,
                                no_move_count=no_move_count)
            self.current_actions[sample_idx, 0] = vg
            self.current_actions[sample_idx, 1] = vr

    def normalize_state(self, real_state: np.ndarray):
        r"""
        return: state: [-1, 1]
        """
        return (real_state - self.state_bot) / (self.state_top - self.state_bot) * 2 - 1

    def normalize_action(self, real_action: np.ndarray):
        r"""
        return: state: [-1, 1]
        """
        return (real_action - self.action_bot) / (self.action_top - self.action_bot) * 2 - 1

    def select_action_for_eval(self):
        r"""
        Select action according to the observations.
        The format of current observations: (current_conductance, target_conductance, vg_set, vds_set, vg_reset, vds_reset)
        """
        for sample_idx in range(self.sampler_num):
            current_g, target_g, _, _, _, _ = tuple(self.current_observations[sample_idx])

            if current_g < target_g:
                # set
                self.current_actions[sample_idx, 0] = self.pos_vg_step
                self.current_actions[sample_idx, 1] = self.pos_vr_step
                # print(f"Sample {sample_idx}: Current -- {current_g}; Target -- {target_g}; Set; delta(Vr, Vg) -- {self.current_actions[sample_idx]}")
            else:
                # reset
                self.current_actions[sample_idx, 0] = 0
                self.current_actions[sample_idx, 1] = self.neg_vr_step

                # print(f"Sample {sample_idx}: Current -- {current_g}; Target -- {target_g}; Reset; delta(Vr, Vg) -- {self.current_actions[sample_idx]}")

    def evaluate(self, max_steps: int, eval_env: OneTOneREnv = None) -> Tuple[List[float], List[Union[int, None]]]:
        r"""
        Interact with multiple envs asynchronously.
        if an env done, the env automatically reset and return the reset state.

        Args:
            eval_env (OneTOneREnv): Env for evaluation.
            max_steps (int): Maximum steps

        Returns:
            Episodic rewards and episodic steps. If an episodic is truncated, its step num is `None`.

        """
        eval_env = eval_env or self.env
        env_num = eval_env.num_envs

        rewards = np.zeros((env_num, max_steps), dtype=float)
        terminals = np.zeros((env_num, max_steps), dtype=bool)
        truncated_buffer = np.zeros((env_num, max_steps), dtype=bool)

        accumulate_rewards = np.zeros(env_num, dtype=float)
        start_indices = np.zeros(env_num, dtype=int)

        episodic_rewards = list()
        episodic_steps = list()

        for step_idx in range(max_steps):
            if (step_idx + 1) % 500 == 0:
                print(f"{step_idx + 1} steps finished.")
            self.select_action_for_eval()
            step_states, step_rewards, step_terminals, step_truncated, _ = eval_env.step(
                do_actions=self.current_actions,
            )
            # print("states: ", step_states)
            frames = self.env.render()
            self.current_observations = step_states

            # record
            # TODO: record rewards of each episodic; record step num of each episodic (normally terminate, no truncate)
            rewards[:, step_idx] = step_rewards
            terminals[:, step_idx] = step_terminals
            truncated_buffer[:, step_idx] = step_truncated

            accumulate_rewards += step_rewards

            for env_idx in range(env_num):
                if step_terminals[env_idx]:
                    episodic_rewards.append(accumulate_rewards[env_idx])
                    accumulate_rewards[env_idx] = 0
                    if step_truncated[env_idx]:
                        step_num = max(1, step_idx - start_indices[env_idx])
                        episodic_steps.append((step_num, None))
                    else:
                        step_num = max(1, step_idx - start_indices[env_idx])
                        episodic_steps.append(step_num)
                    start_indices[env_idx] = step_idx

        return episodic_rewards, episodic_steps

    def record_valuable_data(self):
        r"""
        record the valuable data pair: last_states -> actions according to no_move_count of current_states.
        Do this after doing actions and update observations.
        """
        if not self.start_write:
            return None

        data_buffer = []
        for sample_idx in range(self.sampler_num):
            sample_data = np.zeros(2 + self.action_dim)
            no_move_count = self.current_observations[sample_idx, 2]
            if no_move_count == 0 and not self.truncated[sample_idx]:
                # print('state before norm: ', self.last_observations[sample_idx, :2])
                norm_states = self.normalize_state(self.last_observations[sample_idx, :2])
                # print('state after norm: ', norm_states)
                # print('action before norm: ', self.current_actions[sample_idx])
                norm_actions = self.normalize_action(self.current_actions[sample_idx])
                if (norm_states > 1).any() or (norm_actions >= 1).any() or (norm_states < -1).any() or (norm_actions <= -1).any():
                    raise ValueError('Value exceed [-1, 1].')
                # print('action after norm: ', norm_actions)
                sample_data[:2] = norm_states
                sample_data[2:] = norm_actions
                data_buffer.append(sample_data)
        if len(data_buffer) > 0:
            data_buffer = np.stack(data_buffer, axis=0)
            if self.collected_data is None:
                self.collected_data = data_buffer
            else:
                self.collected_data = np.concatenate((self.collected_data, data_buffer), axis=0)

    def write(self):
        # 判断当前方向，将相反方向的操作重置；根据no_move_count给当前方向的操作加上对应的。
        self.record_valuable_data()
        self.select_action()
        self.last_observations = self.current_observations
        # print('>>> actions:', self.current_actions)
        states, rewards, terminals, truncated, _ = self.env.step(do_actions=self.current_actions)
        self.truncated = truncated
        # print('>>> \tfinished: {} truncated: {}; after observations: '.format(terminals, truncated), states)
        self.current_observations = states

    def collect_data(self, data_num, save_dir, save_name):
        r"""
        Try to collect valid training data of size `data_num`.

        Collected data:
            size: (data_num, 4)

            - along the dimension 1: (current_g, target_g, vg, vr)
        """
        # print(self.current_observations)
        save_path = os.path.join(save_dir, save_name)
        print('>>> Supervised training data for actor net is saved to: \n\t', save_path)
        if os.path.exists(save_path):
            print('>>> \talready exists.')
            return save_path

        self.write()
        self.start_write = True
        data_count = 0
        while self.collected_data is None or self.collected_data.shape[0] < data_num:
            self.write()
            if self.collected_data is not None and self.collected_data.shape[0] > data_count:
                print('{} piece of data collected.'.format(self.collected_data.shape[0]))
                data_count += 1000

        neg_vr_step = self.neg_vr_step
        pos_vg_step = self.pos_vg_step
        pos_vr_step = self.pos_vr_step
        with h5py.File(save_path, 'w') as data_f:
            data_f.create_dataset(name='data', data=self.collected_data)

        return save_path
