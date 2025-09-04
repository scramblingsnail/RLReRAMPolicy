import torch
import logging
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from gym.vector import AsyncVectorEnv

from .actor_critic_base import ActorCriticBase
from ..net import TD3Actor, TwinCritic
from ..replay_buffer import ReplayBuffer
from ..train import init_optimizer


class TD3:
    r"""
    TD3 (Twin delayed DDPG) is a modified version of DDPG(deep deterministic policy gradient).
        refer to https://arxiv.org/pdf/1802.09477.pdf
        On the basis of DDPG, TD3 tries to ease the problem of overestimation by introducing double Q networks and their
        target networks. The minimum value of the two Q network is utilized to calculate the temporal difference error.
        Besides, TD3 adopts a delayed scheme for the updating of policy and the soft updating of target network for the
        sake of waiting for the stabilization of value estimation error. Other tricks: TD3 assumes that similar actions
        should have similar Q values, so a clipped noise is added up to the generated action when calculating the target Q
        value (intrinsically it tries to introduce more randomness to generalize the deterministic policy).
    """
    def __init__(
            self,
            actor: TD3Actor,
            critic: TwinCritic,
            buffer: ReplayBuffer,
            envs: AsyncVectorEnv,
            update_momentum: float,
            gamma: float,
            lr: float,
            delayed_update_period: int,
            clip_grad_norm: float = 3.0,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = buffer
        self.buffer.device = self.device

        self.lr = lr
        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.AdamW(params=self.actor.parameters(), lr=self.lr)
        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.AdamW(params=self.critic.parameters(), lr=self.lr)

        self.envs = envs
        self.action_dim = self.actor.action_dim
        self.state_dim = self.actor.state_dim
        self.state_bot = self.envs.observation_space.__dict__['low'][0]
        self.state_top = self.envs.observation_space.__dict__['high'][0]
        self.action_bot = self.envs.action_space.__dict__['low'][0]
        self.action_top = self.envs.action_space.__dict__['high'][0]
        self.last_state = None

        self.clip_grad_norm = clip_grad_norm
        self.gamma = min(1., max(0., gamma))
        self.update_momentum = min(1., max(0., update_momentum))
        # self.mse_loss = nn.MSELoss()
        self.mse_loss = nn.SmoothL1Loss(reduction="mean")

        self.delayed_update_period = delayed_update_period
        self.total_steps = 0
        self.update_count = 0

    def reset_total_steps(self):
        self.total_steps = 0

    def real_action(self, action: np.ndarray):
        r""" action: [-1, 1] """
        return (action + 1) / 2 * (self.action_top - self.action_bot) + self.action_bot

    def normalize_state(self, real_state: np.ndarray):
        r"""
        return: state: [-1, 1]
        """
        return (real_state - self.state_bot) / (self.state_top - self.state_bot) * 2 - 1

    @staticmethod
    def get_episode_rewards(rewards: np.ndarray, terminals: np.ndarray):
        episode_rewards = []
        episode_lens = []

        reward_sum = None
        length = 0

        for sample_idx in range(rewards.shape[0]):
            for idx in range(rewards.shape[1]):
                if terminals[sample_idx, idx]:
                    if reward_sum is not None:
                        reward_sum += rewards[sample_idx, idx]
                        episode_rewards.append(reward_sum)
                        episode_lens.append(length)
                    reward_sum = 0
                    length = 0
                else:
                    if reward_sum is None:
                        reward_sum = 0
                    reward_sum += rewards[sample_idx, idx]
                    length += 1
        return episode_rewards, episode_lens

    def interact_with_envs(self, interact_steps: int, random_mode: bool=False):
        r"""
        interact with multiple envs asynchronously.
        if an env done, the env automatically reset and return the reset state.

        get At, Rt, St+1, Tt+1

        return states actions rewards terminals
        """
        assert self.envs.num_envs == self.buffer.num_samplers
        states = np.zeros((self.buffer.num_samplers, interact_steps, self.state_dim), dtype=float)
        actions = np.zeros((self.buffer.num_samplers, interact_steps, self.action_dim), dtype=float)
        rewards = np.zeros((self.buffer.num_samplers, interact_steps), dtype=float)
        terminals = np.zeros((self.buffer.num_samplers, interact_steps), dtype=bool)
        truncated_buffer = np.zeros((self.buffer.num_samplers, interact_steps), dtype=bool)

        if self.last_state is None:
            self.last_state = self.envs.reset()[0]
            self.last_state = self.normalize_state(self.last_state)
            self.buffer.reset_buffer(init_states=self.last_state)
        current_state = self.last_state
        logging.info('Interact with envs, step{:d} to step{:d}; '
                     'random_mode: {}'.format(self.total_steps, self.total_steps + interact_steps, random_mode))
        self.total_steps += interact_steps * self.buffer.num_samplers

        for idx in range(interact_steps):
            if random_mode:
                action = np.random.uniform(-1, 1, size=(self.buffer.num_samplers, self.action_dim))
                real_action = self.real_action(action)
            else:
                current_state = torch.tensor(current_state, dtype=self.buffer.sample_dtype, device=self.device)
                action = self.actor.choose_noisy_action(current_state).cpu().detach().numpy()
                # real action
                real_action = self.real_action(action)

            current_state, reward, terminal, truncated, _ = self.envs.step(real_action)
            # normalized state
            current_state = self.normalize_state(current_state)
            # align with gym 0.24.0: truncate also mean terminals
            terminal = np.bitwise_or(terminal, truncated)
            # record
            states[:, idx] = current_state
            actions[:, idx] = action
            rewards[:, idx] = reward
            terminals[:, idx] = terminal
            truncated_buffer[:, idx] = truncated

        self.last_state = current_state
        self.buffer.put(states=states, actions=actions, rewards=rewards, terminals=terminals)
        episode_rewards, episode_lens = self.get_episode_rewards(rewards=rewards, terminals=terminals)
        truncated_times = np.sum(truncated_buffer)
        terminal_times = np.sum(terminals)
        success_times = terminal_times - truncated_times
        return episode_rewards, episode_lens, success_times, truncated_times

    def critic_objective(self, states, actions, rewards, terminals, next_states) -> torch.Tensor:
        r"""
        Calculate the training objective of the critic network.
        Note that clipped noise is added to the next_action that is suggested by the actor_target.

        Args:
            states (torch.Tensor): (batch_size, state_dim), the states sampled from buffer.
            actions (torch.Tensor): (batch_size, action_dim), the corresponding actions sampled from buffer
            rewards (torch.Tensor): (batch_size, ), the corresponding rewards sampled from buffer
            terminals (torch.Tensor): (batch_size, ), the terminals sampled from buffer,
                indicating that whether the next state is the terminal of a trajectory.
            next_states (torch.Tensor): (batch_size, state_dim), the corresponding next states sampled from buffer.

        Returns:
            obj(torch.Tensor): the calculated critic objective.
        """
        with torch.no_grad():
            # q_label = rewards + (1 - terminals) * gamma * (next_q_val - alpha * log_pob(next_action_sampled))
            # random sampling next transition alternate to expectation. An approximation in RL
            next_action_sampled = self.actor_target.choose_target_action(next_states)
            next_q_val = self.critic_target.min_q(state=next_states, action=next_action_sampled).squeeze(1)
            q_label = rewards + (1 - terminals) * self.gamma * next_q_val

        # MSE
        q1, q2 = self.critic.twin_q(state=states, action=actions)
        q1, q2 = q1.squeeze(1), q2.squeeze(1)
        obj = self.mse_loss(q1, q_label) + self.mse_loss(q2, q_label)
        return obj

    def soft_update_critic_target(self):
        r"""
        Soft update the target critic network.
        """
        with torch.no_grad():
            for name in self.critic_target.state_dict().keys():
                self.critic_target.state_dict()[name].copy_(self.update_momentum * self.critic.state_dict()[
                    name] + (1 - self.update_momentum) * self.critic_target.state_dict()[name])

    def soft_update_actor_target(self):
        r"""
        Soft update the target actor network.
        """
        with torch.no_grad():
            for name in self.actor_target.state_dict().keys():
                self.actor_target.state_dict()[name].copy_(self.update_momentum * self.actor.state_dict()[
                    name] + (1 - self.update_momentum) * self.actor_target.state_dict()[name])

    def optimizer_update(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor):
        r"""
        Use the optimizer to update parameters based on the loss.

        Args:
            optimizer(torch.optim.Optimizer): the employed optimizer.
            loss(torch.Tensor): the calculated loss.

        Returns:
            None.

        Note:
            The calculated grads are clipped using `clip_grad_norm_`.
        """
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def update_network(self):
        r"""
        sample from buffer and compute gradients, update critic and actor
        """
        self.update_count += 1
        states, actions, rewards, terminals, next_states = self.buffer.get_frame_data()
        # update critic
        critic_loss = self.critic_objective(states, actions, rewards, terminals, next_states)
        self.optimizer_update(self.critic_optimizer, critic_loss)
        critic_loss = critic_loss.detach().cpu().item()

        if self.update_count % self.delayed_update_period == 0:
            actions_with_grad = self.actor.choose_action(state=states)
            actor_loss = - self.critic.mean_q(state=states, action=actions_with_grad).mean()
            self.optimizer_update(optimizer=self.actor_optimizer, loss=actor_loss)

            # soft update target critic and actor
            self.soft_update_critic_target()
            self.soft_update_actor_target()
            actor_loss = actor_loss.detach().cpu().item()
            return critic_loss, actor_loss
        return critic_loss, 0

    def evaluate(self, eval_envs, max_steps, log_dir, eval_episodic_num = None, save_data: bool = True):
        r"""
        evaluate the trained network's performance: until terminal or exceeding max steps.

        Args:
            eval_envs (gym.vector.AsyncVectorEnv): envs for evaluating
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
        torch.set_grad_enabled(False)
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
        failed_state_path = rf"{log_dir}/eval_{self.total_steps}_failed.txt"
        failed_state = np.zeros_like(state)

        print('Evaluating multi envs ...')
        for step_idx in range(max_steps):
            # normalize state
            state = self.normalize_state(state)
            state = torch.tensor(state, dtype=self.buffer.sample_dtype, device=self.device)
            action = self.actor.choose_action(state=state)
            action = action.detach().cpu().numpy()
            # real_action
            real_action = self.real_action(action)
            # interact with envs
            state, reward, terminal, truncated, _ = eval_envs.step(real_action)
            rewards[:, step_idx] = reward
            terminals[:, step_idx] = terminal
            all_truncated[:, step_idx] = truncated
            # record failed state
            if truncated:
                failed_start_states.append(failed_state)
                failed_state = state

            success = np.bitwise_and(np.bitwise_not(truncated), terminal)
            # align with gym 0.24.0: truncate also mean terminals
            terminal = np.bitwise_or(terminal, truncated)
            episodic_num += np.sum(terminal)
            # for envs that terminate:
            # add the final rewards.
            accumulate_rewards += reward
            episodic_rewards += list(accumulate_rewards[terminal])
            episodic_steps += [int(max(step_idx - start_idx, 1)) for start_idx in start_indices[terminal]]
            success_episodic_steps += [int(max(step_idx - start_idx, 1)) for start_idx in start_indices[success]]
            # Reset record
            accumulate_rewards[terminal] = 0
            start_indices[terminal] = step_idx

            if eval_episodic_num and episodic_num >= eval_episodic_num:
                break

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

        log_str = (
            "Evaluate at step{:d}"
            "\t-- average reward: {:.4f}\t -- average steps num: {:.2f}\t -- average success steps num: {:.2f}\t"
            "-- episodic num: {:d} -- success rate: {:.2f}".format(
                self.total_steps,
                avg_reward,
                avg_step,
                avg_success_step,
                len(episodic_rewards),
                success_rate,
            )
        )
        logging.info(log_str)
        print(log_str)

        # if failed_start_states:
        #     failed_start_states = np.concatenate(failed_start_states, axis=0)
        #     np.savetxt(failed_state_path, failed_start_states)

        torch.set_grad_enabled(True)
        return episodic_rewards, episodic_steps, success_episodic_steps, avg_reward, avg_step
