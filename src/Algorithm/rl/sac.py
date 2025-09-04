import torch
import torch.nn as nn
import numpy as np
import logging
import h5py
import time

from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from gym.vector import AsyncVectorEnv
from gym import Env
from .actor_critic_base import ActorCriticBase
from ..net import SACActor, TwinCritic
from ..replay_buffer import ReplayBuffer


class SAC(nn.Module):
    r"""
    refer to:
        version 1: https://arxiv.org/abs/1801.01290
        version 2: https://arxiv.org/abs/1812.05905
    SAC(Soft actor-critic) introduce the maximum entropy objective to balance exploration and exploitation. It assumes
    that the optimal policy is close to a boltzmann distribution with energy -Q. In its second release, the entropy
    objective is re-described as a constraint. Through lagrange multiplier methods, the temperature(scale factor of
    entropy) can be adjusted automatically during iteration(actually, the second version restrict the value of entropy
    around a preset value (- dim(action space)) softly).
    """
    def __init__(self, actor: SACActor, critic: TwinCritic, buffer: ReplayBuffer, envs: AsyncVectorEnv,
                 update_momentum: float, batch_norm_momentum: float, gamma: float, lr: float,
                 alpha_log_lower: float = -16, alpha_log_upper: float = 2, target_entropy: float = None,
                 clip_grad_norm: float = 3.0):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = buffer
        self.buffer.device = self.device
        self.envs = envs
        self.lr = lr
        self.actor = actor.to(self.device)
        self.teacher_actor = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.AdamW(params=self.actor.parameters(), lr=self.lr)
        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.AdamW(params=self.critic.parameters(), lr=self.lr)
        # alpha: parameter for the importance of entropy
        self.alpha_log = nn.Parameter(torch.tensor(-1, dtype=self.buffer.sample_dtype, device=self.device))
        # TODO, here refer to him.
        self.alpha_optimizer = torch.optim.AdamW(params=(self.alpha_log,), lr=self.lr * 4)
        self.alpha_log_lower, self.alpha_log_upper = alpha_log_lower, alpha_log_upper
        self.action_dim = self.actor.action_dim
        self.state_dim = self.actor.state_dim
        self.state_bot = self.envs.observation_space.__dict__['low'][0]
        self.state_top = self.envs.observation_space.__dict__['high'][0]
        self.action_bot = self.envs.action_space.__dict__['low'][0]
        self.action_top = self.envs.action_space.__dict__['high'][0]
        print('state_bot:', self.state_bot)
        print('state_top:', self.state_top)
        print('action_bot:', self.action_bot)
        print('action_top:', self.action_top)
        assert self.state_bot.shape[0] == self.state_top.shape[0] == self.state_dim
        assert self.action_bot.shape[0] == self.action_top.shape[0] == self.action_dim
        # target_entropy: default: - action_dim
        self.clip_grad_norm = clip_grad_norm
        self.target_entropy = float(-self.action_dim) if target_entropy is None else target_entropy
        self.envs = envs
        self.last_state = None
        self.update_momentum = min(1., max(0., update_momentum))
        self.batch_norm_momentum = batch_norm_momentum
        self.gamma = min(1., max(0., gamma))
        # self.mse_loss = nn.MSELoss()
        self.mse_loss = nn.SmoothL1Loss(reduction="mean")
        self.total_steps = 0
        print('actor lr: ', self.actor_optimizer.param_groups[0]['lr'])
        print('critic lr: ', self.critic_optimizer.param_groups[0]['lr'])
        print('alpha lr: ', self.alpha_optimizer.param_groups[0]['lr'])

    def reset_total_steps(self):
        logging.info("Reset total_steps.")
        self.total_steps = 0

    def real_action(self, action: np.ndarray):
        r""" action: [-1, 1] """
        return (action + 1) / 2 * (self.action_top - self.action_bot) + self.action_bot

    def normalize_state(self, real_state: np.ndarray):
        r"""
        return: state: [-1, 1]
        """
        return (real_state - self.state_bot) / (self.state_top - self.state_bot) * 2 - 1

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
            # TODO: disable state normalize
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
                action = self.actor.choose_action(current_state).cpu().detach().numpy()
                # real action
                real_action = self.real_action(action)

            # TODO: Debug Sepecific for memristor is the action direction correct?
            # # should set
            # less = torch.lt(current_state[:, 0], current_state[:, 1]).detach()
            # # should reset
            # larger = torch.ge(current_state[:, 0], current_state[:, 1]).detach()
            # real_action_torch = torch.tensor(real_action, dtype=self.buffer.sample_dtype, device=self.device)
            # wrong_action_count += torch.lt(real_action_torch[less, 1], 0).sum() + torch.ge(real_action_torch[larger, 1], 0).sum()

            current_state, reward, terminal, truncated, _ = self.envs.step(real_action)

            # normalized state
            # TODO: disable state normalize
            current_state = self.normalize_state(current_state)

            # TODO: align with gym 0.24.0: truncate also mean terminals
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

    def soft_update_critic_target(self):
        r"""
        soft update the target critic network.
        """
        with torch.no_grad():
            for name in self.critic_target.state_dict().keys():
                self.critic_target.state_dict()[name].copy_(self.update_momentum * self.critic.state_dict()[
                    name] + (1 - self.update_momentum) * self.critic_target.state_dict()[name])
    def critic_objective(self, states, actions, rewards, terminals, next_states) -> torch.Tensor:
        r"""
        Calculate the objective of the critic network.

        The objective is the mse_loss(q1, q_label) + mse_loss(q2, q_label). q_label is calculated as follows:
            >- ```q_label = rewards + (1 - terminals) * gamma * (next_q_val - alpha * log_pob(next_action_sampled))```
            >   - `next_action_sampled`, `action_log_prob` is selected by the actor net
            >   - `next_q_val` is calculated by the target critic net, based on the `next_states` and `next_action_sampled`

        Args:
            states(torch.Tensor): (batch_size, state_dim), the states sampled from buffer.
            actions(torch.Tensor): (batch_size, action_dim), the corresponding actions sampled from buffer
            rewards(torch.Tensor): (batch_size, ), the corresponding rewards sampled from buffer
            terminals(torch.Tensor): (batch_size, ), the terminals sampled from buffer,
                indicating that whether the next state is the terminal of a trajectory.
            next_states(torch.Tensor): (batch_size, state_dim), the corresponding next states sampled from buffer.

        Returns:
            obj(torch.Tensor): the calculated critic objective.
        """
        # (batch_size, 1)
        with torch.no_grad():
            # q_label = rewards + (1 - terminals) * gamma * (next_q_val - alpha * log_pob(next_action_sampled))
            # random sampling next transition alternate to expectation. An approximation in RL
            next_action_sampled, action_log_prob = self.actor.get_action_logprob(next_states)
            next_q_val = self.critic_target.min_q(state=next_states, action=next_action_sampled).squeeze(1)
            alpha = torch.exp(self.alpha_log).detach()
            q_label = rewards + (1 - terminals) * self.gamma * (next_q_val - alpha * action_log_prob)
        # MSE
        q1, q2 = self.critic.twin_q(state=states, action=actions)
        q1, q2 = q1.squeeze(1), q2.squeeze(1)
        obj = self.mse_loss(q1, q_label) + self.mse_loss(q2, q_label)
        if torch.isnan(obj).any() or torch.isinf(obj).any():
            print('NAN in critic loss: ', obj)
            # print('actions: {}'.format(actions))
            # print('q1: {}, q2: {}, q_label: {}, action_log_prob: {}'.format(q1, q2, q_label, action_log_prob))
        assert not torch.isnan(obj).any()
        return obj

    def optimizer_update(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor):
        r"""
        using the optimizer to update parameters based on the loss.

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

        assert not torch.isnan(loss).any(), print('NAN exist in loss.')

        with torch.no_grad():
            for name, p in self.actor.named_parameters():
                assert not torch.isnan(p).any(), print('Before step, NAN exist in {}.'.format(name), p)

            for name, p in self.critic.named_parameters():
                assert not torch.isnan(p).any(), print('Before step, NAN exist in {}.'.format(name), p)

        nan_num = 0
        for name, p in self.actor.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print("NAN exists in grad of {}".format(name))
                nan_num += 1

        for name, p in self.critic.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print("NAN exists in grad of {}".format(name))
                nan_num += 1

        if nan_num > 0:
            print('loss: ', loss)

        assert nan_num == 0

        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            for name, p in self.actor.named_parameters():
                assert not torch.isnan(p).any(), print('After step, NAN exist in {}.'.format(name), p)

            for name, p in self.critic.named_parameters():
                assert not torch.isnan(p).any(), print('After step, NAN exist in {}.'.format(name), p)

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

    def get_cumulative_rewards(self, states: np.ndarray, rewards: np.ndarray, terminals: np.ndarray):
        r"""
        The new data in buffer:
        A[i], R[i], S[i], T[i] are corresponding to:
            At, Rt, St+1, Tt+1

        sum[T] = Q(State[T], Sampled_Action) * gamma * (1 - Terminal[T]) + Rewards[T]
        sum[t] = sum[t+1] * gamma * (1 - Terminal[t]) + Rewards[t]

        Args:
            states: (num_envs, explore_len, state_dim)
            rewards: (num_envs, explore_len)
            terminals: (num_envs, explore_len)

        Returns:
            The cumulative rewards (np.ndarray): of size (num_envs, explore_len).
        """
        cumulative = np.empty_like(rewards)
        last_state = torch.as_tensor(states[:, -1], dtype=self.buffer.sample_dtype, device=self.device)
        # the mean q value of the last state.
        with torch.no_grad():
            action_avg, action_std = self.actor.forward(state=last_state)
            # (num_envs, 1)
            next_value = self.critic_target.mean_q(state=last_state, action=action_avg).cpu().numpy()

        explore_len = states.shape[-2]
        for t in range(explore_len - 1, -1, -1):
            next_value = rewards[..., t:t+1] + self.gamma * (1 - terminals[..., t:t+1]) * next_value
            cumulative[..., t:t+1] = next_value
        return cumulative

    def update_norm_values(self):
        states, actions, rewards, terminals = self.buffer.recent_add_data
        cumulative_rewards = self.get_cumulative_rewards(states=states, rewards=rewards, terminals=terminals)
        states = states.reshape((-1, self.state_dim))
        cumulative_rewards = cumulative_rewards.reshape((-1,))

        states = torch.as_tensor(states, dtype=self.buffer.sample_dtype, device=self.device)
        cumulative_rewards = torch.as_tensor(cumulative_rewards, dtype=self.buffer.sample_dtype, device=self.device)
        state_mean, state_std = torch.mean(states, dim=0), torch.std(states, dim=0)
        reward_mean, reward_std = torch.mean(cumulative_rewards, dim=0), torch.std(cumulative_rewards, dim=0)

        if torch.isnan(reward_mean).any() or torch.isinf(reward_mean).any():
            print('NAN appears in new reward mean: ', reward_mean)

        if torch.isnan(reward_std).any() or torch.isinf(reward_std).any():
            print('NAN appears in new reward std: ', reward_std)
            print('raw rewards: ', rewards)
            print('cumulative_rewards: ', cumulative_rewards)

            for name, p in self.critic.named_parameters():
                print('critic param {} max: {}; min: {}'.format(name, torch.max(p.data), torch.min(p.data)))

            for name, p in self.critic_target.named_parameters():
                print('target critic param {} max: {}; min: {}'.format(name, torch.max(p.data), torch.min(p.data)))

        momentum = self.batch_norm_momentum
        self.actor.state_mean.data = (1 - momentum) * self.actor.state_mean.data + momentum * state_mean
        self.actor.state_std.data = (1 - momentum) * self.actor.state_std.data + momentum * state_std
        self.critic.state_mean.data = self.actor.state_mean.data
        self.critic.state_std.data = self.actor.state_std.data

        self.critic.q_mean.data = (1 - momentum) * self.critic.q_mean.data + momentum * reward_mean
        self.critic.q_std.data = (1 - momentum) * self.critic.q_std.data + momentum * reward_std

        if torch.isnan(self.critic.q_mean.data).any() or torch.isinf(self.critic.q_mean.data).any():
            print('NAN appears in q mean; new reward mean: ', reward_mean)

        if torch.isnan(self.critic.q_std.data).any() or torch.isinf(self.critic.q_std.data).any():
            print('NAN appears in q std; new reward std: ', reward_std)
            print('raw rewards: ', rewards)
            print('cumulative_rewards: ', cumulative_rewards)

        if torch.isnan(self.critic.state_mean.data).any() or torch.isinf(self.critic.state_mean.data).any():
            print('NAN appears in state mean; new state_mean: ', state_mean)

        if torch.isnan(self.critic.state_std.data).any() or torch.isinf(self.critic.state_std.data).any():
            print('NAN appears in state std; new state_std: ', state_std)

        # print('reward_mean: {}, reward_std: {}'.format(reward_mean, reward_std))
        # print('state_mean: {}, state_std: {}'.format(state_mean, state_std))
        # print('current_state_mean: {}, current_state_std: {}'.format(self.critic.state_mean.data, self.critic.state_std.data))
        # print('current_value_mean: {}, current_value_std: {}'.format(self.critic.q_mean.data,
        #                                                              self.critic.q_std.data))

    def update_network(self, update_actor: bool=True):
        r"""
        sample from buffer and compute gradients, update critic and actor
        """
        # logging.info('Update networks, at step {:d}.'.format(self.total_steps))

        # frame data
        # (batch_size, state_dim), (batch_size, action_dim), (batch_size,), (batch_size,), (batch_size, state_dim)
        states, actions, rewards, terminals, next_states = self.buffer.get_frame_data()
        # TODO: Sequence data

        # update critic
        critic_loss = self.critic_objective(states, actions, rewards, terminals, next_states)
        self.optimizer_update(self.critic_optimizer, critic_loss)
        self.soft_update_critic_target()

        if update_actor:
            # update alpha
            #   resample actions at current states
            dummy_action, action_log_prob = self.actor.get_action_logprob(states)
            # TODO: here, I use the raw alpha for computing; others use alpha_log?
            # alpha = torch.exp(self.alpha_log)
            # TODO: use alpha_log
            alpha = self.alpha_log
            entropy_diff = (- action_log_prob - self.target_entropy).detach()
            alpha_loss = torch.mean(entropy_diff * alpha)
            self.optimizer_update(self.alpha_optimizer, alpha_loss)
            #   clip the updated alpha
            new_alpha = torch.exp(self.alpha_log).detach()
            with torch.no_grad():
                self.alpha_log.data = torch.clamp(self.alpha_log.data, self.alpha_log_lower, self.alpha_log_upper)

            # logging.info('At step {:d},\t'
            #              'Update alpha_log to {:.4f}'.format(self.total_steps, self.alpha_log.data.detach().cpu().item()))

            alpha_loss = alpha_loss.detach().cpu().item()

            # update actor
            #   (batch_size, 1)
            # TODO: He uses mean q here, but I will use the min q. In addition, he uses the target critic, but I use critic
            q_val = self.critic.min_q(state=states, action=dummy_action).squeeze(1)

            # TODO: ElegantRL 这里用的是:
            # q_val = self.critic_target.mean_q(state=states, action=dummy_action).squeeze(1)

            #   the objective of actor is to make its policy near to a boltzmann distribution
            actor_loss = torch.mean(new_alpha * action_log_prob - q_val)
            self.optimizer_update(self.actor_optimizer, actor_loss)
            actor_loss = actor_loss.detach().cpu().item()
        else:
            actor_loss = 0
            alpha_loss = 0

        critic_loss = critic_loss.detach().cpu().item()
        logging.info('At step {:d}\tcritic_loss: {:.4f}\talpha_loss: {:.4f}\t'
                     'actor_loss: {:.4f}'.format(self.total_steps, critic_loss, alpha_loss, actor_loss))
        # print('At step {:d}\tcritic_loss: {:.4f}\talpha_loss: {:.4f}\t'
        #              'actor_loss: {:.4f}'.format(self.total_steps, critic_loss, alpha_loss, actor_loss))
        return critic_loss, actor_loss, alpha_loss

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
        # TODO: with no grad
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
            # TODO: disable state normalize
            state = self.normalize_state(state)
            state = torch.tensor(state, dtype=self.buffer.sample_dtype, device=self.device)

            # TODO: use action_avg
            action_mean, _ = self.actor(state)
            action = action_mean.tanh().detach().cpu().numpy()

            # action = self.actor.choose_action(state).detach().cpu().numpy()

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

            # print(f"step {step_idx}: reward: ", reward, "terminal: ", terminal)

            success = np.bitwise_and(np.bitwise_not(truncated), terminal)

            # TODO: align with gym 0.24.0: truncate also mean terminals
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

        # record
        reward_name = 'step{}_eval_rewards'.format(self.total_steps)
        terminal_name = 'step{}_eval_terminals'.format(self.total_steps)
        avg_reward_name = 'step{}_average_reward'.format(self.total_steps)
        avg_step_name = 'step{}_average_step'.format(self.total_steps)
        episodic_reward_name = "step{}_episodic_rewards".format(self.total_steps)
        episodic_step_name = "step{}_episodic_steps".format(self.total_steps)

        # if save_data:
        #     with h5py.File(data_save_path, 'a') as eval_record:
        #         # eval_record.create_dataset(name=reward_name, data=rewards)
        #         # eval_record.create_dataset(name=terminal_name, data=terminals)
        #         # eval_record.create_dataset(name=avg_reward_name, data=np.array([avg_reward]))
        #         # eval_record.create_dataset(name=avg_step_name, data=np.array([avg_step]))
        #         eval_record.create_dataset(name=episodic_reward_name, data=np.array(episodic_rewards))
        #         eval_record.create_dataset(name=episodic_step_name, data=np.array(episodic_steps))
        #
        #     with h5py.File(sub_data_save_path, 'w') as sub_record:
        #         sub_record.create_dataset(name="eval_rewards", data=rewards)
        #         sub_record.create_dataset(name="eval_terminals", data=terminals)
        #         sub_record.create_dataset(name="average_reward", data=np.array([avg_reward]))
        #         sub_record.create_dataset(name="average_step", data=np.array([avg_step]))
        #         sub_record.create_dataset(name="episodic_rewards", data=np.array(episodic_rewards))
        #         sub_record.create_dataset(name="episodic_steps", data=np.array(episodic_steps))
        #         sub_record.create_dataset(name="success_episodic_steps", data=np.array(success_episodic_steps))

        torch.set_grad_enabled(True)
        return episodic_rewards, episodic_steps, success_episodic_steps, avg_reward, avg_step

    def evaluate_and_show_mem(self, eval_env, max_steps, episodic_num, save_dir):
        r"""
        Act in environments, and record the interaction data.

        Args:
         eval_env: the environment for evaluation.
         max_steps (int): the maximum steps for evaluation.
         episodic_num (int): the number of episodic in need.
         save_dir (str): the record file path.
        """
        # TODO: with no grad
        torch.set_grad_enabled(False)
        state = eval_env.reset()[0]
        pics_list = []
        pics = []
        rewards = []
        episodic_count = 0
        # logging.info('Evaluating for display ...')
        print('Evaluating single env for display ...')
        for number in range(max_steps):
            # normalize state
            # TODO: disable state normalize
            state = self.normalize_state(state)
            # (num_envs, state_sim)
            state = torch.tensor(state, dtype=self.buffer.sample_dtype, device=self.device)
            action = self.actor.choose_action(state).detach().cpu().numpy()
            # real action
            real_action = self.real_action(action)
            state, reward, done, truncated, _ = eval_env.step(real_action)
            rewards.append(reward)

            frame = eval_env.render()
            if isinstance(frame, list):
                pics += frame
            else:
                pics.append(frame)

            # print('step {:d}, state: {} reward: {} done: {}'.format(number, 1 / state, reward, done))
            if done[0]:
                pics_list.append(pics)
                print(len(pics) / 2)
                pics = []
                episodic_count += 1
            if episodic_count >= episodic_num:
                break

        # TODO:
        torch.set_grad_enabled(True)
        # record
        with h5py.File(save_dir, 'a') as eval_record:
            for idx, each_pics in enumerate(pics_list):
                save_name = 'step{}_eval{}'.format(self.total_steps, idx)
                each_pics = np.stack(each_pics, axis=0)
                eval_record.create_dataset(name=save_name, data=each_pics)

    def evaluate_and_show(self, eval_env: Env, max_steps, episodic_num, save_dir):
        r"""
         Act in a single environment, and record the interaction data.

         Args:
             eval_env (gym.Env): the environment for evaluation.
             max_steps (int): the maximum steps for evaluation.
             episodic_num (int): the number of episodic in need.
             save_dir (str): the record file path.
         """
        # TODO: with no grad
        torch.set_grad_enabled(False)
        assert eval_env.render_mode == "rgb_array"
        state = eval_env.reset()[0]
        pics_list = []
        pics = []
        rewards = []
        episodic_count = 0
        logging.info('Evaluating for display ...')
        print('Evaluating single env for display ...')
        for number in range(max_steps):
            frame = eval_env.render()
            # print('rgb_array size: ', frame.shape)
            if isinstance(frame, list):
                pics += frame
            else:
                pics.append(frame)
            # normalize state
            # TODO: disable state normalize
            state = self.normalize_state(state)
            # (1, state_sim)
            state = torch.tensor(state, dtype=self.buffer.sample_dtype, device=self.device).unsqueeze(0)
            action = self.actor.choose_action(state).detach().cpu().numpy().squeeze(0)
            # real action
            real_action = self.real_action(action)
            state, reward, done, truncated, _ = eval_env.step(real_action)
            rewards.append(reward)
            if done:
                state = eval_env.reset()[0]
                pics_list.append(pics)
                pics = []
                episodic_count += 1
            if episodic_count >= episodic_num:
                break

        # TODO:
        torch.set_grad_enabled(True)
        # record
        with h5py.File(save_dir, 'a') as eval_record:
            for idx, each_pics in enumerate(pics_list):
                save_name = 'step{}_eval{}'.format(self.total_steps, idx)
                each_pics = np.stack(each_pics, axis=0)
                eval_record.create_dataset(name=save_name, data=each_pics)
