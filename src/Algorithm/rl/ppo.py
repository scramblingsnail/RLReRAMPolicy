import torch
import logging
import numpy as np
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
from gym.vector import AsyncVectorEnv
from typing import Tuple, List

from ..net import PPOActor, PPOCritic


class PPO:
    r"""
    PPO algorithm.
    """
    def __init__(
            self,
            actor: PPOActor,
            critic: PPOCritic,
            envs: AsyncVectorEnv,
            gamma: float,
            lambda_gae: float,
            lr: float,
            batch_size: int,
            ratio_clip: float,
            lambda_entropy: float,
            clip_grad_norm: float = 3.0,
            train_repeats: int = 1,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer_dtype = torch.float32

        self.lr = lr
        self.batch_size = batch_size
        self.ratio_clip = ratio_clip
        self.lambda_entropy = lambda_entropy
        self.actor = actor.to(self.device)
        self.actor_optimizer = torch.optim.AdamW(params=self.actor.parameters(), lr=self.lr)
        self.critic = critic.to(self.device)
        self.critic_optimizer = torch.optim.AdamW(params=self.critic.parameters(), lr=self.lr)

        self.envs = envs
        self.action_dim = self.actor.action_dim
        self.state_dim = self.actor.state_dim
        self.state_bot = self.envs.observation_space.__dict__['low'][0]
        self.state_top = self.envs.observation_space.__dict__['high'][0]
        self.action_bot = self.envs.action_space.__dict__['low'][0]
        self.action_top = self.envs.action_space.__dict__['high'][0]

        self.clip_grad_norm = clip_grad_norm
        self.train_repeats = train_repeats
        # self.mse_loss = nn.MSELoss()
        self.mse_loss = nn.SmoothL1Loss(reduction="mean")
        self.gamma = min(1., max(0., gamma))
        self.lambda_gae = min(1., max(0., lambda_gae))
        self.total_steps = 0
        self.last_state = None
        self.train_buffer = {
            "states": torch.empty(1),
            "sampled_actions": torch.empty(1),
            "advantages": torch.empty(1),
            "target_values": torch.empty(1),
            "logprobs": torch.empty(1),
            "buffer_size": 0,
        }

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

    @property
    def buffer_size(self):
        return self.train_buffer["buffer_size"]

    def _update_buffer(
            self,
            states: np.ndarray,
            sampled_actions: np.ndarray,
            logprobs: np.ndarray,
            rewards: np.ndarray,
            terminals: np.ndarray,
            final_state: np.ndarray,
    ):
        states = torch.tensor(states, dtype=self.buffer_dtype, device=self.device)
        sampled_actions = torch.tensor(sampled_actions, dtype=self.buffer_dtype, device=self.device)
        logprobs = torch.tensor(logprobs, dtype=self.buffer_dtype, device=self.device)
        rewards = torch.tensor(rewards, dtype=self.buffer_dtype, device=self.device)
        terminals = torch.tensor(terminals, dtype=self.buffer_dtype, device=self.device)
        final_state = torch.tensor(final_state, dtype=self.buffer_dtype, device=self.device)
        self.train_buffer["buffer_size"] = rewards.size()[0] * rewards.size()[1]
        self.train_buffer["states"] = states
        self.train_buffer["sampled_actions"] = sampled_actions
        self.train_buffer["logprobs"] = logprobs
        advantages, target_values = self.calculate_gae(
            states=states,
            rewards=rewards,
            terminals=terminals,
            final_state=final_state,
        )
        self.train_buffer["advantages"] = advantages
        self.train_buffer["target_values"] = target_values

    def sample_batch_data(self):
        env_num = self.train_buffer["states"].size()[0]
        step_num = self.train_buffer["states"].size()[1]

        indices = torch.randint(step_num * env_num, size=(self.batch_size, ), device=self.device)
        ids0 = torch.floor(indices / step_num).long()
        ids1 = torch.fmod(indices, step_num).long()

        batch_states = self.train_buffer["states"][ids0, ids1]
        batch_sampled_actions = self.train_buffer["sampled_actions"][ids0, ids1]
        batch_logprobs = self.train_buffer["logprobs"][ids0, ids1]
        batch_advantages = self.train_buffer["advantages"][ids0, ids1]
        batch_target_values = self.train_buffer["target_values"][ids0, ids1]
        return batch_states, batch_sampled_actions, batch_logprobs, batch_target_values, batch_advantages

    def interact_with_envs(self, interact_steps: int, random_mode: bool = False):
        r"""
        Exploration.

        Args:
            interact_steps:
            random_mode:

        Returns:
            Tuple[np.ndarray]: (states, actions, logprobs, rewards, terminals)

        """
        states = np.zeros((self.envs.num_envs, interact_steps, self.state_dim), dtype=float)
        sampled_actions = np.zeros((self.envs.num_envs, interact_steps, self.action_dim), dtype=float)
        logprobs = np.zeros((self.envs.num_envs, interact_steps), dtype=float)
        rewards = np.zeros((self.envs.num_envs, interact_steps), dtype=float)
        terminals = np.zeros((self.envs.num_envs, interact_steps), dtype=bool)

        self.total_steps += interact_steps * self.envs.num_envs

        if self.last_state is None:
            self.last_state = self.envs.reset()[0]
            self.last_state = self.normalize_state(self.last_state)

        state = self.last_state

        for idx in range(interact_steps):
            states[:, idx] = state
            state = torch.tensor(state, dtype=self.buffer_dtype, device=self.device)
            sampled_action, logprob = self.actor.get_sampled_action_logprob(state=state)
            action = self.actor.normalized_action(sampled_action).cpu().detach().numpy()
            logprob = logprob.cpu().detach().numpy()
            real_action = self.real_action(action)

            state, reward, terminal, truncated, _ = self.envs.step(real_action)
            # normalize state
            state =  self.normalize_state(state)
            terminal = np.bitwise_or(terminal, truncated)
            sampled_actions[:, idx] = sampled_action.cpu().detach().numpy()
            logprobs[:, idx] = logprob
            rewards[:, idx] = reward
            terminals[:, idx] = terminal

        self.last_state = state
        final_state = state
        self._update_buffer(
            states=states,
            sampled_actions=sampled_actions,
            logprobs=logprobs,
            rewards=rewards,
            terminals=terminals,
            final_state=final_state,
        )

    def calculate_gae(
            self,
            states: torch.Tensor,
            rewards: torch.Tensor,
            terminals: torch.Tensor,
            final_state: torch.Tensor,
    ):
        r"""
        Calculate GAE.

        Args:
            states (torch.Tensor): (env_num, step_num, state_dim)
            rewards (torch.Tensor): (env_num, step_num)
            terminals (torch.Tensor): (env_num, step_num)
            final_state (torch.Tensor): (env_num, state_dim)

        Returns:
            advantages: (env_num, step_num)

        """
        batch_size = 256
        env_num, step_num = states.size()[0], states.size()[1]
        batch_step_num = max(batch_size // env_num, 1)
        values = torch.empty_like(rewards)
        with torch.no_grad():
            final_value = self.critic(final_state)
            for step_idx in range(0, step_num, batch_step_num):
                values[:, step_idx: step_idx+batch_step_num] = self.critic(states[:, step_idx: step_idx+batch_step_num])

        discount = self.gamma * self.lambda_gae
        advantages = torch.zeros_like(values)

        next_value = final_value
        next_adv = torch.zeros_like(next_value)

        for t in range(step_num - 1, -1, -1):
            td_error_t = rewards[:, t] + self.gamma * (1 - terminals[:, t]) * next_value - values[:, t]
            current_adv = discount * (1 - terminals[:, t]) * next_adv + td_error_t
            advantages[:, t] = current_adv

            next_adv = current_adv
            next_value = values[:, t]
        target_values = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(dim=1) + 1e-4)
        return advantages, target_values

    def optimizer_update(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor):
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def update_network(self):
        batch_states, batch_sampled_actions, batch_logprobs, batch_target_values, batch_advantages = self.sample_batch_data()

        predict_val = self.critic(batch_states)
        critic_loss = self.mse_loss(predict_val, batch_target_values)
        self.optimizer_update(optimizer=self.critic_optimizer, loss=critic_loss)
        critic_loss = critic_loss.detach().cpu().item()

        logprob, policy_entropy = self.actor.get_sampled_logprob_entropy(
            state=batch_states,
            sampled_action=batch_sampled_actions,
        )
        ratio = torch.exp(logprob - batch_logprobs)
        actor_loss_raw = batch_advantages * ratio
        actor_loss_clip = batch_advantages * torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
        actor_loss_min = torch.min(actor_loss_raw, actor_loss_clip).mean()
        entropy_bonus = torch.mul(policy_entropy.mean(), self.lambda_entropy)
        actor_loss = - (actor_loss_min + entropy_bonus)
        self.optimizer_update(self.actor_optimizer, actor_loss)
        actor_loss = actor_loss.detach().cpu().item()
        return critic_loss, actor_loss

    def evaluate(self, eval_envs, max_steps: int, log_dir: str):
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
            state = torch.tensor(state, dtype=self.buffer_dtype, device=self.device)
            action = self.actor.deterministic_action(state=state)
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