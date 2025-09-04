import torch
import torch.nn as nn
import numpy as np

from torch.distributions.normal import Normal
from .utils import layer_init_with_orthogonal, construct_mlp
from typing import Union, List, Tuple


class PhysicsAwareActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_net
        self.mos_net

    def forward(self):
        r"""
        1. Sample actor net
        2. Determinist actor net (td3)

        # TODO: MOS 监督训练数据准备；监督训练接口；输入输出关系映射。

        Returns:

        """



class ActorBase(nn.Module):
    def __init__(self, representation_net: nn.Module, output_net: nn.Module,
                 discrete_action: bool, determinist_action: bool, action_dim: int):
        super().__init__()
        self.representation_net = representation_net
        self.output_net = output_net
        self.discrete_action = discrete_action
        self.determinist_action = determinist_action
        self.action_dim = action_dim

    def choose_action(self, states, **kwargs):
        r"""
        inputs: refer to Representation
            (batch_size, state_len, state_dim);
            other information composing a practically used state(e.g. history actions).
        """
        if self.determinist_action:
            return self.forward(states, **kwargs)
        elif self.discrete_action:
            r""" 
                TODO: it seems no rl algorithm implemented uses discrete action now; PPO may use it.
                temporarily we return the output as inputs for softmax function for probability.
            """
            return self.forward(states, **kwargs)
        output = self.forward(states, **kwargs)
        assert output.size()[-1] == self.action_dim * 2
        mean, std = output[:, :self.action_dim], output[:, self.action_dim:]
        actions = torch.distributions.Normal(mean, std).sample()
        actions_log_prob = torch.distributions.Normal(mean, std).log_prob(actions)
        return actions, actions_log_prob

    def forward(self, states, **kwargs):
        r"""
        return:
            if determinist_action or discrete_action:
                (batch_size, action_dim)
            else:
                TODO: Here, we use two correlated net. may use two independent net instead.
                (batch_size, action_dim * 2) (0: action_dim -> mean; action_dim: action_dim * 2 -> std.)
        """
        output = self.representation_net(states, **kwargs)
        output = self.output_net(output)
        return output


class SACActor(nn.Module):
    r"""
    Actor of the SAC algorithm.

    Attributes:
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        actor_net: The actor neural network, which outputs the mean values and log(std) values for sampling actions.
        std_log_lower (int): The log(std) values are clamped between std_log_lower and std_log_upper.
        std_log_upper (int): The log(std) values are clamped between std_log_lower and std_log_upper.
    """
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            actor_net,
            std_log_lower: int = -16,
            std_log_upper: int = 2,
            random_init: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_net = actor_net
        if random_init:
            layer_init_with_orthogonal(self.actor_net[-1], std=0.1)

        self.std_log_lower = std_log_lower
        self.std_log_upper = std_log_upper

        self.state_mean = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: torch.Tensor):
        if (self.state_std.data == 0).any():
            return (state - self.state_mean) / (self.state_std + self.eps)
        return (state - self.state_mean) / self.state_std

    def rsample(self, state: torch.Tensor):
        r"""
         sample action according to the output mean and std of the actor net.
         return:
            dist: distribution
            action: (batch_size, action_dim)
         """
        # state = self.state_norm(state)
        # net_out = self.actor_net(state)
        # action_mean, action_std_log = net_out[..., :self.action_dim], net_out[..., self.action_dim:]
        # action_std = torch.exp(torch.clamp(action_std_log, self.std_log_lower, self.std_log_upper))
        if not torch.isfinite(state).all():
            print(state)
            raise ValueError('Infinite appears in states.')
        if torch.isnan(state).any():
            print(state)
            raise ValueError('Nan appears in states.')
        action_mean, action_std = self.forward(state)
        # print(action_mean, action_std)
        dist = Normal(action_mean, action_std)
        action = dist.rsample()
        return dist, action

    def forward(self, state: torch.Tensor):
        r"""
        Get the net output as the mean and std of action.
         state:
            (batch_size, state_dim) for frame data;

        Returns:
            action_mean: (batch_size, action_dim), the means of action of size (batch_size, action_dim).
            action_std: (batch_size, action_dim), the stds of action of size (batch_size, action_dim).
        """
        state = self.state_norm(state)
        net_out = self.actor_net(state)
        action_mean, action_std_log = net_out[..., :self.action_dim], net_out[..., self.action_dim:]
        action_std = torch.exp(torch.clamp(action_std_log, self.std_log_lower, self.std_log_upper))
        return action_mean, action_std

    def choose_action(self, state: torch.Tensor):
        r"""
         state:
            (batch_size, state_dim) for frame data;
            or
            (batch_size, state_dim + action_dim, max_len) for sequence data ??
            (TODO: to be determined: it means that the sequence of (state, action) is used as an actual state)
        net_output:
            (batch_size, action_dim * 2)
            note that, the last layer do not have activation (it is moved here, after sampling)
        return:
            action: (batch_size, action_dim)
        """
        _, action = self.rsample(state)
        action = torch.tanh(action)
        return action

    def get_action_logprob(self, state: torch.Tensor):
        r"""
        here, compensation is applied to the log_prob.
        this compensation derives from the squashing function tanh:
            s_a: sampled_action;
            a: actual action, a = tanh(s_a);
            log_prob(a|s) = log_prob(dist(s_a|s)) - sum(log(1 - tanh(s_a) ** 2))
        return:
            action: (batch_size, action_dim)
            log_prob: (batch_size, )
        """
        eps = 1e-8
        dist, sampled_action = self.rsample(state)

        action = torch.tanh(sampled_action)
        compensate = 1 - torch.pow(action, 2)
        if (compensate == 0).any():
            compensate[compensate == 0] += eps

        log_prob = dist.log_prob(sampled_action) - torch.log(compensate)
        log_prob = torch.sum(log_prob, dim=1)
        return action, log_prob

    def supervise_action_logprob(self, state: torch.Tensor, action: torch.Tensor):
        r"""
        Args:
            state: the state.
            action: the action for supervised learning.

        Returns:
            exe_action: sampled from current dist.
            log_prob: the log prob of the input action in current dist.
        """
        eps = 1e-8
        # print(state)
        dist, sampled_action = self.rsample(state)
        exe_action = torch.tanh(sampled_action)
        # print(action)
        equal_sample_action = torch.atanh(action)
        # print(equal_sample_action)
        compensate = 1 - torch.pow(action, 2)
        if (compensate == 0).any():
            compensate[compensate == 0] += eps
        log_prob = dist.log_prob(equal_sample_action) - torch.log(compensate)
        log_prob = torch.sum(log_prob, dim=1)
        return exe_action, log_prob


class TD3Actor(nn.Module):
    r"""
    Actor of the TD3 algorithm.

    Attributes:
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        actor_net: The neural network that receives the state as input and outputs the actions.
        explore_std (float): The std of the noise that is sampled from a Normal distribution, and will be added to the
            determined action for the sake of exploration (randomness).
        target_action_noise_std (float):
            When updating the network, add noise to the target_actor's action for generalization.
        target_noise_clip_val (float):
            Clip the target action noise between (-target_noise_clip_val, target_noise_clip_val).
    """
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Union[List[int], Tuple[int]],
            explore_std: float,
            target_action_noise_std: float,
            target_noise_clip_val: float,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_net = construct_mlp(dims=[state_dim, *hidden_dims, action_dim])
        self.explore_std = explore_std
        self.target_action_noise_std = target_action_noise_std
        self.target_noise_clip_val = target_noise_clip_val

    def forward(self, state: torch.Tensor):
        action = self.actor_net(state)
        return action

    def choose_action(self, state: torch.Tensor):
        r""" Use it when evaluating """
        action = self.forward(state)
        action = torch.tanh(action)
        return action

    def choose_noisy_action(self, state: torch.Tensor):
        r""" Use it when exploring """
        action = self.forward(state)
        action = torch.tanh(action)

        noise = torch.randn_like(action) * self.explore_std
        action = torch.clamp(action + noise, -1, 1)
        return action

    def choose_target_action(self, state: torch.Tensor):
        r""" Use it when choosing the action of the target actor for the objective calculations. """
        action = self.forward(state)
        action = torch.tanh(action)

        clipped_noise = (torch.randn_like(action) * self.target_action_noise_std)
        clipped_noise = torch.clamp(clipped_noise, -self.target_noise_clip_val, self.target_noise_clip_val)
        action = torch.clamp(action + clipped_noise, -1, 1)
        return action

class PPOActor(nn.Module):
    r"""
    Actor of PPO algorithm.

    Attributes:
        state_dim (int): The dimension of state space.
        action_dim (int): The dimension of action space.
        actor_net (int): The actor_net outputs the mean values and log_std values; Here we handle the action clipping
            similar to the SAC algorithm.
    """
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Union[Tuple[int], List[int]],
            std_log_lower: int = -16,
            std_log_upper: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_net = construct_mlp(dims=[state_dim, *hidden_dims, 2 * action_dim])
        self.std_log_lower = std_log_lower
        self.std_log_upper = std_log_upper

    def forward(self, state: torch.Tensor):
        r"""

        Args:
            state (torch.Tensor): (batch_size, state_dim)

        Returns:
            action_mean, action_std: (batch_size, action_dim)
        """
        net_out = self.actor_net(state)
        action_mean, action_std_log = net_out[..., :self.action_dim], net_out[..., self.action_dim:]
        action_std = torch.exp(torch.clamp(action_std_log, self.std_log_lower, self.std_log_upper))
        return action_mean, action_std

    def deterministic_action(self, state: torch.Tensor):
        r""" Use it for evaluation. """
        action_mean, action_std = self.forward(state)
        action = torch.tanh(action_mean)
        return action

    def rsample(self, state: torch.Tensor):
        action_mean, action_std = self.forward(state)
        dist = Normal(action_mean, action_std)
        sampled_action = dist.rsample()
        return dist, sampled_action

    def get_sampled_action_logprob(self, state: torch.Tensor):
        dist, sampled_action = self.rsample(state)

        log_prob = dist.log_prob(sampled_action)
        log_prob = torch.sum(log_prob, dim=1)
        return sampled_action, log_prob

    def get_sampled_logprob_entropy(self, state: torch.Tensor, sampled_action: torch.Tensor):
        r"""

        Args:
            state (torch.Tensor): (batch_size, state_dim)
            sampled_action (torch.Tensor): (batch_size, action_dim); range: (-inf, inf)

        Returns:
            log_prob, entropy: (batch_size, )
        """
        dist, _ = self.rsample(state)

        log_prob = dist.log_prob(sampled_action)
        log_prob = torch.sum(log_prob, dim=1)
        entropy = dist.entropy().sum(dim=1)
        return log_prob, entropy

    def normalized_action(self, sampled_action: torch.Tensor):
        r""" Apply Tanh """
        return torch.tanh(sampled_action)

