import torch
import torch.nn as nn
import numpy as np

from typing import Union, Tuple, List
from .utils import layer_init_with_orthogonal, construct_mlp


class CriticBase(nn.Module):
    def __init__(self, representation_net: nn.Module, output_net: nn.Module):
        super().__init__()
        self.representation_net = representation_net
        self.output_net = output_net

    def forward(self, states, action, **kwargs):
        value = self.representation_net(states, action, **kwargs)
        value = self.output_net(value)
        return value



class TwinCritic(nn.Module):
    r""" Here are two Q Net.
    When Q value is required for gradient computation, use the minimum value of the two Q values

    Modified: merge q1 and q2 to be a new q net, it has two outputs.
    """
    def __init__(self, state_dim, action_dim, q_net=None, hidden_dims=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if q_net is None:
            if hidden_dims is None:
                raise ValueError("Should give hidden dims if q_net is Not given.")
            self.critic = construct_mlp(dims=[action_dim+state_dim, *hidden_dims, 2])
        else:
            self.critic = q_net
        layer_init_with_orthogonal(self.critic[-1], std=0.5)

        self.state_mean = nn.Parameter(torch.zeros((state_dim, )), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.q_mean = nn.Parameter(torch.zeros((1, )), requires_grad=False)
        self.q_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
        self.eps = 1e-8

    def state_norm(self, state: torch.Tensor):
        if (self.state_std.data == 0).any():
            return (state - self.state_mean) / (self.state_std + self.eps)

        return (state - self.state_mean) / self.state_std

    def q_denorm(self, q: torch.Tensor):
        return q * self.q_std + self.q_mean

    def twin_q(self, state, action):
        r"""
        calculate the two q values of critic net:

        Inputs:
            state(torch.Tensor):
                >- (batch_size, state_dim), (batch_size, action_dim) for frame data
                >- or
                >- (batch_size, state_dim, max_len), (batch_size, action_dim, max_len) for sequence data
                >- note: action is normalized action, range from (-1 to 1)

        Returns:
            q1(torch.Tensor): (batch_size, 1)
            q2(torch.Tensor): (batch_size, 1)
        """
        # state normalization:
        state = self.state_norm(state)
        batch_input = torch.cat((state, action), dim=1)
        qs = self.critic(batch_input)
        if torch.isnan(qs).any() or torch.isinf(qs).any():
            print('In twin q before denorm, NAN appears.')
            print('batch_input: ', batch_input)
            print('qs: ', qs)

        qs = self.q_denorm(qs)
        if torch.isnan(qs).any() or torch.isinf(qs).any():
            print('In twin q after denorm, NAN appears.')
            print('qs: ', qs)
            print('q_std: ', self.q_std)
            print('q_mean: ', self.q_mean)

        q1, q2 = qs[:, :1], qs[:, 1:]
        return q1, q2

    def mean_q(self, state, action):
        q1, q2 = self.twin_q(state=state, action=action)
        q = (q1 + q2) / 2
        return q

    def min_q(self, state, action):
        r"""
        calculate the min q value.

        Inputs:
            state(torch.Tensor):
                >- (batch_size, state_dim), (batch_size, action_dim) for frame data
                >- or
                >- (batch_size, state_dim, max_len), (batch_size, action_dim, max_len) for sequence data
                >- note: action is normalized action, range from (-1 to 1)

        Returns:
            q(torch.Tensor): (batch_size, 1)
        """
        q1, q2 = self.twin_q(state, action)
        q = torch.minimum(q1, q2)
        return q


class PPOCritic(nn.Module):
    r"""
    Critic of PPO algorithm.
    """
    def __init__(
            self,
            state_dim: int,
            hidden_dims: Union[Tuple[int], List[int]],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.critic = construct_mlp(dims=[state_dim, *hidden_dims, 1])

    def forward(self, state: torch.Tensor):
        r"""

        Args:
            state (torch.Tensor): (batch_size, state_dim)

        Returns:
            state_value: (batch_size, 1)

        """
        state_value = self.critic(state)
        return state_value.squeeze(-1)
