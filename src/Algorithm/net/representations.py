import torch
import torch.nn as nn
import numpy as np

from typing import Union


class Representation(nn.Module):
    r"""
    input data: (batch_size, length, dim)
    """
    def __init__(self, observation_dim: int, state_len: int, observation_embed_size: int, discrete_observation: bool,
                 discrete_observation_num: int, action_dim: int, padding_action: torch.Tensor = None,
                 is_critic: bool = False, action_embed_size: int = None, discrete_action: bool = None,
                 discrete_action_num: int = None):
        super().__init__()
        self.observation_dim = observation_dim
        self.state_len = state_len
        self.action_dim = action_dim
        self.observation_embed_size = observation_embed_size
        self.action_embed_size = action_embed_size
        self.discrete_observation = discrete_observation
        self.discrete_action = discrete_action
        self.padding_action = padding_action

        if self.discrete_observation:
            for dim_idx in range(observation_dim):
                self.__setattr__(name='state_embedding_{:d}'.format(dim_idx),
                                 value=nn.Embedding(discrete_observation_num, observation_embed_size))
        else:
            # may be more complex network
            self.state_embedding = nn.Linear(observation_dim, observation_dim * observation_embed_size)

        # for actions in state history: (S1, A1, S2, A2, ..., At-1, St)
        if state_len > 1:
            if None in (padding_action, action_embed_size, discrete_action, discrete_action_num):
                raise ValueError('Missing padding_action, action_embed_size, discrete_action, discrete_action_num.')
            assert padding_action.size() == (action_dim,)
            if self.discrete_action:
                for dim_idx in range(action_dim):
                    self.__setattr__(name='history_action_embedding_{:d}'.format(dim_idx),
                                     value=nn.Embedding(discrete_action_num, action_embed_size))
            else:
                self.history_action_embedding = nn.Linear(action_dim, action_dim * action_embed_size)

        # action embedding
        if is_critic:
            if None in (action_embed_size, discrete_action):
                raise ValueError('Please set action_embed_size, discrete_action.')
            if self.discrete_action:
                if discrete_action_num is None:
                    raise ValueError('Please set discrete_action_num.')
                for dim_idx in range(self.action_dim):
                    self.__setattr__(name='action_embedding_{:d}'.format(dim_idx),
                                     value=nn.Embedding(discrete_action_num, action_embed_size))
            else:
                self.action_embedding = nn.Linear(action_dim, action_dim * action_embed_size)

    @property
    def state_embedding_size(self):
        return self.observation_dim * self.observation_embed_size + (self.action_dim * self.action_embed_size) * (self.state_len > 1)

    @property
    def action_embedding_size(self):
        return self.action_embed_size * self.action_dim

    def discrete_embedding(self, values, value_dim, layer_flag):
        r"""
        values: (batch_size, ..., value_dim)
        return: (batch_size, ..., value_dim * value_embed_size)
        """
        embeddings = []
        values = torch.swapaxes(values, 0, -1)
        for dim_idx in range(value_dim):
            embedding = self.__getattr__(name='{}_{:d}'.format(layer_flag, dim_idx))
            embeddings.append(embedding(values[dim_idx]))
        embeddings = torch.cat(embeddings, dim=-1)
        embeddings = torch.swapaxes(embeddings, 0, -2)
        return embeddings

    def discrete_history_action_embedding(self, actions):
        r"""
        actions: (batch_size, length, action_dim)
        return: (batch_size, length, action_dim * action_embed_size)
        """
        action_embeddings = self.discrete_embedding(values=actions, value_dim=self.action_dim,
                                                    layer_flag='history_action_embedding')
        return action_embeddings

    def discrete_state_embedding(self, states):
        r"""
        states: (batch_size, ..., observation_dim)
        return: (batch_size, ..., observation_dim * observation_embed_size)
        """
        state_embeddings = self.discrete_embedding(values=states, value_dim=self.observation_dim,
                                                   layer_flag='state_embedding')
        return state_embeddings

    def discrete_action_embedding(self, actions):
        r"""
        actions: (batch_size, action_dim)
        return: (batch_size, action_dim * action_embed_size)
        """
        action_embeddings = self.discrete_embedding(values=actions, value_dim=self.action_dim,
                                                    layer_flag='action_embedding')
        return action_embeddings

    def continuous_state_embedding(self, states):
        r"""
        states: (batch_size, ..., observation_dim)
        return: (batch_size, ..., observation_dim * observation_embed_size)
        """
        state_embeddings = self.state_embedding(states)
        return state_embeddings

    def continuous_history_action_embedding(self, actions):
        r"""
        actions: (batch_size, length, action_dim)
        return: (batch_size, length, action_dim * action_embed_size)
        """
        history_action_embeddings = self.history_action_embedding(actions)
        return history_action_embeddings

    def continuous_action_embedding(self, actions):
        r"""
        actions: (batch_size, action_dim)
        return: (batch_size, action_dim * action_embed_size)
        """
        action_embeddings = self.action_embedding(actions)
        return action_embeddings

    def observation_embedding(self, states, his_actions: torch.Tensor = None):
        r"""
        states: (batch_size, state_len, observation_dim)
        actions:
                (batch_size, state_len - 1, action_dim) if state_len > 1
                None if state_len == 1
        return:
                embeddings:
                if state_len > 1:
                    E.g. state_len = 2: [(St-1, At-1), (St, A_padding)]
                    (batch_size, state_len, observation_dim * observation_embed_size + action_dim * action_embed_size)
                if state_len == 1:
                    [St]
                    (batch_size, 1, observation_dim * observation_embed_size)
        """
        if self.discrete_observation:
            embeddings = self.discrete_state_embedding(states)
        else:
            embeddings = self.continuous_state_embedding(states)
        if self.state_len > 1:
            if his_actions is None:
                raise ValueError('Missing his_actions when setting history for a state.')
            padding = torch.broadcast_to(self.padding_action[None, None, :],
                                         (his_actions.size()[0], 1, his_actions.size()[2]))
            his_actions = torch.cat((his_actions, padding), dim=1)
            if self.discrete_action:
                # (batch_size, length, action_dim * action_embed_size)
                his_action_embeddings = self.discrete_history_action_embedding(his_actions)
            else:
                his_action_embeddings = self.continuous_history_action_embedding(his_actions)
            embeddings = torch.cat((embeddings, his_action_embeddings), dim=-1)
        return embeddings


class ActorRepresentation(Representation):
    def __init__(self, observation_dim: int, state_len: int, observation_embed_size: int,
                 discrete_observation: bool, discrete_observation_num: int, action_dim: int, padding_action: torch.Tensor,
                 action_embed_size: int = None, discrete_action: bool = None, discrete_action_num: int = None):
        super().__init__(observation_dim=observation_dim, state_len=state_len, observation_embed_size=observation_embed_size,
                         discrete_observation=discrete_observation, discrete_observation_num=discrete_observation_num, action_dim=action_dim,
                         padding_action=padding_action, is_critic=False, action_embed_size=action_embed_size,
                         discrete_action=discrete_action, discrete_action_num=discrete_action_num)

    def forward(self, states, his_actions: torch.Tensor = None):
        return self.observation_embedding(states=states, his_actions=his_actions)


class CriticRepresentation(Representation):
    r"""
    For Q(s, a)
    """
    def __init__(self, observation_dim: int, state_len: int, observation_embed_size: int, discrete_observation: bool,
                 discrete_observation_num: int, action_dim: int, padding_action: torch.Tensor, action_embed_size: int,
                 discrete_action: bool, discrete_action_num: int):
        super().__init__(observation_dim=observation_dim, state_len=state_len,
                         observation_embed_size=observation_embed_size, discrete_observation=discrete_observation,
                         discrete_observation_num=discrete_observation_num, action_dim=action_dim,
                         padding_action=padding_action, is_critic=True, action_embed_size=action_embed_size,
                         discrete_action=discrete_action, discrete_action_num=discrete_action_num)

    def forward(self, states, action, his_actions: torch.Tensor = None):
        r"""
        states: (batch_size, state_len, observation_dim)
        action: (batch_size, action_dim)
        his_actions:
                (batch_size, state_len - 1, action_dim) if state_len > 1
                None if state_len == 1
        return:
            embeddings:
                if state_len > 1:
                    E.g. state_len = 2: [(St-1, At-1), (St, A_padding)]
                    (batch_size, state_len, observation_dim * observation_embed_size + action_dim * action_embed_size)
                if state_len == 1:
                    [St]
                    (batch_size, 1, observation_dim * observation_embed_size)
            action_embedding:
                (batch_size, action_dim * action_embed_size)

        """
        observation_embeddings = self.observation_embedding(states=states, his_actions=his_actions)
        if self.discrete_action:
            action_embedding = self.discrete_action_embedding(actions=action)
        else:
            action_embedding = self.continuous_action_embedding(actions=action)
        return observation_embeddings, action_embedding
