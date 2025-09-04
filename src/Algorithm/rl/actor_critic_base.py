import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from ..net import ActorBase, CriticBase
from ..replay_buffer import init_replay_buffer


class ActorCriticBase:
    def __init__(self, env, sampler_num: int, actor: ActorBase = None, critic: CriticBase = None,
                 actor_optimizer: Optimizer = None, critic_optimizer: Optimizer = None):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.env = env
        self.sampler_num = sampler_num

    def select_action(self, current_state, **kwargs):
        if self.actor is None:
            raise AttributeError('Actor Network not implemented.')
        raise NotImplementedError('Please override this function for selecting action.')

    def sample_from_env(self, action):
        if self.env is None:
            raise AttributeError('Environment missing.')
        raise NotImplementedError('Please override this function for sampling.')
