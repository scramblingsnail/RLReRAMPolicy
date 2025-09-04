import numpy as np


class RewardBase:
    def compute_rewards(self, observations):
        raise NotImplementedError


class WritingReward(RewardBase):
    def __init__(self, config: dict):
        self.xx = None
        # TODO: reward design.
