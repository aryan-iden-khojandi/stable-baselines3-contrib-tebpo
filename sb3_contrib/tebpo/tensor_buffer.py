from typing import Any, Dict, Generator, List, Optional, Union, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from functorch import vmap, make_functional, grad

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples


class TensorRewardsRolloutBuffer(RolloutBuffer):
    """
    Version of a RolloutBuffer where rewards can be tensors of shape
    `reward_shape`.
    """
    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 reward_shape: Tuple,
                 device: Union[th.device, str] = "auto",
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1):
        super(TensorRewardsRolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space,
            device=device, gae_lambda=gae_lambda, gamma=gamma,
            n_envs=n_envs)
        self.reward_shape = reward_shape

    def reset(self) -> None:
        super(TensorRewardsRolloutBuffer).reset()
        reward_initr = lambda: np.zeros(
            (self.buffer_size, self.n_envs, *self.reward_shape),
            dtype=np.float32)
        self.rewards = reward_initr()
        self.returns = reward_initr()
        self.values = reward_initr()
        self.advantages = reward_initr()

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None
                     ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].reshape(-1, *self.reward_shape),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].reshape(-1, *self.reward_shape),
            self.returns[batch_inds].reshape(-1, *self.reward_shape),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def augment_rewards(self,
                        obs: np.ndarray,
                        action: np.ndarray,
                        reward: np.ndarray,
                        log_prob: th.Tensor):
        "Turn one-dimensional reward into a reward of size reward_shape"
        return reward

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        aug_reward = self.augment_rewards(obs, action, reward, log_prob)
        super(TensorRewardsRolloutBuffer, self).add(
            obs, action, aug_reward, episode_start, value, log_prob)


class ValueGradientRewardsRolloutBuffer(TensorRewardsRolloutBuffer):
    def __init__(self, *args, policy: nn.Module, **kwargs):
        super(ValueGradientRewardsRolloutBuffer, self).__init__(
            *args, **kwargs)
        self.policy = policy

    def augment_rewards(self,
                        obs: np.ndarray,
                        action: np.ndarray,
                        reward: np.ndarray,
                        log_prob: th.Tensor):
        policy_params = {k: v for (k, v) in self.policy.named_parameters()
                         if "value" not in k}
        # Using the policy gradient trick here
        grad = th.autograd.grad(log_prob, policy_params, retain_graph=True)
        return reward * grad
