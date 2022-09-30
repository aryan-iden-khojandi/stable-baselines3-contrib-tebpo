from typing import Any, Dict, Generator, List, Optional, Union, Tuple

import funcy as f
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.tebpo.actor_critic_policy_with_gradients import ActorCriticPolicyWithGradients
from sb3_contrib.tebpo.utils import get_flat_grads

class TensorRewardsRolloutBuffer(RolloutBuffer):
    """
    Version of a RolloutBuffer where rewards can be tensors of shape
    `reward_shape`.
    Unfortunately much of the original code assumes flat rewards, so we copy it here
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
        self.reward_shape = reward_shape
        super(TensorRewardsRolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space,
            device=device, gae_lambda=gae_lambda, gamma=gamma,
            n_envs=n_envs)

    def reset(self) -> None:
        super(TensorRewardsRolloutBuffer, self).reset()
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
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        aug_reward = self.augment_rewards(obs, action, reward, log_prob)
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(aug_reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            next_non_terminal = next_non_terminal.reshape(-1, 1)
            delta = (self.rewards[step]
                     + self.gamma * next_values * next_non_terminal
                     - self.values[step])
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        self.last_values = last_values
        self.dones = dones


class ValueGradientRewardsRolloutBuffer(TensorRewardsRolloutBuffer):
    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 policy: ActorCriticPolicyWithGradients,
                 device: Union[th.device, str] = "auto",
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1):

        super(ValueGradientRewardsRolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space,
            reward_shape=(policy.n_actor_params + 1, ),
            device=device, gae_lambda=gae_lambda, gamma=gamma,
            n_envs=n_envs)
        self.policy = policy

    def augment_rewards(self,
                        obs: np.ndarray,
                        action: np.ndarray,
                        reward: np.ndarray,
                        log_prob: th.Tensor):
        with th.enable_grad():
            log_probs = (self.policy.get_distribution(self.to_torch(obs))
                         .log_prob(self.to_torch(action).flatten()))
            grads = np.zeros((obs.shape[0], self.policy.n_actor_params))
            for i in range(len(log_probs)):
                self.policy.zero_grad() # FIXME This is dangerous
                log_probs[i].backward(retain_graph=True)
                grads[i] = get_flat_grads(
                    self.policy, pred=lambda name: 'value' not in name)
        reward_rshp = reward.reshape(-1, 1)
        return np.hstack((reward_rshp, reward_rshp * grads))
