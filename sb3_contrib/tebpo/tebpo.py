import copy
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from torch import nn
from torch.nn import functional as F
from sb3_contrib.tebpo.tensor_buffer import ValueGradientRewardsRolloutBuffer, TensorRewardsRolloutBuffer

from sb3_contrib.trpo.trpo import TRPO, TRPO_ANALYSIS
from sb3_contrib.tebpo.actor_critic_policy_with_gradients import ActorCriticPolicyWithGradients


class TEBPO_MC(TRPO):
    def _setup_model(self):
        super(TEBPO_MC, self)._setup_model()
        self.rollout_buffer = TensorRewardsRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs)

    def get_objective_and_kl_fn(self, policy, buffer):
        """
        Returns a closure that accepts a policy, and returns the objective
        value and KL divergence from the original policy.
        """
        flat_obs = buffer.observations_th.view(
            -1, buffer.observations_th.shape[-1])
        flat_actions = buffer.actions_th.view(-1, buffer.actions.shape[-1])
        flat_log_probs = buffer.log_probs_th.view(-1)

        with th.no_grad():
            # Note: is copy enough, no need for deepcopy?
            # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
            # directly to avoid PyTorch errors.
            old_distribution = copy.copy(policy.get_distribution(flat_obs))

        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            flat_actions = buffer.actions_th.view(-1)

        def objective_and_kl_fn(policy):
            distribution = policy.get_distribution(flat_obs)
            log_prob = distribution.log_prob(flat_actions)
            ratio = (th.exp(log_prob - flat_log_probs)
                     .view(buffer.log_probs_th.shape))
            advantages = (self.rollout_buffer
                          ._compute_advantages(weights=ratio))
            Qs = advantages + self.rollout_buffer.values_th
            # (ratio * Qs - self.gamma * Vnexts).sum()

            kl_div = kl_divergence(distribution, old_distribution).mean()
            # if self.normalize_advantage:
            #     # Should we really have gradients through the normalizers? Seems to work best...
            #     advantages = (advantages - advantages.mean()) \
            #         / (advantages.std() + 1e-8)
            # obj = (advantages.squeeze() * ratio).mean()

            if self.normalize_advantage:
                Qs = (Qs - Qs.detach().mean()) \
                    / (Qs.detach().std() + 1e-8)
            obj = ((ratio - self.gamma) * Qs.squeeze()
                   + self.rollout_buffer.rewards_th.squeeze()).mean()
            return obj, kl_div

        return objective_and_kl_fn


class TEBPO(TRPO):
    """
    Taylor-Expansion-Based Policy Optimization (TEBPO)

    Paper: https://arxiv.org/abs/1502.05477
    <Add new TEBPO writeup/paper>

    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    and Stable Baselines (TRPO from https://github.com/hill-a/stable-baselines)

    Introduction to TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size for the value function
    :param gamma: Discount factor
    :param cg_max_steps: maximum number of steps in the Conjugate Gradient algorithm
        for computing the Hessian vector product
    :param cg_damping: damping in the Hessian vector product computation
    :param line_search_shrinking_factor: step-size reduction factor for the line-search
        (i.e., ``theta_new = theta + alpha^i * step``)
    :param line_search_max_iter: maximum number of iteration
        for the backtracking line-search
    :param n_critic_updates: number of critic updates per policy update
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param target_kl: Target Kullback-Leibler divergence between updates.
        Should be small for stability. Values like 0.01, 0.05.
    :param sub_sampling_factor: Sub-sample the batch to make computation faster
        see p40-42 of John Schulman thesis http://joschu.net/docs/thesis.pdf
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicyWithGradients,
        # "CnnPolicy": ActorCriticCnnPolicy,
        # "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        **kwargs):
        super(TEBPO, self).__init__(policy, env, **kwargs)

    def _setup_model(self):
        super(TEBPO, self)._setup_model()
        self.rollout_buffer = ValueGradientRewardsRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.policy,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs)

    def _compute_policy_grad(self, policy_objective):
        termA = super(TEBPO, self)._compute_policy_grad(policy_objective)
        termB = (self.rollout_buffer.advantages[:, 1:] *
                 self.rollout_buffer.log_probs).sum(axis=0)
        return termA + termB

    def get_objective_and_kl_fn(self, policy, data):
        """
        Returns a closure that accepts a policy, and returns the objective
        value and KL divergence from the original policy.
        """
        advantages = data.advantages[:, 0]
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) \
                / (data.advantages.std() + 1e-8)

        with th.no_grad():
            # Note: is copy enough, no need for deepcopy?
            # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
            # directly to avoid PyTorch errors.
            old_distribution = copy.copy(
                policy.get_distribution(data.observations))

        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = data.actions.long().flatten()

        def objective_and_kl_fn(policy):
            distribution = policy.get_distribution(data.observations)
            log_prob = distribution.log_prob(actions)
            ratio = th.exp(log_prob - data.old_log_prob)
            kl_div = kl_divergence(distribution, old_distribution).mean()
            obj = (advantages * ratio).mean()
            return obj, kl_div

        return objective_and_kl_fn

    def value_loss(self, data):
        values_pred = self.policy.predict_values(data.observations)
        return F.mse_loss(data.returns, values_pred)


class TEBPO_MC_ANALYSIS(TRPO_ANALYSIS):
    def _setup_model(self):
        super(TEBPO_MC_ANALYSIS, self)._setup_model()
        self.rollout_buffer = TensorRewardsRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs)

