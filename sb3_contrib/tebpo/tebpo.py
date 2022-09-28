import copy
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from functorch import vmap, make_functional, grad
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from torch import nn
from torch.nn import functional as F

from sb3_contrib.trpo.trpo import TRPO
from sb3_contrib.tebpo.actor_critic_policy_with_gradients import ActorCriticPolicyWithGradients


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

    def value_loss(self, data):
        actions, value_preds, value_grad_preds, log_probs = \
            self.policy.forward_with_gradients(data.observations)
        value_loss = F.mse_loss(data.returns, value_preds.flatten())
        th.autograd.functional.Jacobian(log_probs, )

        # Compute per-sample policy gradients
        func_model, params = make_functional(self.policy)
        def get_log_probs(params, obs, acts):
            return func_model(params, obs).log_prob(acts)
        reward_weights = get_log_probs(params, obs, acts)
        grads = vmap(grad(get_log_probs), in_dims=(0, None, None))(
            data.observations, data.actions)

        value_grad_targets = self.compute_returns_and_advantage()
        value_grad_loss = F.mse_loss(value_grad_targets, value_grad_preds)
        return value_loss + value_grad_loss

    def compute_returns_and_advantage(self,
                                      buffer: RolloutBuffer,
                                      last_values: th.Tensor,
                                      dones: np.ndarray,
                                      weights: np.ndarray,
                                      values: np.ndarray) -> np.ndarray:
        "See documentation in stable_baselines3.buffers.RolloutBuffer"
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        advantages = np.zeros(buffer.buffer_size)
        for step in reversed(range(buffer.buffer_size)):
            if step == buffer.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - buffer.episode_starts[step + 1]
                next_values = values[step + 1]
            delta = (weights * buffer.rewards[step]
                     + buffer.gamma * next_values * next_non_terminal
                     - values[step])
            last_gae_lam = delta + buffer.gamma * buffer.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        return advantages + values

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) -> Tuple[List[nn.Parameter], th.Tensor, th.Tensor, List[Tuple[int, ...]]]:
        """
        Compute actor gradients for kl div and surrogate objectives.

        :param kl_div: The KL divergence objective
        :param policy_objective: The surrogate objective ("classic" policy gradient)
        :return: List of actor params, gradients and gradients shape.
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # The policy objective is also called surrogate objective
        policy_objective_gradients = []
        # Contains the gradients of the KL divergence
        grad_kl = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters
        grad_shape = []
        # Contains the parameters which have non-zeros KL divergence gradients
        # The list is used during the line-search to apply the step to each parameters
        actor_params = []

        for name, param in self.policy.named_parameters():
            # Skip parameters related to value function based on name
            # this work for built-in policies only (not custom ones)
            if "value" in name:
                continue

            # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            # If the gradient is not zero (not None), we store the parameter in the actor_params list
            # and add the gradient and its shape to grad_kl and grad_shape respectively
            if kl_param_grad is not None:
                # If the parameter impacts the KL divergence (i.e. the policy)
                # we compute the gradient of the policy objective w.r.t to the parameter
                # this avoids computing the gradient if it's not going to be used in the conjugate gradient step
                policy_objective_grad_term_a, *_ = th.autograd.grad(
                    policy_objective, param,
                    retain_graph=True, only_inputs=True)
                # TODO: Figure out the last state, dones
                policy_objective_grad_term_b = \
                    self.rollout_buffer.compute_returns_and_advantage(
                        self.policy.forward_with_grads(
                            obs=self.rollout_buffer.get(batch_size=self.batch_size),
                            deterministic=True
                        )[2])

                policy_objective_grad = policy_objective_grad_term_a + policy_objective_grad_term_b

                grad_shape.append(kl_param_grad.shape)
                grad_kl.append(kl_param_grad.reshape(-1))
                policy_objective_gradients.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)

        # Gradients are concatenated before the conjugate gradient step
        policy_objective_gradients = th.cat(policy_objective_gradients)
        grad_kl = th.cat(grad_kl)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape
