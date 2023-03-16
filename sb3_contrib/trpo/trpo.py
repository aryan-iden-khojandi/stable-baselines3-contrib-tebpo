import copy
import pickle
import os, time, csv
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.trpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from sb3_contrib.trpo.utils import \
    get_flat_grads, get_flat_params, set_flat_params, is_actor


class TRPO(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization (TRPO)

    Paper: https://arxiv.org/abs/1502.05477
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
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        init_policy_file: str = None,
        fixed_policy_file: str = None,
        experiment_index: int = None,
        model_save_path: str = None,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,  # entropy bonus is not used by TRPO
            vf_coef=0.0,  # value function is optimized separately
            max_grad_norm=0.0,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.model_save_path = model_save_path

        if init_policy_file:
            with open(init_policy_file, 'rb') as f:
                self.init_policy = pickle.load(f)
                print("Found and read init policy.")
        else:
            self.init_policy = None
            print("Did not find and read init policy.")
        if fixed_policy_file:
            with open(fixed_policy_file, 'rb') as f:
                self.fixed_policy = pickle.load(f)
                print("Found and read fixed policy.")
        else:
            self.fixed_policy = None
            print("Did not find and read fixed policy.")

        self.normalize_advantage = normalize_advantage
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            if normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        # Conjugate gradients parameters
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        # Backtracking line search parameters
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        self.target_kl = target_kl
        self.n_critic_updates = n_critic_updates
        self.sub_sampling_factor = sub_sampling_factor

        if _init_setup_model:
            self._setup_model()

        if self.init_policy:
            set_flat_params(self.policy, get_flat_params(self.init_policy))

        self.experiment_index = experiment_index

    def _compute_policy_grad(self, policy_objective):
        policy_objective.backward(retain_graph=True)
        return get_flat_grads(self.policy, pred=is_actor)

    def _compute_kl_grad(self, kl_div: th.Tensor):
        """
        Compute actor gradients for kl div and surrogate objectives.  Edit: It does not compute gradients for the
        surrogate objective, as this is handled in _compute_policy_grad(), right?

        :param kl_div: The KL divergence objective
        :param policy_objective: The surrogate objective ("classic" policy gradient)
        :return: List of actor params, gradients and gradients shape.
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # Contains the gradients of the KL divergence
        grad_kl = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters

        for name, param in self.policy.named_parameters():
            # Skip parameters related to value function based on name
            # this work for built-in policies only (not custom ones)
            if "value" in name:  # Note: This is rather brittle
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
                # policy_objective_grad, *_ = th.autograd.grad(policy_objective, param, retain_graph=True, only_inputs=True)
                grad_kl.append(kl_param_grad.reshape(-1))
                # policy_objective_gradients.append(
                #     policy_objective_grad.reshape(-1))

        # Gradients are concatenated before the conjugate gradient step
        grad_kl = th.cat(grad_kl)

        return grad_kl

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []

        # value_losses = self.update_critic(
        #     [param for name, param in self.policy.named_parameters()
        #      if "value" not in name])
        # This will only loop once (get all data in one go)
        # Re-sample the noise matrix because the log_std has changed
        # if self.use_sde:
        #     # batch_size is only used for the value function
        #     self.policy.reset_noise(actions.shape[0])

        # KL divergence
        obj_and_kl_fn = self.get_objective_and_kl_fn(
            self.policy,
            self.rollout_buffer)
            # rollout_data)
        policy_objective, kl_div = obj_and_kl_fn(self.policy)

        # Surrogate & KL gradient
        actor_params = [param for name, param
                        in self.policy.named_parameters()
                        if is_actor(name)]
        original_actor_params = get_flat_params(
            self.policy, pred=is_actor).detach().clone()

        self.policy.optimizer.zero_grad()  # We need to do this because otherwise, it will add the new gradients to the
                                           # to the previous ones rather than replace the previous ones?
        policy_objective_gradients = self._compute_policy_grad(
            policy_objective)

        # This zero gradding is a bit iffy
        self.policy.optimizer.zero_grad()
        grad_kl = self._compute_kl_grad(kl_div)

        # Hessian-vector dot product function used in the conjugate gradient step
        hessian_vector_product_fn = partial(
            self.hessian_vector_product, actor_params, grad_kl)

        # Computing search direction
        search_direction = conjugate_gradient_solver(
            hessian_vector_product_fn,
            policy_objective_gradients,
            max_iter=self.cg_max_steps,
        )

        # Maximal step length
        line_search_max_step_size = 2 * self.target_kl
        line_search_max_step_size /= th.matmul(
            search_direction,
            hessian_vector_product_fn(search_direction, retain_graph=False)
        )
        line_search_max_step_size = th.sqrt(line_search_max_step_size)

        is_line_search_success = False
        with th.no_grad():
            # Line-search (backtracking)
            # Note that the returned method mutates self.policy by calling set_flat_params()
            linesearch_obj_fn = self.get_linesearch_obj_fn(
                original_actor_params,
                search_direction,
                obj_and_kl_fn)

            # Note that this line mutates self.policy by calling set_flat_params() within linesearch_obj_fn()
            is_line_search_success, new_policy, new_obj, kl = \
                self.linesearch(linesearch_obj_fn,
                                line_search_max_step_size)
            line_search_results.append(is_line_search_success)

            if not is_line_search_success:
                # If the line-search wasn't successful we revert to the original parameters
                set_flat_params(self.policy, original_actor_params,
                                pred=is_actor)
                policy_objective_values.append(policy_objective.item())
                kl_divergences.append(0)
            else:
                policy_objective_values.append(new_obj.item())
                kl_divergences.append(kl.item())

        value_losses = self.update_critic(actor_params)
        self._n_updates += 1

        explained_var = explained_variance(
            np.asarray(self.rollout_buffer.values.flatten()),
            np.asarray(self.rollout_buffer.returns.flatten()))

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def get_objective_and_kl_fn(self, policy, data_source):
        """
        Returns a closure that accepts a policy, and returns the objective
        value and KL divergence from the original policy.
        """

        data = next(data_source.get(batch_size=None))

        # Optional: sub-sample data for faster computation
        if self.sub_sampling_factor > 1:
            data = RolloutBufferSamples(
                data.observations[:: self.sub_sampling_factor],
                data.actions[:: self.sub_sampling_factor],
                None,  # old values, not used here
                data.old_log_prob[:: self.sub_sampling_factor],
                data.advantages[:: self.sub_sampling_factor],
                None,  # returns, not used here
            )

        advantages = data.advantages
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
        else:
            actions = data.actions

        episode_idx = data_source.to_torch(
            data_source.episode_starts
            .transpose()
            .flatten()
            .cumsum()
        ).to(th.int64)
        episode_idx = episode_idx - episode_idx[0]

        def objective_and_kl_fn(policy):
            distribution = policy.get_distribution(data.observations)
            log_prob = distribution.log_prob(actions)
            ratio = th.exp(log_prob - data.old_log_prob)
            kl_div = kl_divergence(distribution, old_distribution).mean()
            # obj = (advantages * ratio).mean()
            advantages_with_importance = (ratio - 1.0) * data.returns

            episode_sums = th.zeros(int(episode_idx.max()) + 1)
            episode_sums.scatter_add_(
                0, episode_idx, advantages_with_importance)
            obj = episode_sums.mean()

            return obj, kl_div

        return objective_and_kl_fn

    def get_linesearch_obj_fn(self,
                              actor_params: th.Tensor,
                              search_direction: th.Tensor,
                              obj_and_kl_fn: Callable) -> Callable:
        """
        Returns a closure f(coeff) that evaluates obj and kl for the policy
        original_params + search_direction * coeff
        """
        def linesearch_obj_fn(coeff: float):
            """Return a tuple of policy, objective, constraint"""
            set_flat_params(self.policy,
                            actor_params + coeff * search_direction,
                            pred=is_actor)
            # self.update_params(
            #     actor_params, original_params, grad_shape, flat_direction)
            obj, kl = obj_and_kl_fn(self.policy)
            return (self.policy, obj, kl)
        return linesearch_obj_fn

    def update_params(self, actor_params, original_params, grad_shape,
                      flat_direction):
        start_idx = 0
        # Applying the scaled step direction
        for param, original_param, shape in zip(
                actor_params, original_params, grad_shape):
            n_params = param.numel()
            param.data = (
                original_param.data
                + flat_direction[start_idx:(start_idx + n_params)].view(shape)
            )
            start_idx += n_params

    def linesearch(self, obj_fn, max_step_size):
        success = False
        step_size = max_step_size
        _, init_obj, _ = obj_fn(0.)
        for _ in range(self.line_search_max_iter):
            new_policy, new_obj, new_kl = obj_fn(step_size)
            if ((new_kl < self.target_kl) and (new_obj > init_obj)):
                success = True
                break
            else:
                step_size *= self.line_search_shrinking_factor
        return success, new_policy, new_obj, new_kl

    def update_critic(self, actor_params):
        # Critic update
        value_losses = []
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                value_loss = self.value_loss(rollout_data)
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()
        return value_losses

    def value_loss(self, data):
        values_pred = self.policy.predict_values(data.observations)
        return F.mse_loss(data.returns, values_pred.flatten())

    def hessian_vector_product(
        self, params: List[nn.Parameter], grad_kl: th.Tensor, vector: th.Tensor, retain_graph: bool = True
    ) -> th.Tensor:
        """
        Computes the matrix-vector product with the Fisher information matrix.

        :param params: list of parameters used to compute the Hessian
        :param grad_kl: flattened gradient of the KL divergence between the old and new policy
        :param vector: vector to compute the dot product the hessian-vector dot product with
        :param retain_graph: if True, the graph will be kept after computing the Hessian
        :return: Hessian-vector dot product (with damping)
        """
        jacobian_vector_product = (grad_kl * vector).sum()
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TRPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OnPolicyAlgorithm:

        model_to_return = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(self.policy, f)

        return model_to_return


class TRPO_ANALYSIS(TRPO):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        init_policy_file: str = None,
        fixed_policy_file: str = None,
        experiment_index: int = None
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            gamma,
            cg_max_steps,
            cg_damping,
            line_search_shrinking_factor,
            line_search_max_iter,
            n_critic_updates,
            gae_lambda,
            use_sde,
            sde_sample_freq,
            normalize_advantage,
            target_kl,
            sub_sampling_factor,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
            init_policy_file,
            fixed_policy_file,
            experiment_index
        )

        self.normalize_advantage = False

    def train(self) -> None:
        """
        Log surrogate-objective values using the currently gathered rollout buffer, on a fixed policy (i.e. without
        updating the policy).
        """
        # KL divergence
        obj_and_kl_fn = self.get_objective_and_kl_fn(
            self.policy,
            self.rollout_buffer)
            # rollout_data)
        policy_objective, kl_div = obj_and_kl_fn(self.fixed_policy)

        set_flat_params(self.policy, get_flat_params(self.fixed_policy))

        # Logs
        self.logger.record("train/policy_objective", policy_objective.item())

        if self.experiment_index is not None:
            filename = "experimental_results/{exp_idx}/{model_name}_{timestamp}".format(exp_idx=self.experiment_index,
                                                                   model_name=self.__class__.__name__,
                                                                   timestamp=time.time())
            with open(filename, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                row = ['policy_objective', policy_objective.item()]
                writer.writerow(row)

        self.logger.record("train/kl_divergence_loss", kl_div)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
