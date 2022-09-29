import gym
import numpy as np
import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy


class ActorCriticPolicyWithGradients(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )

        # Instantiate new network (layer) for the gradients of the value function, with output dimension equal to the
        # number of edges in the policy network
        if type(self.net_arch) == dict:
            num_policy_edges = self.net_arch['pi'].sum()
        elif type(self.net_arch) == list:
            if type(self.net_arch[-1]) == dict:
                num_policy_edges = sum(self.net_arch[:-1]) + self.net_arch[-1]['pi'].sum()
            else:
                num_policy_edges = sum(self.net_arch)
        else:
            raise
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf,
                                   num_policy_edges + 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output

        # Reinstantiate optimizer in order to pass on parameters from the value-gradient network
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def predict_values_with_gradients(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get estimates of the value function and gradient of the value function, according to the current policy
        given the observations.

        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf), self.value_grad_net(latent_vf)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)

        return actions, values, log_prob

    def forward_with_gradients(self, obs: th.Tensor,
                               deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor, critic, and critic-gradient)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values, value_grads = self.value_net(latent_vf), self.value_grad_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)

        return actions, values, value_grads, log_prob
