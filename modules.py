"""
Implementation of all relevant neural network architectures in pytorch.
Some modules were borrowed from Spinning Up's SAC implementation.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Builds a simple MLP.
    Args:
        sizes : List of integers representing hidden layer sizes.
        activation : Activation function for hidden layers.
        output_activation : Activation for last layer.
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):
    """
    Stochastic actor that processes its input through an MLP and outputs a Gaussian policy.
    Samples from the policy are then squashed through a tanh function.
    Args:
        obs_dim (int): Size of observations.
        act_dim (int): Size of actions.
        hidden_sizes : List of integers representing sizes of hidden layers.
        activation : Activation for hidden layers.
        act_limit (float): Symmetric maximum absolute value of actions.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True, action=None, with_distr=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        if action is not None:
            # Only return logprob of action
            policy_output = torch.atanh(action / self.act_limit)
            log_pi = pi_distribution.log_prob(policy_output).sum(axis=-1)
            return log_pi

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        if with_distr:
            return pi_action, logp_pi, log_std, mu, std
        return pi_action, logp_pi, log_std


class MLPQFunction(nn.Module):
    """
    Q-function parameterized as a MLP.
    Args:
        obs_dim (int): Size of observations.
        act_dim (int): Size of actions.
        hidden_sizes : List of integers representing sizes of hidden layers.
        activation : Activation for hidden layers.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    """
    Module containing a policy network and two Q-networks, implementing an actor critic architecture.
    Args:
        obs_dim (int): Size of observations.
        act_dim (int): Size of actions.
        hidden_sizes : List of integers representing sizes of hidden layers for all networks.
        activation : Activation for hidden layers in all networks.
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 lambda_param=False, lambda_init=0.87, activation=nn.ReLU):
        super().__init__()

        if isinstance(observation_space, gym.spaces.dict.Dict):
            obs_dim = observation_space['observation'].shape[0] + observation_space['desired_goal'].shape[0]
        else:
            obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        if lambda_param:
            sigmoid_bias_init = math.log(lambda_init/(1-lambda_init))
            self.lambda_ = torch.tensor((sigmoid_bias_init,), requires_grad=True)
            self.lambda_net = lambda x: torch.sigmoid(self.lambda_)
            self.lambda_params = [self.lambda_]
        else:
            self.lambda_net = mlp([obs_dim, 128, 128, 1], activation=nn.ReLU, output_activation=nn.Sigmoid)
            state_dict = self.lambda_net.state_dict()
            sigmoid_bias_init = math.log(lambda_init/(1-lambda_init))
            state_dict[list(state_dict.keys())[-1]].fill_(sigmoid_bias_init)
            self.lambda_net.load_state_dict(state_dict)
            self.lambda_params = list(self.lambda_net.parameters())

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, logp_pi, log_std = self.pi(obs, deterministic, True)
            return a, logp_pi, log_std


class CNNActorCritic(nn.Module):
    """
    Module containing a policy network and two Q-networks, implementing an actor critic architecture.
    All networks take images as input and share the same convolutional backbone.
    Args:
        observation_space : Observation space of the environment.
        action_space : Action space of the environment.
        hidden_sizes : List of integers representing sizes of hidden dense layers.
        activation : Activation for hidden dense layers.
    """

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256), lambda_param=False, lambda_init=0.87, activation=nn.ReLU):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Build policy and value functions
        self.cnn = BaseCNN()
        with torch.no_grad():
            obs_dim = self.cnn(torch.zeros((1, 3, 64, 64))).shape[-1]
        self.pi = SquashedGaussianCNNActor(self.cnn, obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = CNNQFunction(self.cnn, obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = CNNQFunction(self.cnn, obs_dim, act_dim, hidden_sizes, activation)

        self.lambda_net = mlp([obs_dim, 128, 128, 1], activation=nn.ReLU, output_activation=nn.Sigmoid)
        state_dict = self.lambda_net.state_dict()
        sigmoid_bias_init = math.log(lambda_init/(1-lambda_init))
        state_dict[list(state_dict.keys())[-1]].fill_(sigmoid_bias_init)
        self.lambda_net.load_state_dict(state_dict)
        self.lambda_net = CNNLambdaNetwork(cnn=self.cnn, lambda_net=self.lambda_net)
        self.lambda_params = list(self.lambda_net.parameters())

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, logp_pi, log_std = self.pi(obs, deterministic, True)
            return a, logp_pi, log_std


class CNNLambdaNetwork(nn.Module):
    def __init__(self, cnn, lambda_net):
        super().__init__()
        self.cnn = cnn
        self.lambda_net = lambda_net

    def forward(self, obs):
        return self.lambda_net.forward(self.cnn(obs))


class CNNQFunction(MLPQFunction):
    """
    A MLPQFunction including a CNN backbone to extract representations from RGB inputs.
    """
    def __init__(self, cnn, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation)
        self.cnn = cnn

    def forward(self, obs, act):
        return super().forward(self.cnn(obs), act)


class SquashedGaussianCNNActor(SquashedGaussianMLPActor):
    """
    A SquashedGaussianMLPActor including a CNN backbone to extract representations from RGB inputs.
    """
    def __init__(self, cnn, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.cnn = cnn


    def forward(self, obs, deterministic=False, with_logprob=True):
        return super().forward(self.cnn(obs), deterministic, with_logprob)


class BaseCNN(nn.Module):
    """
    A simple CNN.
    """
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, 1), nn.ReLU(), nn.Flatten())
    
    def forward(self, x):
        return self.net(x)
