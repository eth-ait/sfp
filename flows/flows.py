"""
Implementation of torch modules for generative flows. Most of the codebase was adapted
from Ilya Kostrikov's implementation at github.com/ikostrikov/pytorch-flows
"""

import math
import torch
import torch.nn as nn


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 shared=False,
                 pre_mlp=False,
                 pre_mlp_units=64):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask
        self.shared = shared
        self.pre_mlp = pre_mlp
        self.pre_mlp_units = pre_mlp_units

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + (num_cond_inputs if not self.pre_mlp else self.pre_mlp_units)
            if self.pre_mlp:
                self.mlp = nn.Sequential(
                    nn.Linear(num_cond_inputs, self.pre_mlp_units), t_act_func(),
                    nn.Linear(self.pre_mlp_units, self.pre_mlp_units), t_act_func(),
                    nn.Linear(self.pre_mlp_units, self.pre_mlp_units)
                    )
        else:
            total_inputs = num_inputs

        if self.shared:
            self.shared_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_hidden), t_act_func())
            self.scale_net = nn.Sequential(nn.Linear(num_hidden, num_inputs))
            self.translate_net = nn.Sequential(nn.Linear(num_hidden, num_inputs))
        else:
            self.scale_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_inputs))
            self.translate_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask
        if cond_inputs is not None:
            cond_inputs = cond_inputs if not self.pre_mlp else self.mlp(cond_inputs)
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
        
        if mode == 'direct':
            if self.shared:
                masked_inputs = self.shared_net(masked_inputs)
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            if self.shared:
                masked_inputs = self.shared_net(masked_inputs)
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class CNNCouplingLayer(CouplingLayer):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803) conditioned on images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pre_mlp:
            self.mlp = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 16, 3, 1), nn.ReLU(), nn.Flatten(),
                nn.Linear(2304, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, kwargs['pre_mlp_units']))


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples


class Deterministic(nn.Module):
    """ Simple MLP that learns deterministic mappings instead of conditional distributions. """
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_outputs), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        device = next(self.parameters()).device
        cond_inputs = cond_inputs.to(device)
        return self.forward(cond_inputs)


class Gaussian(nn.Module):
    """ Simple MLP that learns deterministic mappings instead of conditional distributions. """
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU()
        )
        self.mu_net = nn.Linear(num_hidden, num_outputs)
        self.log_std_net = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        feat = self.net(x)
        return self.mu_net(feat), self.log_std_net(feat)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        device = next(self.parameters()).device
        cond_inputs = cond_inputs.to(device)
        return self.forward(cond_inputs)