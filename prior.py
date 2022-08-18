"""
Implementation of a temporal prior as a generative model.
"""

import numpy as np
import torch
from utils import FLOW_PATH
import sys

class TemporalPrior():
    """
    Generative model that maps past information to a probability distribution.
    Args:
        env : Environment.
        device : Device to load the models to.
        model (string): Type of model in 'flow', 'vae', 'lscde'
        cond (string): Type of information to condition on.
        n_step (int): Number of past steps for conditioning.
        low (float): Low threshold for certainty.
        high (float): High threshold for uncertainty.
        fn (string): String representing class of mixing function.
    """

    def __init__(self, env, device, model, cond, n_step, ratio, smoothing, clamp, one_step, **kwargs):
        self.device = device
        self.model_class = model
        self.conditioning = cond
        self.n_step = n_step
        self.one_step = one_step
        flow_checkpoint = env.get_primitive_flow_checkpoint(model=model, cond=cond, n_step=n_step, ratio=ratio, one_step=one_step)
        self.flow = self.flow = torch.load(FLOW_PATH + flow_checkpoint, map_location=self.device)
        for param in self.flow.parameters():
            param.requires_grad = False
        self.flow.eval()

        self.smoothing = smoothing
        self.clamp = clamp

    def sample(self, o, last_action):
        """
        Samples from conditional density.
        Args:
            o : Last observation.
            last_actions : Previous actions.
        """
        if self.conditioning == "action":
            cond_inputs = last_action
        elif self.conditioning == "state+goal":
            cond_inputs = o
        elif self.conditioning == "state":
            cond_inputs = o
        elif self.conditioning == 'action+state':
            assert(len(o.shape) <= 2), "Cannot concatenate actions with observation"
            cond_inputs = torch.cat([last_action, o], -1)
        else:
            raise NotImplementedError
        
        num_samples = cond_inputs.shape[0]
        if self.one_step:
            cond_inputs = None

        if self.model_class == 'deterministic':
             action = self.flow(cond_inputs)
        elif self.model_class == 'gaussian':
             mu, std = self.flow(cond_inputs)
             return mu, std
        else:
            action = self.flow.sample(num_samples=num_samples, cond_inputs=cond_inputs)

        # clamp actions
        action = torch.clamp(action, -(1-0.01-self.clamp), (1-0.01-self.clamp)) + torch.clamp(torch.randn_like(action) * 0.01, -self.clamp, self.clamp)

        # smooth prior
        action = torch.where(torch.rand(size=(action.shape[0], 1), device=self.device) < self.smoothing, torch.rand(size=action.shape, device=self.device)*2 - 1, action)
        action = torch.nan_to_num(action)
        return action 
