'''
Environment wrappers with a gym3 API.
'''

import numpy as np
from scipy.stats import norm
import torch
from gym.spaces import Box
from utils import FLOW_PATH


def wrap_env_fn(env_fn, args):
    '''
    Takes a method creating an environment and wraps it. The returned method creates wrapped environments.
    Args:
        env_fn: Environment constructor.
        args: Arguments containing environment parameters.
    '''
    device = 'cuda' if args.gpu else 'cpu'
    if args.action_prior == 'nstep':
        return lambda: NStepWrapper(env_fn, args.n_step)
    elif args.action_prior == 'diagonal':
        return lambda: ActionRepeatWrapper(env_fn, args.n_step)
    elif args.action_prior == 'densediagonal':
        return lambda: DenseActionRepeatWrapper(env_fn, args.n_step)
    elif args.action_prior == 'ez-greedy':
        return lambda: EzGreedyWrapper(env_fn)
    elif args.action_prior == 'flow':
        return lambda: FlowWrapper(env_fn, args.n_step, args.env, args.cond, device)
    elif args.action_prior == 'gaussian_flow':
        return lambda: GaussianFlowWrapper(env_fn, args.n_step, args.env, args.cond, device)
    elif args.action_prior == 'unbounded_flow':
        return lambda: UnboundedFlowWrapper(env_fn, args.n_step, args.env, args.cond, device)
    elif args.action_prior == 'scaled_flow':
        return lambda: ScaledFlowWrapper(env_fn, args.n_step, args.env, args.cond, device)
    elif args.action_prior == 'ppf_flow':
        return lambda: PpfFlowWrapper(env_fn, args.n_step, args.env, args.cond, device)
    elif args.action_prior == 'parrot_state':
        return lambda: ParrotWrapper(env_fn, name=args.env, image=False, device=device, one_step=args.one_step)
    elif args.action_prior == 'parrot_image':
        return lambda: ParrotWrapper(env_fn, name=args.env, image=True, device=device)
    elif args.action_prior == 'parrot_action':
        return lambda: ActionParrotWrapper(env_fn, name=args.env, device=device)
    elif args.action_prior == 'none':
        return lambda: env_fn()
    raise NotImplementedError(args.action_prior)


class GaussianBox(Box):
    '''
    Gym Box sampled according to a Gaussian instead of a uniform distribution.
    '''
    def sample(self):
        sample = np.random.normal(size=self.shape)
        return np.clip(sample, self.low, self.high).astype(self.dtype)


class NStepWrapper:
    '''
    Reduces the frequency of control in an environment. The wrapped environment takes n actions at each step and
    enacts them all before returning the final observation and the cumulative reward.
    Args:
        env_fn : Environment constructor.
        n (int): Number of actions to execute for each control step.
    '''
    def __init__(self, env_fn, n=2):
        self._env = env_fn()
        self.n = n
        assert n > 0
        inner = self._env.action_space
        self.action_space = Box(low=np.concatenate([inner.low]*self.n),
                                high=np.concatenate([inner.high]*self.n),
                                dtype=inner.dtype)

    def __getattr__(self, attr):
        return getattr(self._env, attr)

    def step(self, action):
        tot_rew = 0.
        for a in np.split(action, self.n):
            obs, rew, done, info = self._env.step(a)
            tot_rew += rew
        return obs, tot_rew, done, info

    def compute_reward(self, achieved_goal, goal, info):
        return self._env.compute_reward(achieved_goal, goal, info) * self.n


class ActionRepeatWrapper:
    '''
    Reduces the frequency of control in an environment by repeating a single action for n steps.
    Each step returns the final observation and the cumulative reward.
    Args:
        env_fn : Environment constructor.
        n (int): Number of times to repeat action.
    '''

    def __init__(self, env_fn, n=2):
        self._env = env_fn()
        self.n = n
        self.action_space = self._env.action_space

    def __getattr__(self, attr):
        return getattr(self._env, attr)

    def step(self, action):
        tot_rew = 0.
        for _ in range(self.n):
            obs, rew, done, info = self._env.step(action)
            tot_rew += rew
        return obs, tot_rew, done, info

    def compute_reward(self, achieved_goal, goal, info):
        return self._env.compute_reward(achieved_goal, goal, info) * self.n


class DenseActionRepeatWrapper:
    '''
    Reduces the frequency of control in an environment by repeating a single action for n steps.
    Each step returns the final observation and the cumulative reward.
    Args:
        env_fn : Environment constructor.
        n (int): Number of times to repeat action.
    '''

    def __init__(self, env_fn, n=2):
        self._env = env_fn()
        self.n = n
        self.repeat_for = 0

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    def reset(self):
        self.repeat_for = 0
        return self._env.reset()

    def step(self, action):
        if self.repeat_for == 0:
            self.last_action = action
            self.repeat_for = self.n-1
        else:
            action = self.last_action
            self.repeat_for = self.repeat_for - 1
        return self._env.step(action)


class EzGreedyWrapper:
    '''
    Reduces the frequency of control in an environment by repeating a single action for n steps.
    Each step returns the final observation and the cumulative reward.
    Args:
        env_fn : Environment constructor.
        n (int): Number of times to repeat action.
    '''

    def __init__(self, env_fn, mu=2., epsilon=0.3):
        self._env = env_fn()
        self.mu = mu
        self.epsilon = epsilon
        self.last_action=None

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    def reset(self):
        self.repeat_for = 0
        return self._env.reset()

    def step(self, action):
        if self.repeat_for == 0:
            if np.random.uniform() < self.epsilon:
                self.last_action = action
                self.repeat_for = int(np.random.zipf(self.mu)) - 1
        else:
            action = self.last_action
            self.repeat_for = self.repeat_for - 1
        return self._env.step(action)


class FlowWrapper:
    '''
    Wrapper that lets the agent act in the "latent" space of an NVP Flow and maps actions back to the original action space.
    Supports taking multiple consecutive actions per control step, as well as conditioning the flow on past actions. However,
    conditioning on past actions is not sound as the agent cannot predict the transformation of the action space.
    Args:
        env_fn : Environment constructor.
        n (int): Number of actions to execute for each control step.
        name (string): Name of the environment.
        cond (bool): Whether to condition the flow on past actions.
        device (string): Device for storing torch models.
    '''
    def __init__(self, env_fn, n, name, cond, device):
        self._env = env_fn()
        self.n = n
        self.device = device
        assert n > 0
        assert (not cond) or (n > 1)
        self.cond = cond
        flow_checkpoint = self._env.get_primitive_flow_checkpoint(model='flow', cond=cond, n_step=self.n)
        if self.cond:  # conditional flow forces number of steps to one
            self.n = 1
        self.inner = self._env.action_space
        self.action_space = Box(low=np.concatenate([self.inner.low]*self.n),
                                high=np.concatenate([self.inner.high]*self.n),
                                dtype=self.inner.dtype)
        self.flow = torch.load(FLOW_PATH + flow_checkpoint)
        for param in self.flow.parameters():
            param.requires_grad = False
        self.flow.eval()

    def __getattr__(self, attr):
        return getattr(self._env, attr)

    def reset(self):
        if self.cond:
           dataset = self._env.get_primitive_dataset()['actions']
           self.last_action = dataset[np.random.choice(len(dataset)), :].astype(np.float32)
        return self._env.reset()

    def step(self, action):
        tot_rew = 0.
        with torch.no_grad():
            # transform action
            cond_inputs = torch.tensor(self.last_action, device=self.device).unsqueeze(0) if self.cond else None
            action = self.flow(torch.tensor(action, device=self.device).unsqueeze(0), cond_inputs=cond_inputs, mode='inverse')[0].squeeze().numpy()
            action = np.clip(action, np.concatenate([self.inner.low]*self.n), np.concatenate([self.inner.high]*self.n))
        for a in np.split(action, self.n):
            obs, rew, done, info = self._env.step(a)
            self.last_action = a
            tot_rew += rew
        return obs, tot_rew, done, info

    def compute_reward(self, achieved_goal, goal, info):
        return self._env.compute_reward(achieved_goal, goal, info) * self.n


class GaussianFlowWrapper(FlowWrapper):
    '''
    FlowWrapper using a GaussianBox instead of a default Box.
    '''
    def __init__(self, env_fn, n, name, cond):
        super().__init__(env_fn, n, name, cond)
        self.action_space = GaussianBox(low=np.concatenate([self.inner.low]*self.n),
                                        high=np.concatenate([self.inner.high]*self.n),
                                        dtype=self.inner.dtype)


class ScaledFlowWrapper(FlowWrapper):
    '''
    FlowWrapper using a GaussianBox instead of a default Box, with some adjustments to allow the agent to sample unlikely actions.
    '''
    def __init__(self, env_fn, n, name, cond):
        super().__init__(env_fn, n, name, cond)
        self.action_space = GaussianBox(low=np.concatenate([self.inner.low * 3]*self.n),
                                        high=np.concatenate([self.inner.high * 3]*self.n),
                                        dtype=self.inner.dtype)


class UnboundedFlowWrapper(FlowWrapper):
    '''
    FlowWrapper using a GaussianBox instead of a default Box. Unboundedness allows the actor to sample any point in the
    "latent" space of the flow.
    '''
    def __init__(self, env_fn, n, name, cond):
        super().__init__(env_fn, n, name, cond)
        self.action_space = Box(low=-np.inf, high=np.inf,
                                shape=tuple(self.n * s for s in self.inner.shape),
                                dtype=self.inner.dtype)


class PpfFlowWrapper(FlowWrapper):
    '''
    FlowWrapper using the inverse PDF of a gaussian to transform an uniform action distribution into a Gaussian.
    It can be used to sample the "latent" space of the flow according to its prior.
    '''
    def step(self, action):
        margin = .05  # TODO:tune
        scaled_action = margin + (1.0 - 2*margin) * (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        return super().step(norm.ppf(scaled_action).astype(np.float32))


class ParrotWrapper:
    """
    Environment wrapper using a state-conditional flowto transform its action space as in arxiv.org/abs/2011.10024
    Supports both vector and RGB states.
    """
    def __init__(self, env_fn, name, image, device, one_step=False):
        self._env = env_fn()
        self.device = device
        self.one_step = one_step
        flow_checkpoint = f'/{self._env._short_name}{"_primitive_state" if not image else "_primitive_image"}.pt'
        if self.one_step:
            flow_checkpoint = flow_checkpoint[:-3] +'_one_step.pt'
        self.flow = torch.load(FLOW_PATH + flow_checkpoint, map_location=self.device)
        for param in self.flow.parameters():
            param.requires_grad = False
        self.flow.eval()
        self.last_state = None

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    def reset(self):
        self.last_state = self._env.reset()
        return self.last_state.copy()

    def step(self, action):
        # Transform action from uniform to the prior distribution of the flow
        margin = .05  # TODO:tune
        scaled_action = margin + (1.0 - 2*margin) * (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        action = norm.ppf(scaled_action).astype(np.float32)

        with torch.no_grad():
            if isinstance(self.last_state, dict):
                cond_inputs = torch.tensor(np.concatenate([self.last_state['observation'], self.last_state['desired_goal']], -1), dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                cond_inputs = torch.tensor(self.last_state.copy(), device=self.device).unsqueeze(0).float()
            if self.one_step:
                cond_inputs = None
            action = self.flow(torch.tensor(action, device=self.device).unsqueeze(0), cond_inputs=cond_inputs, mode='inverse')[0].squeeze().detach().cpu().numpy()
            action = np.clip(action, self._env.action_space.low, self._env.action_space.high)
            action = np.nan_to_num(action)
            s, r, d, i = self._env.step(action)
            self.last_state = s.copy()
        return s, r, d, i


class ActionParrotWrapper:
    def __init__(self, env_fn, name, device):
        self._env = env_fn()
        self.device = device
        flow_checkpoint = f'/{self._env._short_name}_primitive_flow_action_1.pt'
        self.flow = self.flow = torch.load(FLOW_PATH + flow_checkpoint, map_location=self.device)
        for param in self.flow.parameters():
            param.requires_grad = False
        self.flow.eval()
        self.last_action = self.sample_actions_from_dataset()

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    def reset(self):
        self.last_action = self.sample_actions_from_dataset()
        return self._env.reset()

    def step(self, action):
        margin = .05
        scaled_action = margin + (1.0 - 2*margin) * (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        action = norm.ppf(scaled_action).astype(np.float32)
        with torch.no_grad():
            cond_inputs = torch.tensor(self.last_action.copy(), device=self.device).unsqueeze(0).float()
            action = self.flow(torch.tensor(action, device=self.device).unsqueeze(0), cond_inputs=cond_inputs, mode='inverse')[0].squeeze().detach().cpu().numpy()
            action = np.clip(action, self._env.action_space.low, self._env.action_space.high)
            action = np.nan_to_num(action)
            s, r, d, i = self._env.step(action)
            self.last_action = action.copy()
        return s, r, d, i

    def sample_actions_from_dataset(self):
        idx = np.random.choice(len(self._env.get_primitive_dataset()['actions']))
        return self._env.get_primitive_dataset()['actions'][idx]
