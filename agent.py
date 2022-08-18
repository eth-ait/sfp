"""
Implementation of SAC, adapted from Spinning Up.
"""

from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from modules import MLPActorCritic
from prior import TemporalPrior
from utils import count_vars
from polyrl import PolyRL

class Agent:
    """
    Abstract class representing an RL agent.
    """

    def __init__(self):
        pass

    def act(self, o, deterministic=False, last_action=None):
        """
        Returns an action when prompted with an observation.
        """
        raise NotImplementedError()

    def update(self, data):
        """
        Training step.
        """
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()


class SAC(Agent):
    """
    Soft Actor-Critic (SAC) adapted from SpinningUp.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        logger : A logger instance.
        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target
            networks.
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)
        prior_kwargs (dict): Parameters for computing mixing weight.
        clip_gradients (bool): Flag that enables gradient clipping for Q-networks.
        gpu (bool): Flag that enables training on cuda devices.
    """

    def __init__(self, env_fn, logger, actor_critic=MLPActorCritic, ac_kwargs=dict(), gamma=0.99, polyak=0.995,
                 lr=1e-3, alpha=0.2, beta=0.2, prior_kwargs={}, clip_gradients=False, use_polyrl=False, gpu=False,
                 epsilon=1.0):
        super().__init__()
        self.logger = logger
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.device = 'cuda' if gpu else 'cpu'
        self.epsilon = epsilon
        self.kl_reg = prior_kwargs['kl_reg']

        # Create actor-critic module and target networks
        temp_env = env_fn()

        self.ac = actor_critic(temp_env.observation_space, temp_env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        # List of parameters for both Q-networks
        self.q_params = set(list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()))
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.lambda_optimizer = Adam(self.ac.lambda_params, lr=self.lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        self.use_prior = prior_kwargs['use_prior']
        if self.use_prior or self.kl_reg:
            self.temporal_prior = TemporalPrior(env=temp_env, device=self.device, **prior_kwargs)
        self.clip_gradients = clip_gradients
        self.weight_schedule = 0.
        self.initial_prior = prior_kwargs['initial']

        # Polyrl stuff
        self.use_polyrl = use_polyrl
        if self.use_polyrl:
            self.select_action_target = lambda o: self.ac.act(torch.as_tensor(
                np.concatenate([o['observation'], o['desired_goal']], -1) if isinstance(o, dict) else o,
                dtype=torch.float32, device=self.device).unsqueeze(0), deterministic=False)[0][0]
            self.polyrl = PolyRL(gamma=self.gamma, env=temp_env, actor_target_function=self.select_action_target)        

    def act(self, o, deterministic=False, last_action=None):
        """
        Produces an action as a response to an observation.
        Args:
            o : Current observation.
            deterministic (bool): Flag that forces the actor to return the mean of the policy
                instead of sampling from it.
            last_actions : Last action taken in environment.
        """
        if isinstance(o, dict):
            o = np.concatenate([o['observation'], o['desired_goal']], -1)
        o = torch.as_tensor(o.copy(), dtype=torch.float32, device=self.device).unsqueeze(0)
        last_action = torch.tensor(last_action, dtype=torch.float32, device=self.device).reshape(1, -1)
        action, logp_pi, _ = self.ac.act(o, deterministic)

        if deterministic:
            pass
        elif self.use_prior:
            with torch.no_grad():
                lambda_ = self.ac.lambda_net(o).cpu().detach().numpy() * self.weight_schedule
                sample_action = np.random.uniform() < lambda_
                self.logger.store(MixingWeight=lambda_)
                if sample_action:
                    prior_action = self.temporal_prior.sample(o, last_action)
                    prior_action = torch.clamp(prior_action, -1*self.ac.pi.act_limit, 1*self.ac.pi.act_limit)
                    action = prior_action
                    self.logger.store(PriorP=1.)
                else:
                    self.logger.store(PriorP=0.)
        else:
            self.logger.store(MixingWeight=0.)
            self.logger.store(PriorP=0.)
        return action.cpu().numpy()[0]

    def initial_explore(self, env, o, last_action, step_number):
        """
        Produces an explorative action during an initial phase of training. Samples from a prior, if available, else uniformly.
        """
        if isinstance(o, dict):
            o = np.concatenate([o['observation'], o['desired_goal']], -1)
        last_action = torch.tensor(last_action, dtype=torch.float32, device=self.device).reshape(1, -1)
        if self.use_polyrl:
            self.logger.store(PriorP=0.)
            self.logger.store(MixingWeight=0.)
            action = self.polyrl.select_action(o, last_action[0], step_number=step_number)
            action = torch.clamp(action, -1, 1).reshape(-1).cpu().numpy()
            return action
        if self.use_prior and self.initial_prior:
            self.logger.store(PriorP=1.)
            self.logger.store(MixingWeight=1.)
            o = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.temporal_prior.sample(o, last_action)
            action = torch.clamp(action, -1, 1).reshape(-1).cpu().numpy()
            return action
        self.logger.store(PriorP=0.)
        self.logger.store(MixingWeight=0.)
        return env.action_space.sample()

    def compute_loss_q(self, data):
        """
        Computes Q-loss.
        
        Args:
            data : Batch of experiences from replay buffer.
        """
        o, a, r, o2, d, w, la = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['weights'], data['last_act']

        if isinstance(o, dict):
            o = torch.cat([o['observation'], o['desired_goal']], -1)

        if isinstance(o2, dict):
            o2 = torch.cat([o2['observation'], o2['desired_goal']], -1)

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _, mu, std = self.ac.pi(o2, with_distr=True)

            if self.use_prior:
                # Target actions come from *current* policy
                lambda_ = self.ac.lambda_net(o2)
                sample_prior = torch.rand(size=lambda_.shape, device=lambda_.device) < lambda_
                a2_prior = self.temporal_prior.sample(o2, last_action=la)
                a2 = torch.where(sample_prior, a2_prior, a2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            n_step = data['n_step'] if 'n_step' in data.keys() else 1
            if self.kl_reg:
                mu_prior, log_std_prior = self.temporal_prior.sample(o2, last_action=la)
                std_prior = torch.exp(log_std_prior)
                kl_div = 0.5*(torch.log(std_prior.prod(-1)/std.prod(-1)) + (std/std_prior).sum(-1) + ((mu_prior - mu)**2 / std_prior).sum(-1))
                backup = r + (self.gamma ** n_step) * (1 - d) * (q_pi_targ - self.alpha * kl_div)
            else:
                backup = r + (self.gamma ** n_step) * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = (w*((q1 - backup) ** 2)).mean()
        loss_q2 = (w*((q2 - backup) ** 2)).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy(),
                      Weights=w.detach().cpu().numpy())
        
        # Priorities for PER
        priorities = ((torch.abs(q1 - backup) + torch.abs(q2 - backup))/2.0).abs().detach().cpu().numpy()

        return loss_q, q_info, priorities

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        """
        Computes policy loss.
        
        Args:
            data : Batch of experiences from replay buffer.
        """
        o, w, la = data['obs'], data['weights'], data['last_act']

        if isinstance(o, dict):
            o = torch.cat([o['observation'], o['desired_goal']], -1)

        pi, logp_pi, logstd, mu, std = self.ac.pi(o, with_distr=True)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        with torch.no_grad():
            if self.use_prior:
                lambda_ = self.ac.lambda_net(o).nan_to_num()
            else:
                lambda_ = torch.zeros_like(q_pi)

        # Entropy-regularized policy loss
        if self.kl_reg:
            mu_prior, log_std_prior = self.temporal_prior.sample(o, last_action=la)
            std_prior = torch.exp(log_std_prior)
            kl_div = 0.5*(torch.log(std_prior.prod(-1)/std.prod(-1)) + (std/std_prior).sum(-1) + ((mu_prior - mu)**2 / std_prior).sum(-1))
            loss_pi = (w*(self.alpha * kl_div - (1-lambda_)*q_pi)).mean()
        else:
            loss_pi = (w*(self.alpha * logp_pi - (1-lambda_)*q_pi)).mean()

        # Useful info for logging
        h = logstd.sum(-1) + (1+np.log(2*np.pi)) * 0.5 * logstd.shape[-1]
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy(), LogStd=logstd.detach().cpu().numpy(), H=h.detach().cpu().numpy())

        return loss_pi, pi_info
    
    def compute_loss_lambda(self, data):
        la, o2 = data['last_act'], data['obs2']
        if isinstance(o2, dict):
            o2 = torch.cat([o2['observation'], o2['desired_goal']], -1)

        with torch.no_grad():
            pi, logp_pi_policy, _ = self.ac.pi(o2)
            pi_bar = self.temporal_prior.sample(o2, last_action=la)

            q1_pi = self.ac.q1(o2, pi)
            q2_pi = self.ac.q2(o2, pi)
            q_pi = torch.min(q1_pi, q2_pi)

            q1_pi_bar = self.ac.q1(o2, pi_bar)
            q2_pi_bar = self.ac.q2(o2, pi_bar)
            q_pi_bar = torch.min(q1_pi_bar, q2_pi_bar)
        
        # nan_to_num prevent numerics issue when $\lambda \approx 0$
        lambda_ = self.ac.lambda_net(o2).nan_to_num()
        loss_lambda = (self.epsilon * (-lambda_ * (self.beta + q_pi_bar - q_pi))).mean()
        return loss_lambda

    def update(self, data):
        """
        Update networks by computing and backpropagating losses.
        
        Args:
            data : Batch of experiences from replay buffer.
        """
        # First run one gradient descent step for Q1 and Q2

        data = {k: v.to(self.device) if not isinstance(v, dict) else {k2: v2.to(self.device) for k2, v2 in v.items()} for k, v in data.items()}

        self.q_optimizer.zero_grad()
        loss_q, q_info, priorities = self.compute_loss_q(data)
        loss_q.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.q_params, 1)
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), Priorities=priorities, **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.use_prior:
            self.lambda_optimizer.zero_grad()
            loss_lambda = self.compute_loss_lambda(data)
            loss_lambda.backward()
            self.lambda_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        return priorities


    def imitate(self, data, sil_weight, value_weight):
        """
        Update networks through Self Imitation Learning.
        
        Args:
            data : Batch of experiences from replay buffer.
            sil_weight (float): Weight of SIL losses wrt SAC losses.
            value_weight (float): Weight of SIL value loss wrt SIL policy loss.
        """
        o, a, r, w = data['obs'], data['act'], data['ret'], data['weights']

        if isinstance(o, dict):  # Goal-conditioned framework
            o = torch.cat([o['observation'], o['desired_goal']], -1)

        # SIL value loss
        self.q_optimizer.zero_grad()
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)
        q1_loss = (0.5 * w * (torch.maximum(torch.zeros_like(r), r - q1) ** 2)).mean()
        q2_loss = (0.5 * w * (torch.maximum(torch.zeros_like(r), r - q2) ** 2)).mean()
        loss_q = (q1_loss + q2_loss) * sil_weight * value_weight
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze parameters to save computational effort
        for p in self.q_params:
            p.requires_grad = False

        # SIL policy loss
        self.pi_optimizer.zero_grad()
        q1d = q1.detach()
        q2d = q2.detach()
        error = r - torch.min(q1d, q2d)
        priorities = torch.maximum(torch.zeros_like(r), error)
        logprob = self.ac.pi(o, action=a)
        logprob = torch.clip(logprob, -20, 20)
        loss_pi = (- w * logprob * priorities).mean() * sil_weight
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossQSIL=loss_q.item(), LossPiSIL=loss_pi.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return priorities.detach().cpu().numpy()

    def clone(self, data):
        """ Performs behavior cloning with policy network. """
        a, o = data
        self.pi_optimizer.zero_grad()
        loss_pi = -self.ac.pi(o, action=a).mean()
        loss_pi.backward()
        self.pi_optimizer.step()

    def pre_ep(self, n):
        if self.use_polyrl:
            self.polyrl.reset_parameters_in_beginning_of_episode(n)

    def post_step(self, o, o2):
        if self.use_polyrl:
            if isinstance(o, dict):
                o = np.concatenate([o['observation'], o['desired_goal']], -1)
            if isinstance(o2, dict):
                o2 = np.concatenate([o2['observation'], o2['desired_goal']], -1)
            self.polyrl.update_parameters(o, o2)
        
    def save(self):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()
