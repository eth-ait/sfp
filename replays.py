"""
Replay buffer implementation.
"""

import numpy as np
import torch
import gym
from utils import SegmentTree, discounted_sum

class HERReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents. Saves entire trajectories.
    Supports goal-conditioned RL, Hindsight Experience Replay, Prioritized Experience Replay and n-step reward computation.
    Args:
        obs_space : Observation space of the environment.
        act_dim (int): Size of the action space of the environment.
        size (int): Number of transitions to store.
        T (int): Length of every trajectory.
        her (bool): Flag to enable Hindsight Experience Replay.
        replay_k (int): Replay ratio for HER.
        reward_fun : Reward function for HER relabeling.
        prioritize (bool): Flag to enable Prioritized Experience Replay.
        alpha (float): Alpha parameter for PER.
        beta (float): Beta parameter for PER.
        epsilon (float): Epsilon parameter for PER.
        gamma (float): Discount factor.
        n_step (int): Number of steps between state pairs and for reward computation.
    """

    def __init__(self, obs_space, act_dim, size, T, her, replay_k, reward_fun, prioritize=False,
                 alpha=0.6, beta=0.4, epsilon=1e-5, gamma=0.99, n_step=1, clip_rew=True, prior_n_step=1):
        self.T = T
        self.max_size = size // T
        self.replay_k = replay_k
        self.reward_fun = reward_fun
        self.her = her
        self.clip_rew = clip_rew
        if isinstance(obs_space, gym.spaces.dict.Dict):  # Goal-conditioned framework
            self.goal_cond = True
            obs_dim = obs_space['observation'].shape[0]
            goal_dim = obs_space['desired_goal'].shape[0]
            self.obs_buf = np.zeros((self.max_size, T, obs_dim), dtype=np.float32)
            self.obs2_buf = np.zeros((self.max_size, T, obs_dim), dtype=np.float32)
            self.dg_buf = np.zeros((self.max_size, T, goal_dim), dtype=np.float32)
            self.dg2_buf = np.zeros((self.max_size, T, goal_dim), dtype=np.float32)
            self.ag_buf = np.zeros((self.max_size, T, goal_dim), dtype=np.float32)
            self.ag2_buf = np.zeros((self.max_size, T, goal_dim), dtype=np.float32)
        else:
            self.goal_cond = False
            obs_dim = obs_space.shape
            self.obs_buf = np.zeros((self.max_size, T, *obs_dim), dtype=np.float32)
            self.obs2_buf = np.zeros((self.max_size, T, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_size, T, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.max_size, T), dtype=np.float32)
        self.ret_buf = np.zeros((self.max_size, T), dtype=np.float32)
        self.done_buf = np.zeros((self.max_size, T), dtype=np.float32)
        self.last_act_buf = np.zeros((self.max_size, T, act_dim*prior_n_step), dtype=np.float32)
        # ptr points to the next free slot, size tracks the current size of the buffer
        # max_size represents thecapacity ofthe buffer
        self.ptr, self.size, self.max_size = 0, 0, size

        self.prioritize = prioritize
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.priority_buf = np.zeros((self.max_size, T), dtype=np.float32)
        self.n_step = n_step

        if self.prioritize:  # Setup data structures
            self.power_of_2_size = 1
            while self.power_of_2_size < size:
                self.power_of_2_size *= 2
            self.sum_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.SUM)
            self.min_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.MIN)
            self.max_tree = SegmentTree(self.power_of_2_size, SegmentTree.Operation.MAX)
            self.maximal_priority = 1.0

    def store_episode(self, trajectory):
        """ Stores a single trajectory in buffer. """
        obs = trajectory['obs'][:-1]
        obs2 = trajectory['obs'][1:]
        act = trajectory['act']
        rew = trajectory['rew']
        last_act = trajectory['last_act']
        ret = discounted_sum(rew, self.gamma)  # Compute complessive return of trajectory
        done = trajectory['done']
        for o, o2, a, r, r2, d, la in zip(obs, obs2, act, rew, ret, done, last_act):
            if self.goal_cond:
                self.obs_buf[self.ptr // self.T, self.ptr % self.T] = o['observation']
                self.obs2_buf[self.ptr // self.T, self.ptr % self.T] = o2['observation']
                self.ag_buf[self.ptr // self.T, self.ptr % self.T] = o['achieved_goal']
                self.ag2_buf[self.ptr // self.T, self.ptr % self.T] = o2['achieved_goal']
                self.dg_buf[self.ptr // self.T, self.ptr % self.T] = o['desired_goal']
                self.dg2_buf[self.ptr // self.T, self.ptr % self.T] = o2['desired_goal']
            else:
                self.obs_buf[self.ptr // self.T, self.ptr % self.T] = o
                self.obs2_buf[self.ptr // self.T, self.ptr % self.T] = o2
            self.act_buf[self.ptr // self.T, self.ptr % self.T] = a
            self.rew_buf[self.ptr // self.T, self.ptr % self.T] = r
            self.ret_buf[self.ptr // self.T, self.ptr % self.T] = r2
            self.done_buf[self.ptr // self.T, self.ptr % self.T] = d
            self.last_act_buf[self.ptr // self.T, self.ptr % self.T] = la
            if self.prioritize:
                self.sum_tree.add(self.ptr, self.maximal_priority ** self.alpha, self.ptr)
                self.min_tree.add(self.ptr, self.maximal_priority ** self.alpha, self.ptr)
                self.max_tree.add(self.ptr, self.maximal_priority, self.ptr)
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, progress, force_uniform=False):
        """
            Samples a single batch from replay buffer.
        Args:
            batch_size (int): Number of transitions to collect.
            progress (float): Ratio between the currentand total number of steps.
            force_uniform (bool): Flag to forcefully disable PER.
        """
        if self.prioritize and not force_uniform:  # Sample according to priority and compute importance weights
            idxs, weights = [], []
            min_probability = self.min_tree.total_value() / self.sum_tree.total_value()  # min P(j) = min p^a / sum(p^a)
            max_weight = (min_probability * self.size) ** (-self.beta*(1-progress))  # max wi
            for _ in range(batch_size):
                _, priority, transition_id = self.sum_tree.get_element_by_partial_sum(np.random.uniform(0, self.sum_tree.total_value()))
                idxs.append(transition_id)
                weights.append(priority)
            idxs, weights = np.array(idxs), np.array(weights)
            weights /= self.sum_tree.total_value()   # P(j) = p^a / sum(p^a)
            weights = (self.size * weights) ** (-self.beta*(1-progress))  # (N * P(j)) ^ -beta
            weights = weights / max_weight  # wj = ((N * P(j)) ^ -beta) / max wi
        else:  # Sample uniformly
            idxs = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones_like(idxs)

        batch = dict(obs=self.obs_buf[idxs // self.T, idxs % self.T],
                     obs2=self.obs2_buf[idxs // self.T, idxs % self.T],
                     act=self.act_buf[idxs // self.T, idxs % self.T],
                     rew=self.rew_buf[idxs // self.T, idxs % self.T],
                     ret=self.ret_buf[idxs // self.T, idxs % self.T],
                     done=self.done_buf[idxs // self.T, idxs % self.T],
                     last_act=self.last_act_buf[idxs // self.T, idxs % self.T],
                     weights=weights)
        
        if self.n_step > 1:  # Compute discounted n-step rewards
            # Indices of first transition to not consider for reward computation
            last_rew_idxs = np.minimum((idxs % self.T)+self.n_step, self.T)
            # Number of steps for each sample in the batch
            # Can be lower than self.n_step in case idxs were sampled close to the end of a trajectory
            actual_n_step = last_rew_idxs - (idxs % self.T)
            rew = np.zeros_like(idxs)
            for i, (a, b) in enumerate(zip(idxs, last_rew_idxs)):
                rew[i] = discounted_sum(self.rew_buf[a // self.T, (a % self.T):b], self.gamma)[0]
            batch.update(dict(rew=rew, n_step=actual_n_step, obs2=self.obs2_buf[idxs // self.T, last_rew_idxs - 1]))
        
        if not self.goal_cond:  # Return simple observations
            if self.clip_rew:
                batch['rew'] = np.clip(batch['rew'], 0, 1)
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}, idxs

        batch['ag'] = self.ag_buf[idxs // self.T, idxs % self.T]
        batch['ag2'] = self.ag2_buf[idxs // self.T, idxs % self.T]
        batch['dg'] = self.dg_buf[idxs // self.T, idxs % self.T]
        batch['dg2'] = self.dg2_buf[idxs // self.T, idxs % self.T]

        if self.n_step > 1:
            batch.update(dict(ag2=self.ag2_buf[idxs // self.T, last_rew_idxs - 1], dg2=self.dg2_buf[idxs // self.T, last_rew_idxs - 1]))

        if self.her:
            # Probability of relabeling
            future_p = 1 - (1. / (1 + self.replay_k))

            # Indexes of samples that will be relabeled
            her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)[0]
            # Uniformly sample desired goal
            future_offset = np.random.uniform(size=batch_size) * (self.T - (idxs % self.T))
            future_offset = future_offset.astype(int)
            future_t = ((idxs % self.T) + future_offset)[her_indexes]
            assert len(future_t) == len(her_indexes), (len(future_t), len(her_indexes))

            future_ag = self.ag2_buf[(idxs // self.T)[her_indexes], future_t]
            batch['dg'][her_indexes] = future_ag
            batch['rew'] = self.reward_fun(batch['ag2'], batch['dg'], {})

            if self.n_step > 1:  # Need to recompute n-step rewards for relabeled samples
                her_flags = np.zeros(batch_size)
                her_flags[her_indexes] = 1
                for i, (_dg, a, b) in enumerate(zip(batch['dg'], idxs, last_rew_idxs)):
                    if her_flags[i]:
                        recomputed_rew = self.reward_fun(self.ag2_buf[a // self.T, (a % self.T):b], np.stack([_dg]*(b-(a % self.T)), 0), {})
                        batch['rew'][i] = discounted_sum(recomputed_rew, self.gamma)[0]

            if self.clip_rew:
                batch['rew'] = np.clip(batch['rew'], 0, 1)

        # Reconstruct obs dict
        batch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
        batch['obs'] = {'observation': batch.pop('obs'), 'achieved_goal': batch.pop('ag'), 'desired_goal': batch.pop('dg')}
        batch['obs2'] = {'observation': batch.pop('obs2'), 'achieved_goal': batch.pop('ag2'), 'desired_goal': batch.pop('dg2')}
        return batch, idxs

    def update_priorities(self, p, idxs):
        """
        Update priorities for PER or SIL.
        Args:
            p : Priorities.
            idxs : Indices of transitions that need tobe updated.
        """
        if self.prioritize:
            if np.any(p) < 0:
                raise ValueError("The priorities must be non-negative values")
            for _p, _idx in zip(p, idxs):
                _priority = _p + self.epsilon
                self.sum_tree.update(_idx, _priority ** self.alpha)
                self.min_tree.update(_idx, _priority ** self.alpha)
                self.max_tree.update(_idx, _priority)
                self.maximal_priority = self.max_tree.total_value()
