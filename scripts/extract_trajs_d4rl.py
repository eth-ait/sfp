"""
Script for collecting a large amount of task-agnostic expert trajectories from d4rl datasets.
"""

from copy import deepcopy
from torch.optim import Adam
import gym
import numpy as np
import torch.nn as nn
import os.path as osp
import numpy as np
import d4rl

def save_trajectory(name, seed=42, data=None):
    """ Saves a single trajectory"""

    terminals = np.zeros(500)
    terminals[-1] = 1
    buffer = {'actions': data['actions'][seed*1000:(seed*1000+500)],
                'states': data['observations'][seed*1000:(seed*1000+500)],
                'goals': data['next_observations'][seed*1000:(seed*1000+500)],
                'terminals': terminals,
                'timeouts': np.zeros_like(terminals)}
    np.save(osp.join(data_dir, f'{name}_{seed}.npy'), buffer)


if __name__ == '__main__':
    num_trajs = 1000
    env_name = 'hopper-expert-v2'
    name = 'hopper-primitive'
    env_fn = lambda : gym.make(env_name)

    import os
    import os.path as osp
    data_dir = osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{name}/')
    env = env_fn()
    data = env.get_dataset()
    while True:
        data_dir = osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{name}/')
        existing_trajs = [int(''.join([c if c.isdigit() else '' for c in name])) for name in os.listdir(data_dir)]
        trajs_to_save = [i for i in np.arange(num_trajs) if i not in existing_trajs]
        if not trajs_to_save:
            exit(0)
        seed = np.random.choice(trajs_to_save)
        print(f'Seed: {seed}')
        save_trajectory(name=name, seed=seed, data=data)
