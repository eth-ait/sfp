"""
Implementation of datasets for training conditional models.
"""
import numpy as np
import torch
from envs import get_primitive_dataset
import os.path as osp
import os


class Dataset(torch.utils.data.Dataset):
    '''
    General dataset class. When dealing with images, it reads trajectories individually from disks. When
    dealing with vectorized states or actions, it loads all trajectories directly into memory.

    Args:
        env (string): Name of the dataset to load.
        mode (string): What information fo condition upon. One of 'action', 'state', 'action+state', 'image'.
        train (bool): Whether to load training or validation samples.
        one_step (bool): Flag to enable learning non-conditional single action probability density.
        n_step (int): Number of past steps to condition upon.
        debug (bool): Flag that reduces the amount of data for faster debugging.
    '''

    def __init__(self, env, mode, train, one_step, n_step, data_ratio=1.0, debug=False):
        self.mode = mode
        self.one_step = one_step
        self.n_step = n_step
        self.label_keys = {'action': ['actions'], 'state': ['states'], 'image': ['observations'], 
                           'action+state': ['actions', 'states', 'goals'], 'state+goal': ['states', 'goals']}[mode]
        env = 'sawyer' if env == 'meta-world' else env
        env = 'maze' if env == 'point-maze' else env        

        assert not (self.one_step and n_step!=1)
        assert not ('observations' in self.label_keys and (n_step > 1 or len(self.label_keys) > 1)), 'If working with images you cannot condition on else.'
        # assert env in ['sawyer_primitive', 'maze_primitive', 'swimmer_primitive'], f'Dataset {env} is not supported.'

        if mode == 'image':
            data_dir = osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{env}/')
            self.trajs = [np.load(osp.join(data_dir, p), allow_pickle=True).tolist() for p in os.listdir(data_dir)]
        else:
            x = get_primitive_dataset(env)
            # multi-step datasets need to split the dataset into trajectories to only recover meaningful pairs
            done = np.logical_or(x['terminals'], x['timeouts'])  # last state of trajectory is marked as terminal
            self.trajs = {k: np.split(v, np.nonzero(done)[0] + 1) for k, v in x.items()}
            self.trajs = [{k: v[i] for k, v in self.trajs.items()} for i in range(len(self.trajs['actions']))][:-1]
        if data_ratio < 1.0:
            original_len = len(self.trajs)
            self.trajs = self.trajs[:int(len(self.trajs)*data_ratio)]
            print(f'Training on {len(self.trajs)} trajectories.')
            n_repeat = int(1.0 / data_ratio) + 1
            self.trajs = [item for _ in range(n_repeat) for item in self.trajs ][:original_len]
        if debug:  # use a fraction of trajectories for training
            self.trajs = self.trajs[:10]
        # training-validation split
        self.trajs = self.trajs[:int(0.9*len(self.trajs))] if train else self.trajs[int(0.9*len(self.trajs)):]

        self.map_to_path_id = []  # maps index to path
        self.map_to_sample_id = []  # maps index to sample within trajectory
        self.len = 0
        for i, traj in enumerate(self.trajs):  # load each trajectory once to compute total length
            data = traj
            if self.one_step:
                n_samples = len(data['actions'])
            elif mode in ['image', 'state', 'state+goal']:
                n_samples = len(data['actions'])
            elif 'action' in mode:
                n_samples = (len(data['actions']) - 1)
            else:
                raise NotImplementedError

            n_samples = n_samples - (self.n_step - 1)
            self.map_to_path_id.extend([i] * n_samples)
            self.map_to_sample_id.extend(np.arange(n_samples) + self.n_step - 1)
            self.len += n_samples

        self.n_dims = data['actions'].shape[-1]
        self.n_cond_dims = sum([data[k][0].shape[-1] for k in self.label_keys]) * n_step

    def __getitem__(self, item):
        """ Retrieves a single sample. """
        traj = self.trajs[self.map_to_path_id[item]]
        id = self.map_to_sample_id[item]
        data = traj['actions']
        if self.one_step:
            return torch.tensor(data[id]).float()
        if 'actions' in self.label_keys:
            labels = [traj[k][:-1] if k=='actions' else traj[k][1:] for k in self.label_keys]
            data = data[1:]
        else:
            labels = [traj[k] for k in self.label_keys]
        labels = [l[(id - self.n_step + 1):id+1] for l in labels]  # select n steps
        labels = np.concatenate(labels, -1) # concatenate into n_steps x m
        data, labels = torch.tensor(data[id]).float(), torch.tensor(labels).float()
        if 'observations' in self.label_keys:  # dealing with images, need to normalize
            return data, labels[0] / 255.
        return data, labels.reshape(-1)

    def __len__(self):
        """ Number of samples in the dataset. """
        return self.len

    def get_bn_batch(self, size):
        """ Returns a random batch for visual settings, else the whole dataset. """
        idxs = np.random.choice(self.len, size=size) if self.mode == 'image' else np.arange(self.len)
        if self.one_step:
            return torch.stack([self.__getitem__(i) for i in idxs], 0).float()
        data, labels = zip(*[self.__getitem__(i) for i in idxs])
        return torch.stack(data, 0), torch.stack(labels, 0)


def get_datasets(args):
    """ Creates datasets according to provided parameters. """
    return (Dataset(args.env, args.mode, train=True, one_step=args.one_step, n_step=args.n_step, data_ratio=args.data_ratio, debug=args.debug), 
            Dataset(args.env, args.mode, train=False, one_step=args.one_step, n_step=args.n_step, debug=args.debug))
