import argparse
import operator
import subprocess
import os
import os.path as osp
import time
from enum import Enum
import json
import glob
import hashlib
import numpy as np
from scipy import signal
import torch

FLOW_PATH = osp.join(osp.abspath(osp.expanduser("~")), 'flows')
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.expanduser("~")), 'logs')
DATASET_DIR = osp.join(osp.abspath(osp.expanduser("~")), '.d4rl/datasets')


def str2bool(v):
    """
    Interprets a string as a boolean value. Credit to StackOverflow.
    Args:
        v (string): string representing a boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_logger_kwargs(exp_name, seed=None, debug=False, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs. Credits to SpinningUp.
    Args:
        exp_name (string): Name for experiment.
        seed (int): Seed for random number generators used by experiment.
        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.
        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.
    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    datestamp = datestamp
    ymd_time = time.strftime("_%Y-%m-%d_%H-%M-%S") if datestamp else ''
    relpath = ''.join([exp_name, ymd_time, f'_s{seed}'])
    data_dir = data_dir or DEFAULT_DATA_DIR
    if debug:
        data_dir = '/tmp'
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                         exp_name=exp_name)
    return logger_kwargs


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON.  Credits to SpinningUp. """
    try:
        json.dumps(obj)
        return obj
    except:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def set_seeds(seed):
    """ Sets library seeds.  Credits to SpinningUp. """
    torch.manual_seed(seed)
    np.random.seed(seed)


def discounted_sum(l, gamma):
    """
    Computes discounted sums according to the recurrence C[i] = R[i] + discount * C[i+1]. Credits to StackOverflow.
    Args:
        l : List of floats to sum.
        gamma : Discount factor.
    """
    l = l[::-1]
    a = [1, -gamma]
    b = [1]
    y = signal.lfilter(b, a, x=l)
    return y[::-1]


def get_hashes() -> str:
    """
    Gets hash of current commit. Courtesy of StackOverflow.
    https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script/21901260#21901260
    """
    hashes = dict(git=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
    filenames = glob.glob(osp.join(FLOW_PATH, '*.pt')) + glob.glob(osp.join(DATASET_DIR, '*.npy'))

    for filename in filenames:
        with open(filename, 'rb') as inputfile:
            data = inputfile.read()
            hashes[filename] = hashlib.md5(data).hexdigest()

    return hashes


class BCDataset(torch.utils.data.Dataset):

    def __init__(self, short_name, visual):

        self.short_name = short_name
        env = 'f{short_name}_primitive'
        self.label_key = 'states' if not visual else 'observations'
        self.visual = visual
        if visual:
            data_dir = osp.join(DATASET_DIR, f'{env}/')
            self.trajs = [osp.join(data_dir, p) for p in os.listdir(data_dir)]
        else:
            from envs import get_primitive_dataset
            x = get_primitive_dataset(short_name)
            done = np.logical_or(x['terminals'], x['timeouts'])  # last state of trajectory is marked as terminal
            self.trajs = {k: np.split(v, np.nonzero(done)[0] + 1) for k, v in x.items()}
            self.trajs = [{k: v[i] for k, v in self.trajs.items()} for i in range(len(self.trajs['actions']))]

        self.map_to_path_id = []  # maps index to path
        self.map_to_sample_id = []  # maps index to sample within trajectory
        self.len = 0
        for i, traj in enumerate(self.trajs):  # load each trajectory once to compute total length
            if visual:
                data = np.load(traj, allow_pickle=True).tolist()
            else:
                data = traj
            n_samples = len(data['actions'])
            self.map_to_path_id.extend([i] * n_samples)
            self.map_to_sample_id.extend(list(range(n_samples)))
            self.len += n_samples

        self.n_dims = data['actions'].shape[-1]

    def __getitem__(self, item):
        traj = self.trajs[self.map_to_path_id[item]]
        id = self.map_to_sample_id[item]
        if self.visual:
            traj = np.load(traj, allow_pickle=True).tolist() # load trajectory from disk
        data = traj['actions']
        # need to clip to a smaller value than 1 because behaviour cloning will maximize
        # the probability for atanh(action), which is out of support for action = +-1
        data = np.clip(data, -0.995, .995)
        labels = traj[self.label_key]
        if len(labels.shape) > 2:  # dealing with images, need to normalize
            return torch.tensor(data[id]).float(), torch.tensor(labels[id]).float() / 255.
        # use final state as goal
        goals = traj['goals']
        if any([s in self.short_name for s in ['swimmer', 'hopper', 'halfcheetah', 'ant', 'walker']]):
            return torch.tensor(data[id]).float(), torch.tensor(labels[id]).float()
        return torch.tensor(data[id]).float(), torch.tensor(np.concatenate([labels[id], goals[id]], -1)).float()

    def __len__(self):
        return self.len

# ##### MPI UTILS - Taken from SpinningUp, hacked to remove MPI dependency without breaking the logger.


def proc_id():
    """ Get rank of calling process. """
    return 0


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean) ** 2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std

# ##### DATATYPES


class SegmentTree(object):
    """
    A tree which can be used as a min/max heap or a sum tree
    Add or update item value - O(log N)
    Sampling an item - O(log N)

    Credits to https://github.com/IntelLabs/coach/blob/master/rl_coach/memories/non_episodic/prioritized_experience_replay.py
    """
    class Operation(Enum):
        MAX = {"operator": max, "initial_value": -float("inf")}
        MIN = {"operator": min, "initial_value": float("inf")}
        SUM = {"operator": operator.add, "initial_value": 0}

    def __init__(self, size, operation):
        self.size = size
        if not (size > 0 and size & (size - 1) == 0):
            raise ValueError("A segment tree size must be a positive power of 2. The given size is {}".format(self.size))
        self.operation = operation
        self.tree = np.ones(2 * size - 1) * self.operation.value['initial_value']
        self.data = [None] * size

    def _propagate(self, node_idx):
        """
        Propagate an update of a node's value to its parent node
        :param node_idx: the index of the node that was updated
        :return: None
        """
        parent = (node_idx - 1) // 2

        self.tree[parent] = self.operation.value['operator'](self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])

        if parent != 0:
            self._propagate(parent)

    def _retrieve(self, root_node_idx, val):
        """
        Retrieve the first node that has a value larger than val and is a child of the node at index idx
        :param root_node_idx: the index of the root node to search from
        :param val: the value to query for
        :return: the index of the resulting node
        """
        left = 2 * root_node_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return root_node_idx

        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val-self.tree[left])

    def total_value(self):
        """
        Return the total value of the tree according to the tree operation. For SUM for example, this will return
        the total sum of the tree. for MIN, this will return the minimal value
        :return: the total value of the tree
        """
        return self.tree[0]

    def add(self, next_leaf_idx_to_write, val, data):
        """
        Add a new value to the tree with data assigned to it
        :param val: the new value to add to the tree
        :param data: the data that should be assigned to this value
        :return: None
        """
        self.data[next_leaf_idx_to_write] = data
        self.update(next_leaf_idx_to_write, val)

    def update(self, leaf_idx, new_val):
        """
        Update the value of the node at index idx
        :param leaf_idx: the index of the node to update
        :param new_val: the new value of the node
        :return: None
        """
        node_idx = leaf_idx + self.size - 1
        if not 0 <= node_idx < len(self.tree):
            raise ValueError("The given left index ({}) can not be found in the tree. The available leaves are: 0-{}"
                             .format(leaf_idx, self.size - 1))

        self.tree[node_idx] = new_val
        self._propagate(node_idx)

    def get_element_by_partial_sum(self, val):
        """
        Given a value between 0 and the tree sum, return the object which this value is in it's range.
        For example, if we have 3 leaves: 10, 20, 30, and val=35, this will return the 3rd leaf, by accumulating
        leaves by their order until getting to 35. This allows sampling leaves according to their proportional
        probability.
        :param val: a value within the range 0 and the tree sum
        :return: the index of the resulting leaf in the tree, its probability and
                 the object itself
        """
        node_idx = self._retrieve(0, val)
        leaf_idx = node_idx - self.size + 1
        data_value = self.tree[node_idx]
        data = self.data[leaf_idx]

        return leaf_idx, data_value, data

    def __str__(self):
        result = ""
        start = 0
        size = 1
        while size <= self.size:
            result += "{}\n".format(self.tree[start:(start + size)])
            start += size
            size *= 2
        return result
