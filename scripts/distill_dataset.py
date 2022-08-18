"""
Scripts that takes data from a folder produced by collect_large_dataset.py and
distills it in a single file by discarding images.
"""

import os
import argparse
import os.path as osp
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--env', default='maze_primitive', type=str, help='environment for trajectory extraction (one of point-maze and meta-world)')
    args = parser.parse_args()

    if args.env == 'meta-world':
        name = 'sawyer_primitive' 
    elif args.env == 'point-maze':
        name = 'maze_primitive'
    else:
        raise Exception('Unknown environment name.')

    data_dir = osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{name}/')
    paths = [osp.join(data_dir, p) for p in os.listdir(data_dir)]
    all_trajs = {'actions': [], 'states': [], 'goals': [], 'terminals': [], 'timeouts': []}
    for path in sorted(paths):
        print(path)
        data = np.load(path, allow_pickle=True).item()
        print([data[k].shape for k, _ in all_trajs.items()])
        all_trajs = {k: v + [data[k]] for k, v in all_trajs.items()}
    all_trajs = {k: np.concatenate(v, 0) for k, v in all_trajs.items()}
    print('Final action size:', all_trajs['actions'].shape)
    np.save(osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{name}_distilled.npy'), all_trajs)