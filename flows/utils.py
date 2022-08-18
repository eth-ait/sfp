"""
General utilities and plotting.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp

# TODO: remove these duplicates
FLOW_PATH = osp.join(osp.abspath(osp.expanduser("~")), 'flows')
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.expanduser("~")), 'logs')
DATASET_DIR = osp.join(osp.abspath(osp.expanduser("~")), '.d4rl/datasets')

def save_plot(epoch, best_model, dataset, exp_name):
    '''
    Queries best model to generate plots. Only useful for non-conditional flows.
    '''

    # generate some examples
    best_model.eval()
    with torch.no_grad():
        x_synth = best_model.sample(500).detach().cpu().numpy()

    fig = plt.figure()
 
    # plot side by side
    ax = fig.add_subplot(121)
    data = np.stack([dataset[i] for i in np.random.choice(np.arange(len(dataset)), 100)], 0)
    ax.plot(data[:, 0], data[:, 1], '.')
    ax.set_title('Real data')
    ax = fig.add_subplot(122)
    ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
    ax.set_title('Synth data')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    plt.savefig(exp_name + '/plot_{:03d}.png'.format(epoch))
    plt.close()


def str2bool(v):
    '''
    Converts various strings to booleans.
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
