'''
Main script for training NVP-Flow models. Most of the codebase was adapted from Ilya Kostrikov's implementation at github.com/ikostrikov/pytorch-flows
'''

import argparse
from datetime import datetime
import copy
import os.path as osp
import json
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from utils import DEFAULT_DATA_DIR, FLOW_PATH
from datasets import get_datasets
import flows as fnn


def train(model, optimizer, train_loader, device, deterministic, gaussian):
    """ Trains model for a single epoch. """
    model.train()
    train_loss = 0

    for data in train_loader:
        if isinstance(data, list):  # conditional flow
            x = data[0].to(device)
            y = data[1].to(device)
        else:  # non-conditional flow
            x = data.to(device)
            y = None

        optimizer.zero_grad()
        if deterministic:
            loss = 0.5 * ((model(y) - x)**2).mean()
        elif gaussian:
            mu, logstd = model(y)
            std = torch.exp(logstd)
            pi_distribution = Normal(mu, std)
            y = y.clip(-0.9999, 0.9999)
            policy_output = torch.atanh(y)
            loss = -pi_distribution.log_prob(policy_output).sum(axis=-1).mean()
        else:
            loss = -model.log_probs(x, y).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    compute_bn_statistics = False  # only compute bn statistics if a bn layer is present
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0
            compute_bn_statistics = True

    if compute_bn_statistics:  # update bn mean and variance estimates
        with torch.no_grad():
            if isinstance(data, list):
                data, labels = train_loader.dataset.get_bn_batch(x.shape[0])
                model(data.to(device), labels.to(device))
            else:
                model(train_loader.dataset.get_bn_batch(x.shape[0]).to(device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1
    
    return train_loss / len(train_loader)


def validate(model, val_loader, device, deterministic, gaussian):
    ''' Standard validation routine. '''
    model.eval()
    val_loss = 0

    for data in val_loader:
        if isinstance(data, list):  # conditional flow
            x = data[0].to(device)
            y = data[1].to(device)
        else:
            x = data.to(device)
            y = None
        with torch.no_grad():
            if deterministic:
                val_loss = 0.5 * ((model(y) - x)**2).mean().item()
            elif gaussian:
                mu, logstd = model(y)
                std = torch.exp(logstd)
                pi_distribution = Normal(mu, std)
                y = y.clip(-0.9999, 0.9999)
                policy_output = torch.atanh(y)
                val_loss = -pi_distribution.log_prob(policy_output).sum(axis=-1).mean().item()
            else:
                val_loss = -model.log_probs(x, y).mean().item()

    return val_loss / len(val_loader)


def main(args):

    # torch setup
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    # initialize dataloaders
    train_dataset, val_dataset = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device == 'cuda'))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, pin_memory=(device == 'cuda'))

    # build model
    num_inputs = train_dataset.n_dims
    num_hidden = args.num_hidden
    if args.deterministic:  # deterministic mapping from conditional variables to outputs
        model = fnn.Deterministic(num_inputs=train_dataset.n_cond_dims,
                                  num_hidden=num_hidden,
                                  num_outputs=num_inputs)
    elif args.gaussian:
        model = fnn.Gaussian(num_inputs=train_dataset.n_cond_dims,
                                  num_hidden=num_hidden,
                                  num_outputs=num_inputs)    
    else:  # probabilistic flow
        modules = []
        mask = (torch.arange(0, num_inputs) % 2).to(device).float()
        for _ in range(args.num_blocks):
            if args.mode in ['action', 'state', 'action+state', 'state+goal']:
                modules += [fnn.CouplingLayer(num_inputs, num_hidden, mask, num_cond_inputs=None if args.one_step else train_dataset.n_cond_dims,
                            s_act=args.s_act, t_act=args.t_act, shared=args.shared, pre_mlp = args.pre_mlp, pre_mlp_units = args.pre_mlp_units)]
            elif args.mode == 'image':
                modules += [fnn.CNNCouplingLayer(num_inputs, num_hidden, mask, num_cond_inputs=None if args.one_step else args.pre_mlp_units,
                            s_act=args.s_act, t_act=args.t_act, shared=args.shared, pre_mlp = args.pre_mlp, pre_mlp_units = args.pre_mlp_units)]
            else:
                raise NotImplementedError()
            if args.bn:
                modules += [fnn.BatchNormFlow(num_inputs)]
            mask = 1 - mask
        model = fnn.FlowSequential(*modules)

        # initialize biases
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    exp_name = f'{args.exp_name if ("exp_name" in args) else "debug"}-{datetime.now().strftime("%m-%d-%H-%M-%S")}'
    default_data_dir = DEFAULT_DATA_DIR if not args.debug else '/tmp'
    writer = SummaryWriter(osp.join(default_data_dir, exp_name))
    with open(osp.join(default_data_dir, exp_name) + '/params.json', 'w') as fp:
        json.dump(vars(args), fp)
    log_file = open(osp.join(default_data_dir, exp_name) + '/logs.json', 'w')

    best_val_loss = float('inf')
    best_val_epoch = 0
    best_model = model

    # main loop
    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))

        train_loss = train(model, optimizer, train_loader, device, args.deterministic, args.gaussian)
        val_loss = validate(model, val_loader, device, args.deterministic, args.gaussian)

        if epoch - best_val_epoch >= 30:  # early stopping
            break

        if val_loss < best_val_loss:
            best_val_epoch = epoch
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)

        msg = [f'Train Loss: {train_loss:.4f}',
               f'Val Loss: {val_loss:.4f}',
               f'Best at epoch {best_val_epoch}: {best_val_loss:.4f}', '']
        print(*msg)
        log_file.writelines(msg)

        if epoch % 10 == 0:
            if args.one_step:
                utils.save_plot(epoch, model,val_dataset, osp.join(default_data_dir, exp_name))
            print('Saving best model ...', '')
            torch.save(best_model, osp.join(default_data_dir, exp_name) + "/model.pt")
            print('DONE')
    filename = f'{args.env}_flow_{args.mode}_{args.n_step}.pt'
    torch.save(best_model, osp.join(FLOW_PATH, filename))
    log_file.close()

if __name__ == '__main__':
    config = json.load(open(osp.join(osp.dirname(osp.abspath(__file__)), "default_params.json")))
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v[0], type=type(v[0]) if type(v[0]) != bool else utils.str2bool, help=v[1])
    args = parser.parse_args()
    main(args)
