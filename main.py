"""
Simple script to run a single experiment.
Parameters can be set directly in default_params.json or updated as command line arguments.
"""

import argparse
import json
import torch

from agent import SAC
from envs import make_env
from logger import EpochLogger
from modules import MLPActorCritic, CNNActorCritic
from replays import HERReplayBuffer
from trainer import Trainer
from utils import setup_logger_kwargs, set_seeds, str2bool
from wrappers import wrap_env_fn

DEFAULT_CONFIG_PATH = 'default_params.json'


def main(args):

    set_seeds(args.seed)
    torch.set_num_threads(args.num_threads)

    # Set up logger
    logger = EpochLogger(**setup_logger_kwargs(args.exp_name, args.seed, args.debug))
    logger.save_config(args)

    # Wrap environment to account for action prior, if enabled.
    env_fn = lambda: make_env(args)
    wrapped_env_fn = wrap_env_fn(env_fn, args)

    # Create agent
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l, lambda_param=args.lambda_param, lambda_init=args.lambda_init)
    sil_kwargs = dict(sil=args.sil, sil_bs=args.sil_bs, sil_weight=args.sil_weight,
                      sil_value_weight=args.sil_value_weight, sil_m=args.sil_m)
    prior_kwargs = dict(ratio=args.data_ratio, model=args.prior_model, cond=args.prior_cond, kl_reg=args.kl_reg,
                        use_prior=args.use_prior, initial=args.prior_initial, n_step=args.prior_n_step, schedule=args.prior_schedule,
                        smoothing=args.prior_smoothing, clamp=args.prior_clamp, one_step=args.one_step)

    agent = SAC(env_fn=wrapped_env_fn, logger=logger,
                actor_critic=MLPActorCritic if not args.visual else CNNActorCritic,ac_kwargs=ac_kwargs,
                gamma=args.gamma, polyak=args.polyak, lr=args.lr, alpha=args.alpha, beta=args.beta, prior_kwargs=prior_kwargs,
                clip_gradients=args.clip_gradients, use_polyrl=args.use_polyrl, gpu=args.gpu, epsilon=args.epsilon)

    if args.goal_cond and args.terminate_on_success:
        raise Exception('Goal-conditioning does not support variable-length episodes.')

    # Create replay buffer
    temp_env = wrapped_env_fn()
    buffer = HERReplayBuffer(obs_space=temp_env.observation_space, act_dim=temp_env.action_space.shape[0],
                             size=args.replay_size, T=temp_env.max_steps, her=args.her,
                             replay_k=args.replay_k, reward_fun=temp_env.compute_reward, prioritize=args.prioritize,
                             alpha=args.prioritize_alpha, beta=args.prioritize_beta, epsilon=args.prioritize_epsilon,
                             gamma=args.gamma, n_step=args.n_step_rew, clip_rew=args.clip_rew, prior_n_step=args.prior_n_step)

    # Create trainer
    trainer = Trainer(logger, steps_per_epoch=args.steps_per_epoch, start_steps=args.start_steps,
                      update_after=args.update_after, update_every=args.update_every,
                      batch_size=args.batch_size,
                      num_test_episodes=args.num_test_episodes, max_ep_len=args.max_ep_len,
                      save_freq=args.save_freq, sil_kwargs=sil_kwargs, prior_kwargs=prior_kwargs,
                      bc_epochs=args.bc_epochs, rand_init_cond=args.rand_init_cond)

    trainer.train(agent, buffer, wrapped_env_fn, args.epochs)


if __name__ == '__main__':
    # Load default parameters
    config = json.load(open(DEFAULT_CONFIG_PATH))

    # Update default parameters according to command line
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v[0], type=type(v[0]) if type(v[0]) != bool else str2bool, help=v[1])
    args = parser.parse_args()
    main(args)
