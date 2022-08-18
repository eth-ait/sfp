import time
import numpy as np
import torch
from utils import BCDataset

class Trainer:
    '''
    Trainer class implementing standard RL training loops.
    Args:
        logger : Logger instance.
        steps_per_epoch (int): Number of environment steps to perform at each epoch.
        start_steps (int): Initial steps of random exploration.
        update_after (int): Number of steps to wait before updating networks.
        update_every (int): Update frequency in steps.
        batch_size (int): Batch size for optimization.
        num_test_episodes (int): Number of test episodes.
        max_ep_len (int): Maximum number of steps for each episode.
        save_freq (int): Number of epochs between model saving.
        sil_kwargs (dict): Parameters for Self Imitation Learning.
    '''

    def __init__(self, logger, steps_per_epoch=4000, start_steps=10000, update_after=1000, update_every=50,
                 batch_size=100, num_test_episodes=10, max_ep_len=1000, save_freq=1, sil_kwargs={}, prior_kwargs = {},
                 bc_epochs=0, rand_init_cond=True):
        self.logger = logger
        self.steps_per_epoch = steps_per_epoch
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.sil = sil_kwargs['sil']
        self.sil_bs = sil_kwargs['sil_bs']
        self.sil_weight = sil_kwargs['sil_weight']
        self.sil_value_weight = sil_kwargs['sil_value_weight']
        self.sil_m = sil_kwargs['sil_m']
        self.prior_n_step = prior_kwargs['n_step']
        self.bc_epochs = bc_epochs
        self.rand_init_cond=rand_init_cond
        self.prior_schedule = prior_kwargs['schedule']

    def train(self, agent, buffer, env_fn, epochs=100):
        '''
        Training loop.
        Args:
            agent: RL agent.
            buffer: Replay buffer.
            env_fn: Environment constructor.
            epochs: Number of epochs for training.
        '''
        env, test_env = env_fn(), env_fn()
        env.action_space.np_random.seed(np.random.randint(2**32-1))

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * epochs
        last_update = 0
        epoch_start = 0
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0
        # Initialize previous actions by sampling dataset
        last_action = self.sample_actions_from_dataset(env)
        trajectory = dict(obs=[o], act=[], rew=[], done=[], last_act=[])
        self.n_env_reset = 0
        agent.pre_ep(self.n_env_reset)

        if self.bc_epochs > 0:
            loader = torch.utils.data.DataLoader(BCDataset(env._short_name, visual=env.visual),
                                                 batch_size=self.batch_size, shuffle=True)
            for _ in range(self.bc_epochs):
                for batch in loader:
                    agent.clone(batch)

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            agent.weight_schedule = 1. if self.prior_schedule == 0. else min(1., t/self.prior_schedule)

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a = agent.act(o, last_action=last_action)
            else:
                a = agent.initial_explore(env, o, last_action=last_action, step_number=t)
            # Memorize previous action
            last_action = np.concatenate([last_action[1:], a[np.newaxis, ...]], 0)

            # Step the env
            o2, r, d, _ = env.step(a)
            agent.post_step(o, o2)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            for k, v in zip(('obs', 'act', 'rew', 'done', 'last_act'), (o2, a, r, d, last_action.reshape(-1))):
                trajectory[k].append(v)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len >= self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, SuccessRate=ep_ret > 0)
                buffer.store_episode(trajectory)
                o, ep_ret, ep_len = env.reset(), 0, 0
                # Update previous actions by sampling dataset
                last_action = self.sample_actions_from_dataset(env)
                trajectory = dict(obs=[o], act=[], rew=[], done=[], last_act=[])
                self.n_env_reset += 1
                agent.pre_ep(self.n_env_reset)

            # Update handling
            if t >= self.update_after and (t - last_update) >= self.update_every:
                last_update = t
                for _ in range(self.update_every):
                    batch, indices = buffer.sample(self.batch_size, progress=float(t/total_steps), force_uniform=self.sil)
                    priorities = agent.update(data=batch)
                    if not self.sil:
                        buffer.update_priorities(priorities, indices)
                if self.sil:
                    # Do Self Imitation Learning
                    for _ in range(self.update_every * self.sil_m):
                        batch, indices = buffer.sample(self.sil_bs, float(t/total_steps))
                        priorities = agent.imitate(data=batch, sil_weight=self.sil_weight, value_weight=self.sil_value_weight)
                        buffer.update_priorities(priorities, indices)

            # End of epoch handling
            if (t + 1) >= epoch_start + self.steps_per_epoch:
                epoch_start = t + 1
                epoch = (t + 1) // self.steps_per_epoch

                # Temporarily disabled due to lambda expression in actor critic
                # Save model
                # if (epoch % self.save_freq == 0) or (epoch == epochs):
                #     self.logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent(agent, test_env)

                # Record videos of agent, disabled to save memory/compute
                # if epoch % 10 == 1:
                #     self.record_agent(agent, test_env, t)

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('SuccessRate', with_min_and_max=True)
                self.logger.log_tabular('TestSuccessRate', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('Weights', with_min_and_max=True)
                self.logger.log_tabular('Priorities', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LogStd', with_min_and_max=True)
                self.logger.log_tabular('H', with_min_and_max=True)
                self.logger.log_tabular('PriorP', with_min_and_max=True)
                self.logger.log_tabular('MixingWeight', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()

    def test_agent(self, agent, test_env):
        """
        Testing loop.
        Args:
            agent: RL agent.
            test_env: Costructor for test environment.
        """
        for _ in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            # Sample first action
            last_action = self.sample_actions_from_dataset(test_env)
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                a = agent.act(o, True, last_action=last_action)
                o, r, d, _ = test_env.step(a)
                last_action = np.concatenate([last_action[1:], a[np.newaxis, ...]], 0)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len, TestSuccessRate=ep_ret > 0)

    def record_agent(self, agent, test_env, global_step):
        """
        Records videos of a policy rollout.
        Args:
            agent: RL agent.
            test_env: Costructor for test environment.
            global_step: Number of steps since the beginning of training.
        """
        if not (hasattr(test_env, 'sim') and hasattr(test_env, 'metadata')):
            return  # environment does not support rendering
        # Collect a video for both training and testing policies
        for video_name, deterministic in [('train', False), ('test', True)]:
            o, d, ep_len = test_env.reset(), False, 0
            # Sample first action
            last_action = self.sample_actions_from_dataset(test_env)
            frames = []
            while not (d or (ep_len >= self.max_ep_len)):
                a = agent.act(o, deterministic, last_action=last_action)
                o, _, d, _ = test_env.step(a)
                last_action = np.concatenate([last_action[1:], a[np.newaxis, ...]], 0)
                ep_len += 1
                frames.append(test_env.sim.render(320, 240, mode='offscreen', camera_name='corner')[:,:,::-1])
            frames = np.expand_dims(np.stack([np.moveaxis(f, -1, 0) for f in frames]), 0)
            self.logger.log_video(video_name, frames, global_step, test_env.metadata['video.frames_per_second'])

    def sample_actions_from_dataset(self, env):
        if not self.rand_init_cond:
            idx = np.random.choice(len(env.get_primitive_dataset()['actions'])//500) * 500
        elif self.prior_n_step == 1:
            idx = np.random.choice(len(env.get_primitive_dataset()['actions']))
        else:
            idx = np.random.choice(len(env.get_primitive_dataset()['actions'])-self.prior_n_step)
        return env.get_primitive_dataset()['actions'][idx:idx+self.prior_n_step]
