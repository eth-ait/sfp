"""
Script for collecting a large amount of task-agnostic expert trajectories.
Saves each trajectory separately to disk. Can be multiple times in parallel.
"""

import os
import argparse
import os.path as osp
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from envs import MazeEnv

def save_trajectory(name, seed=42):
    """ Saves a single trajectory"""
    if 'primitive' not in name:
        raise NotImplementedError('Only primitive datasets have been implemented.')

    data_dir = osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{name}/')
    np.random.seed(seed+42)

    if 'maze' in name:
        env = MazeEnv(name='room-v1',
                       sparsity=15.0,
                       terminate_on_success=False,
                       neg_rew=False,
                       goal_cond=False,
                       visual=False)
    else:
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v2-goal-observable']
        env = env_cls(seed=np.random.randint(low=1000, high=20000000))
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env.reset_model()  # might not be necessary
        _HAND_LOW = [-0.45, .4, .1]
        _HAND_HIGH = [+0.45, 0.85, .45]
    state = env.reset()

    if 'maze' in name:
        goal = np.random.uniform(low=-env.sparsity+0.5, high=env.sparsity-0.5, size=(2,))
    else:
        goal = np.array([np.random.uniform(a, b) for a, b in zip(_HAND_LOW, _HAND_HIGH)])
        env._target_pos = goal

    if 'maze' in name:
        env.visual = True
        old_goal = env.goal
        env.goal = goal
        obs = (env._get_obs() * 255.).astype(np.uint8)
        env.goal = old_goal
        env.visual = False
    else:
        obs = env.sim.render(64, 64, mode='offscreen', camera_name='corner')[:,:,::-1].astype(np.uint8)
        for site in env._target_site_config:
            env._set_pos_site(*site)
        obs = env.sim.render(64, 64, mode='offscreen', camera_name='corner')[:,:,::-1].astype(np.uint8)
        obs = np.moveaxis(obs, -1, 0)

    terminals = []
    actions = []
    states = [state if 'maze' in name else state[:36]]
    goals = [goal]
    images = [obs]
    steps_since_start = 0
    for _ in range(500):
        if 'maze' in name:
            # straight to goal!
            action = goal - state
            action = action / np.sqrt(np.sum(action ** 2))
            # action = np.clip(action, -1, 1)
            action = action + np.random.normal(size=action.shape) * 0.05
        else:
            action = env.action_space.sample()
            action[:3] = 5. * (goal - state[:3])

            action = np.clip(action, -1, 1)
            action = np.random.normal(action, .01*np.ones(4)*(env.action_space.high - env.action_space.low))
        state, _, _, _ = env.step(action)
        steps_since_start += 1
        if 'maze' in name:
            env.visual = True
            old_goal = env.goal
            env.goal = goal
            obs = (env._get_obs() * 255.).astype(np.uint8)
            env.goal = old_goal
            env.visual = False
        else:
            obs = env.sim.render(64, 64, mode='offscreen', camera_name='corner')[:,:,::-1].astype(np.uint8)
            obs = np.moveaxis(obs, -1, 0)
        actions.append(action)
        states.append(state if 'maze' in name else state[:36])
        goals.append(goal)
        images.append(obs)
        terminals.append(0)
        if 'maze' in name:
            if (np.sqrt(np.sum((goal - state)**2)) <= 0.5) or (steps_since_start > 4*env.sparsity):
                goal = np.random.uniform(low=-env.sparsity+0.5, high=env.sparsity-0.5, size=(2,))
                steps_since_start = 0
        else:
            if (np.linalg.norm(state[:3] - goal, axis=-1) <= 0.05):
                goal = np.array([np.random.uniform(a, b) for a, b in zip(_HAND_LOW, _HAND_HIGH)])
                env._target_pos = goal
                env.sim.render(64, 64, mode='offscreen', camera_name='corner')[:,:,::-1].astype(np.uint8)
                for site in env._target_site_config:
                    env._set_pos_site(*site)
                steps_since_start = 0
    terminals[-1] = 1

    buffer = {'actions': np.array(actions),
                'states': np.array(states)[:-1],
                'goals': np.array(goals)[:-1],
                'observations': np.array(images)[:-1],
                'terminals': np.array(terminals),
                'timeouts': np.zeros_like(terminals)}
    np.save(osp.join(data_dir, f'{name}_{seed}.npy'), buffer)

if __name__ == "__main__":
    num_trajs = 4000
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--env', default='maze_primitive', type=str, help='environment for trajectory extraction (one of point-maze and meta-world)')
    args = parser.parse_args()

    if args.env == 'meta-world':
        name = 'sawyer_primitive' 
    elif args.env == 'point-maze':
        name = 'maze_primitive'
    else:
        raise Exception('Unknown environment name.')

    while True:
        data_dir = osp.join(osp.abspath(osp.expanduser("~")), f'.d4rl/datasets/{name}/')
        existing_trajs = [int(''.join([c if c.isdigit() else '' for c in name])) for name in os.listdir(data_dir)]
        trajs_to_save = [i for i in np.arange(num_trajs) if i not in existing_trajs]
        if not trajs_to_save:
            exit(0)
        seed = np.random.choice(trajs_to_save)
        print(f'Seed: {seed}')
        save_trajectory(name=name, seed=seed)
