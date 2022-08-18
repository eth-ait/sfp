"""
Implementation of environments and data collection methods.
Two main families of environments are present:
- a family of PointMazes, adapted from an implementation by Salesforce Research;
- robotic manipulations environments adapted from the metaworls suite.
"""

import os.path as osp
import functools
import numpy as np
import d4rl
import gym
from gym import spaces, Env
from gym.spaces import Box
# from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from utils import DATASET_DIR

SUPPORTED_SAWYER_ENVS = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'door-close-v2',
                         'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2', 'button-press-v2',
                         'peg-insert-side-v2', 'window-open-v2', 'window-close-v2', 'sweep-v2',
                         'basketball-v2', 'shelf-place-v2', 'sweep-into-v2', 'lever-pull-v2']


def make_env(args):
    """ Creates environment according to parameters. """
    if args.env in ['maze-v1', 'corridor-v1', 'room-v1']:
        return MazeEnv(name=args.env, sparsity=int(args.sparsity), terminate_on_success=args.terminate_on_success, neg_rew=args.neg_rew, goal_cond=args.goal_cond, visual=args.visual)
    elif args.env in SUPPORTED_SAWYER_ENVS:
        return SawyerWrapper(args.env, terminate_on_success=args.terminate_on_success, fix_seed=args.fix_env_seed,  neg_rew=args.neg_rew, goal_cond=args.goal_cond, visual=args.visual)
    elif any([s in args.env for s in ['swimmer', 'hopper', 'halfcheetah', 'ant', 'walker']]):
        return MujocoEnv(name=args.env, sparsity=args.sparsity, bias=args.bias, permute_action=args.permute_action)
    else:
        raise Exception('I am pretty sure you do not really want to create this environment.')


class SawyerWrapper:
    """
    Interface to metaworld environments.
    Args:
        env_name (string): Name of the environment.
        terminate_on_success (bool): Enables sending a done signal on goal achievement.
        fix_seed (bool): Flag to disable random goal sampling at each reset.
        neg_rew (bool): Scales rewards from [0,1] to [-1,0].
        goal_cond (bool): Includes achieved and desired goals in observations.
        visual (bool): Enables returning RGB observations instead of vectorized states.
    """

    def __init__(self, env_name, terminate_on_success=False, fix_seed=True, neg_rew=False, goal_cond=False, visual=False):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
        env = env_cls(seed=0)
        if fix_seed:
            env.random_init = False
        # Some magic that is performed in the example script from metaworld
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env.reset_model()  # Might not be necessary
        self._env = env
        self._env_name = env_name
        self._short_name = 'sawyer'
        self.terminate_on_success = terminate_on_success
        self.goal_cond = goal_cond
        self.visual = visual
        self.neg_rew = neg_rew
        self.max_steps = 500
        self.obs = None

        if self.visual:
            self.observation_space = spaces.Box(0., 1., [3, 64, 64], dtype='float32')
            self.goal_cond=False
        else:
            if self.goal_cond: 
                self.observation_space = spaces.Dict(dict(
                    desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                    achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                    observation=spaces.Box(-np.inf, np.inf, shape=(36,), dtype='float32'),
                ))
            else:
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)

    
    def _get_obs(self):
        """ Returns the current observation. """
        if self.visual:
            obs = self._env.sim.render(64, 64, mode='offscreen', camera_name='corner')[:,:,::-1].astype(np.float32) / 255.
            return np.moveaxis(obs, -1, 0)
        if not self.goal_cond:
            return self.obs[:36]  # Discard goal
        state = self.obs
        return {
            'observation': state[:36].copy(),
            'achieved_goal': state[:3].copy() if self._env_name == 'reach-v2' else state[4:7].copy(),
            'desired_goal': state[36:].copy()
        }

    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes rewards according to currently achieved and desired goal.
        metaworld did not expose this function, so it had to be extracted manually.
        Args:
            achieved_goal : 1D or 2D array containing one or a batch of achieved goals.
            goal : 1D or 2D array containing one or a batch of desired goals.
            info (dict): Additional information (currently not in use).
        """
        if self._env_name == 'reach-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.06).astype(np.float32)
            # reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.05).astype(np.float32)
        elif self._env_name == 'door-open-v2':
            reward = (np.transpose(np.abs(achieved_goal - goal))[0] <= 0.08).astype(np.float32)
        elif self._env_name == 'door-close-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.08).astype(np.float32)
        elif self._env_name in ['window-open-v2', 'window-close-v2']:
            reward = (np.transpose(np.abs(achieved_goal - goal))[0] <= 0.05).astype(np.float32)
        elif self._env_name in ['push-v2', 'sweep-v2']:
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.05).astype(np.float32)
        elif self._env_name in ['pick-place-v2', 'shelf-place-v2']:
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.07).astype(np.float32)
        elif self._env_name == 'drawer-open-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.03).astype(np.float32)
        elif self._env_name == 'drawer-close-v2':
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.055).astype(np.float32)
        elif self._env_name == 'button-press-topdown-v2':
            reward = (np.transpose(np.abs(achieved_goal - goal))[2] <= 0.02).astype(np.float32)
        elif self._env_name == 'button-press-v2':
            reward = (np.transpose(np.abs(achieved_goal - goal))[1] <= 0.02).astype(np.float32)
        elif self._env_name == 'basketball-v2':
            goal = goal.copy()
            goal[2] = 0.3
            target_to_obj = (achieved_goal - goal) * np.array([[1., 1., 2.]])
            reward = (np.linalg.norm(target_to_obj, axis=-1)  <= 0.08).astype(np.float32)
        elif self._env_name == 'sweep-into-v2':
            goal = goal.copy()
            goal[:, 2] = achieved_goal[:, 2]
            reward = (np.linalg.norm(achieved_goal - goal, axis=-1) <= 0.05).astype(np.float32)
        elif self._env_name == 'lever-pull-v2':
            raise Exception ('Sadly this reward is not easily computable from states.')
        elif self._env_name == 'peg-insert-side-v2':
            raise Exception ('Reward computation requires access to simulation.')
        else:
            raise Exception(f'{self._env_name} not implemented yet.')
        return reward

    def __getattr__(self, attr):
        """ Handles wrapping. """
        return getattr(self._env, attr)

    def reset(self):
        """ Resets the environment. """
        self.obs = self._env.reset()
        return self._get_obs()

    def step(self, action):
        """ Advances simulation by a step. """
        self.obs, _, done, info = self._env.step(action)
        if self.goal_cond and self._env_name != 'peg-insert-side-v2':
            reward = self.compute_reward(self._get_obs()['achieved_goal'], self._get_obs()['desired_goal'], {})
        else:
            reward = 1.0 if info['success'] else 0.0  # Reward signal is adapted to be sparse.
        done = True if ((self._env.curr_path_length >= self.max_steps) or (self.terminate_on_success and reward > 0)) else done
        reward = reward - 1 if self.neg_rew else reward
        return self._get_obs(), reward, done, info

    def get_primitive_flow_checkpoint(self, model, cond, n_step, ratio=1.0, one_step=False):
        """ Returns the name of the file containing a flow model trained on task-agnostic trajectories. """
        if ratio < 1.0:
            return f'/{self._short_name}_primitive_{model}_{cond}_{n_step}_{ratio}.pt'
        if one_step:
            return f'/{self._short_name}_primitive_{model}_{cond}_{n_step}_one_step.pt'
        return f'/{self._short_name}_primitive_{model}_{cond}_{n_step}.pt'
    
    def get_primitive_dataset(self):
        """ Returns a dataset of task-agnostic expert trajectories. """
        return get_primitive_dataset(self._short_name)


class MujocoEnv:

    def __init__(self, name, sparsity, bias, permute_action):

        if 'swimmer' in name:
            env = gym.make('Swimmer-v2')
            self._short_name = 'swimmer'
        elif 'hopper' in name:
            env = gym.make('Hopper-v2')
            self._short_name = 'hopper'
        elif 'walker' in name:
            env = gym.make('Walker2d-v2')
            self._short_name = 'walker'
        elif 'ant' in name:
            env = gym.make('Ant-v2')
            self._short_name = 'ant'
        elif 'halfcheetah' in name:
            env = gym.make('HalfCheetah-v2')
            self._short_name = 'halfcheetah'
        else:
            raise Exception('Unknown Mujoco environment.')

        self._env = env
        self._env_name = name
        self.sparsity = sparsity
        self.max_steps = 1000
        self.obs = None
        self.starting_obs = None
        self.n_steps = 0
        self.visual = False
        self.permute_action = permute_action

        if 'noisy' in self._env_name:
            self.process = lambda x: x + np.random.normal(size=x.shape) * 2
        if 'biased' in self._env_name:
            self.process = lambda x: x + bias
        else:
            self.process = lambda x: x
    
    def _get_obs(self):
        """ Returns the current observation. """
        return self.obs

    def __getattr__(self, attr):
        """ Handles wrapping. """
        return getattr(self._env, attr)

    def reset(self):
        """ Resets the environment. """
        self.obs = self._env.reset()
        self.obs = self.process(self.obs)
        self.starting_obs = np.copy(self._env.sim.data.qpos)
        self.n_steps = 0
        return self._get_obs()

    def step(self, action):
        """ Advances simulation by a step. """
        if self.permute_action:
            action = action[::-1]
        self.obs, rew, done, info = self._env.step(action)
        self.obs = self.process(self.obs)
        self.n_steps += 1
        if 'sparse' in self._env_name:
            rew = 1.0 if (self._env.sim.data.qpos[0] - self.starting_obs[0]) > self.sparsity else 0.
        done = True if (self.n_steps >= self.max_steps) else done
        return self._get_obs(), rew, done, info

    def get_primitive_flow_checkpoint(self, model, cond, n_step, ratio=1.0, one_step=False):
        """ Returns the name of the file containing a flow model trained on task-agnostic trajectories. """
        if ratio < 1.0:
            return f'/{self._short_name}_primitive_{model}_{cond}_{n_step}_{ratio}.pt'
        if one_step:
            return f'/{self._short_name}_primitive_{model}_{cond}_{n_step}_one_step.pt'
        return f'/{self._short_name}_primitive_{model}_{cond}_{n_step}.pt'
    
    def get_primitive_dataset(self):
        """ Returns a dataset of task-agnostic expert trajectories. """
        return get_primitive_dataset(self._short_name)


class MazeEnv:
    """
    Simple implementation of a PointMaze environment.
    Args:
        name (string): Name of the maze.
        max_steps (int): Maximum number of steps in an episode.
        sparsity (int): Length of corridors used in some simple mazes.        
        terminate_on_success (bool): Enables sending a done signal on goal achievement.
        fix_seed (bool): Flag to disable random goal sampling at each reset.
        neg_rew (bool): Scales rewards from [0,1] to [-1,0].
        goal_cond (bool): Includes achieved and desired goals in observations.
        visual (bool): Enables returning RGB observations instead of vectorized states.
    """

    def __init__(self, name, max_steps=500, sparsity=15, terminate_on_success=False, neg_rew=False, goal_cond=False, visual=False):
        self._env_name = name
        self._short_name = 'maze'
        self.max_steps = max_steps
        self.dist_threshold = 1.2
        self.state_size = 2
        self.action_size = 2
        self.goal_cond = goal_cond
        self.visual=visual
        self.action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.neg_rew = neg_rew

        self.state = None
        self.goal = None
        self.t = None
        self.sparsity = sparsity
        self.terminate_on_success = terminate_on_success
        if name == 'corridor-v1':
            self.sim = make_corridor(self.sparsity)
        elif name == 'room-v1':
            self.sim = make_room(self.sparsity)
        elif name =='maze-v1':
            self.sim = make_maze(self.sparsity)
        else:
            raise NotImplementedError
        self.metadata = {'video.frames_per_second': 30}

        if self.visual:
            self.observation_space = spaces.Box(0., 1., [3, 64, 64], dtype='float32')
            self.goal_cond = False
        else:
            low = np.array([min(x) - .5 for x in zip(*[v['loc'] for k, v in self.sim._segments.items()])])
            high = np.array([max(x) + .5 for x in zip(*[v['loc'] for k, v in self.sim._segments.items()])])
            self.scale = lambda x: (x - low) / (high-low)
            self.unscale = lambda x: x * (high-low) + low
            if self.goal_cond: 
                self.observation_space = spaces.Dict(dict(
                    desired_goal=spaces.Box(low=0., high=1., shape=(2,), dtype='float32'),
                    achieved_goal=spaces.Box(low=0., high=1., shape=(2,), dtype='float32'),
                    observation=spaces.Box(low=0., high=1., shape=(2,), dtype='float32'),
                ))
            else:
                self.observation_space = Box(low=0., high=1., shape=(2,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        """ Returns the current observation. """
        if self.visual:
            if self.canvas is None:
                # self.canvas contains a rendering of the maze structure. It is recomputed at each reset.
                # self.to_coord is a lambda projecting environment coordinates to pixel coordinates.
                self.canvas, self.to_coord = self.sim.get_canvas()
            # Render the agent's and the goal's positions
            enlarge = lambda pos: [(pos[0] - 1 + (i // 3), pos[1] - 1 + (i % 3)) for i in range(9)]
            clip = lambda l: [np.clip(e, 0, 63) for e in l]
            to_idx = lambda l: (tuple([e[0] for e in l]), tuple([e[1] for e in l]))
            canvas = self.canvas.copy()
            canvas[to_idx(clip(enlarge(self.to_coord(self.state))))] = [1., 0., 0.]
            canvas[to_idx(clip(enlarge(self.to_coord(self.goal))))] = [0., 1., 0.]
            return np.moveaxis(np.flip(canvas, 0), -1, 0)
        if not self.goal_cond:
            return self.scale(self.state.copy())
        return {
            'observation': self.scale(self.state.copy()),
            'achieved_goal': self.scale(self.state.copy()),
            'desired_goal': self.scale(self.goal.copy())
        }

    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes rewards according to currently achieved and desired goal by measuring euclidean distances.
        Args:
            achieved_goal : 1D or 2D array containing one or a batch of achieved goals.
            goal : 1D or 2D array containing one or a batch of desired goals.
            info (dict): Additional information (currently not in use).
        """
        achieved_goal, goal = self.unscale(achieved_goal), self.unscale(goal)
        return (np.linalg.norm(achieved_goal - goal, axis=-1) <= self.dist_threshold).astype(np.float32)

    def reset(self):
        """ Resets the environment. """
        self.state = np.array(self.sim.sample_start())
        self.goal = np.array(self.sim.sample_goal())
        self.t = 0
        self.canvas = None
        return self._get_obs()

    def step(self, action):
        """ Advances simulation by a step. """
        self.t += 1
        self.state = self.sim.move((self.state[0], self.state[1]), (action[0], action[1]))
        self.state = np.array(self.state)
        if self.goal_cond:
            reward = self.compute_reward(self._get_obs()['achieved_goal'], self._get_obs()['desired_goal'], {})
        else:
            reward = (np.linalg.norm(self.state - self.goal, axis=-1) <= self.dist_threshold).astype(np.float32)
        done = (self.t >= self.max_steps) or (self.terminate_on_success and reward > 0)
        reward = reward - 1 if self.neg_rew else reward
        return self._get_obs(), reward, done, {}
    
    def get_primitive_flow_checkpoint(self, model, cond, n_step, ratio=1.0, one_step=False):
        """ Returns the name of the file containing a flow model trained on task-agnostic trajectories. """
        if ratio < 1.0:
            return f'/maze_primitive_{model}_{cond}_{n_step}_{ratio}.pt'
        if one_step:
            return f'/maze_primitive_{model}_{cond}_{n_step}_one_step.pt'
        return f'/maze_primitive_{model}_{cond}_{n_step}.pt'

    def get_primitive_dataset(self):
        """ Returns a dataset of task-agnostic expert trajectories. """
        return get_primitive_dataset(self._short_name)


class Maze:
    """
    Physics simulation of a PointMaze environment. Credits to SalesForce.
    """

    def __init__(self, *segment_dicts, goal_squares=None, start_squares=None):
        self.pos, self.goal = None, None
        self._segments = {'origin': {'loc': (0.0, 0.0), 'connect': set()}}
        self._locs = set()
        self._locs.add(self._segments['origin']['loc'])
        self._walls = set()
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(self._segments['origin']['loc'], direction))
        self._last_segment = 'origin'
        self.canvas = None

        if goal_squares is None:
            self._goal_squares = None
        elif isinstance(goal_squares, str):
            self._goal_squares = [goal_squares.lower()]
        elif isinstance(goal_squares, list):
            self._goal_squares = goal_squares
        else:
            raise TypeError

        if start_squares is None:
            self.start_squares = ['origin']
        elif isinstance(goal_squares, str):
            self.start_squares = [start_squares.lower()]
        elif isinstance(goal_squares, list):
            self.start_squares = start_squares
        else:
            raise TypeError

        for segment_dict in segment_dicts:
            self._add_segment(**segment_dict)
        self._finalize()

    @staticmethod
    def _wall_line(coord, direction):
        x, y = coord
        if direction == 'up':
            w = [(x - 0.5, x + 0.5), (y + 0.5, y + 0.5)]
        elif direction == 'right':
            w = [(x + 0.5, x + 0.5), (y + 0.5, y - 0.5)]
        elif direction == 'down':
            w = [(x - 0.5, x + 0.5), (y - 0.5, y - 0.5)]
        elif direction == 'left':
            w = [(x - 0.5, x - 0.5), (y - 0.5, y + 0.5)]
        else:
            raise ValueError
        w = tuple([tuple(sorted(line)) for line in w])
        return w

    def _add_segment(self, name, anchor, direction, connect=None, times=1):
        name = str(name).lower()
        original_name = str(name).lower()
        if times > 1:
            assert connect is None
            last_name = str(anchor).lower()
            for time in range(times):
                this_name = original_name + str(time)
                self._add_segment(name=this_name.lower(), anchor=last_name, direction=direction)
                last_name = str(this_name)
            return

        anchor = str(anchor).lower()
        assert anchor in self._segments
        direction = str(direction).lower()

        final_connect = set()

        if connect is not None:
            if isinstance(connect, str):
                connect = str(connect).lower()
                assert connect in ['up', 'down', 'left', 'right']
                final_connect.add(connect)
            elif isinstance(connect, (tuple, list)):
                for connect_direction in connect:
                    connect_direction = str(connect_direction).lower()
                    assert connect_direction in ['up', 'down', 'left', 'right']
                    final_connect.add(connect_direction)

        sx, sy = self._segments[anchor]['loc']
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx -= 1
            final_connect.add('right')
        elif direction == 'right':
            dx += 1
            final_connect.add('left')
        elif direction == 'up':
            dy += 1
            final_connect.add('down')
        elif direction == 'down':
            dy -= 1
            final_connect.add('up')
        else:
            raise ValueError

        new_loc = (sx + dx, sy + dy)
        assert new_loc not in self._locs

        self._segments[name] = {'loc': new_loc, 'connect': final_connect}
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(new_loc, direction))
        self._locs.add(new_loc)

        self._last_segment = name

    def _finalize(self):
        for segment in self._segments.values():
            n = segment['loc']
            adjacents = {'right': (n[0]+1, n[1]),
                         'left': (n[0]-1, n[1]),
                         'up': (n[0], n[1]+1),
                         'down': (n[0], n[1]-1)}
            segment['connect'] = [k for k, v in adjacents.items() if v in self._locs]
            for c_dir in list(segment['connect']):
                wall = self._wall_line(segment['loc'], c_dir)
                if wall in self._walls:
                    self._walls.remove(wall)

        if self._goal_squares is None:
            self.goal_squares = [self._last_segment]
        else:
            self.goal_squares = []
            for gs in self._goal_squares:
                assert gs in self._segments
                self.goal_squares.append(gs)

    def sample_start(self):
        min_wall_dist = 0.05

        s_square = self.start_squares[np.random.randint(low=0, high=len(self.start_squares))]
        s_square_loc = self._segments[s_square]['loc']

        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = s_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        self.pos = loc[0], loc[1]
        return loc[0], loc[1]

    def sample_goal(self, min_wall_dist=None):
        g_square = self.goal_squares[np.random.randint(low=0, high=len(self.goal_squares))]
        g_square_loc = self._segments[g_square]['loc']
        self.goal = g_square_loc[0], g_square_loc[1]
        return g_square_loc[0], g_square_loc[1]

    def move(self, coord_start, coord_delta, depth=None):
        if depth is None:
            depth = 0
        cx, cy = coord_start
        loc_x0 = np.round(cx)
        loc_y0 = np.round(cy)
        dx, dy = coord_delta
        loc_x1 = np.round(cx + dx)
        loc_y1 = np.round(cy + dy)
        d_loc_x = int(np.abs(loc_x1 - loc_x0))
        d_loc_y = int(np.abs(loc_y1 - loc_y0))
        xs_crossed = [loc_x0 + (np.sign(dx) * (i + 0.5)) for i in range(d_loc_x)]
        ys_crossed = [loc_y0 + (np.sign(dy) * (i + 0.5)) for i in range(d_loc_y)]

        rds = []

        for x in xs_crossed:
            r = (x - cx) / dx
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'right' if dx > 0 else 'left'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        for y in ys_crossed:
            r = (y - cy) / dy
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'up' if dy > 0 else 'down'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        # The wall will only stop the agent in the direction perpendicular to the wall
        if rds:
            rds = sorted(rds)
            r, direction = rds[0]
            if depth < 3:
                new_dx = r * dx
                new_dy = r * dy
                repulsion = float(np.abs(np.random.rand() * 0.01))
                if direction in ['right', 'left']:
                    new_dx -= np.sign(dx) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = (0.0, (1 - r) * dy)
                else:
                    new_dy -= np.sign(dy) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = ((1 - r) * dx, 0.0)
                return self.move(partial_coords, remaining_delta, depth+1)
        else:
            r = 1.0

        dx *= r
        dy *= r
        self.pos = cx + dx, cy + dy
        return cx + dx, cy + dy

    def get_canvas(self):
        """ Renders the structure of the maze as an RGB. Automatically adjusts the spacial resolution. """
        canvas = np.zeros((64, 64, 3), dtype=np.float32)

        # Compute size of each cell, as well as padding and positioning
        wall_size = 0
        max_h = int(max([v['loc'][1] for _, v in self._segments.items()]))
        min_h = int(min([v['loc'][1] for _, v in self._segments.items()]))
        h_range = int(max_h - min_h + 1)
        max_w = int(max([v['loc'][0] for _, v in self._segments.items()]))
        min_w = int(min([v['loc'][0] for _, v in self._segments.items()]))
        w_range = int(max_w - min_w + 1)
        cell_size = (64 - wall_size) // max(h_range, w_range)
        cell_size -= wall_size
        w_padding = (64 - ((cell_size + wall_size) * w_range + wall_size)) // 2
        h_padding = (64 - ((cell_size + wall_size) * h_range + wall_size)) // 2

        # Compute a projection from environment coordinates to image coordinates
        to_coord = lambda pos: (int(np.rint(h_padding+wall_size+(pos[1]+0.5-min_h)*(cell_size+wall_size))),
                                int(np.rint(w_padding+wall_size+(pos[0]+0.5-min_w)*(cell_size+wall_size))))

        for _, v in self._segments.items():
            x, y = int(v['loc'][0]), int(v['loc'][1])
            idxs = (*to_coord((x-0.5, y-0.5)), *to_coord((x+0.5, y+0.5)))
            # Draw a single cell
            canvas[idxs[0]:(idxs[2]-wall_size), idxs[1]:(idxs[3]-wall_size)] = 1.
            for d in v['connect']:
                if d == 'left':
                    ridxs = (*to_coord((x-1.5, y)), *to_coord((x-0.5, y)))
                elif d == 'right':
                    ridxs = (*to_coord((x+0.5, y)), *to_coord((x+1.5, y)))
                elif d == 'up':
                    ridxs = (*to_coord((x, y+0.5)), *to_coord((x, y+1.5)))
                elif d == 'down':
                    ridxs = (*to_coord((x, y-1.5)), *to_coord((x, y-0.5)))
                # Draw a rectangle joining the two cells
                canvas[min(ridxs[0], idxs[0]):(max(ridxs[2], idxs[2])-wall_size), min(ridxs[1], idxs[1]):(max(ridxs[3], idxs[3])-wall_size)] = 1.

        return canvas, to_coord

    def render(self, w, h, mode='offscreen', camera_name='corner'):
        if self.canvas is None:
            self.canvas, self.to_coord = self.get_canvas()
        canvas = self.canvas.copy()
        enlarge = lambda pos: [(pos[0] - 1 + (i // 3), pos[1] - 1 + (i % 3)) for i in range(9)]
        clip = lambda l: [np.clip(e, 0, 63) for e in l]
        to_idx = lambda l: (tuple([e[0] for e in l]), tuple([e[1] for e in l]))
        canvas[to_idx(clip(enlarge(self.to_coord(self.pos))))] = [1., 0., 0.]
        canvas[to_idx(clip(enlarge(self.to_coord(self.goal))))] = [0., 1., 0.]
        return np.flip(canvas, 0)


def make_corridor(sparsity):
    """ Function creating an u-shaped maze. """

    sparsity = 60
    sparsity = int(sparsity)
    assert sparsity >= 1

    segments = []
    last = 'origin'
    for x in range(1, sparsity + 1):
        next_name = '0,{}'.format(x)
        segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
        last = str(next_name)

    assert last == '0,{}'.format(sparsity)

    up_size = 2

    for x in range(1, up_size+1):
        next_name = '{},{}'.format(x, sparsity)
        segments.append({'anchor': last, 'direction': 'up', 'name': next_name})
        last = str(next_name)

    assert last == '{},{}'.format(up_size, sparsity)

    for x in range(1, sparsity + 1):
        next_name = '{},{}'.format(up_size, sparsity - x)
        segments.append({'anchor': last, 'direction': 'left', 'name': next_name})
        last = str(next_name)

    assert last == '{},0'.format(up_size)

    return Maze(*segments, goal_squares=last)


def make_room(sparsity):

    sparsity = 14

    segments = []
    added = [(0,0)]
    names = [(x, y) for x in range(1-sparsity, sparsity) for y in range(1-sparsity, sparsity)] # if not (max(abs(x), abs(y)) == 1 and x*y != 0)]
    names.remove((0,0))
    to_add = None
    while names:
        for n in names:
            adjacents = [((n[0], n[1]+1), 'down', n),
                         ((n[0], n[1]-1), 'up', n),
                         ((n[0]+1, n[1]), 'left', n),
                         ((n[0]-1, n[1]), 'right', n)]
            for a in adjacents:
                if a[0] in added:
                    to_add = a
                    break
            if to_add is not None:
                break
        anchor = f'{to_add[0][0]},{to_add[0][1]}' if to_add[0] != (0,0) else 'origin'
        segments.append({'anchor': anchor, 'direction': to_add[1], 'name': f'{to_add[2][0]},{to_add[2][1]}'})
        added.append(to_add[2])
        names.remove(to_add[2])
        to_add = None

    l = sparsity-1
    return Maze(*segments, goal_squares=[f'{l},{l}', f'{-l},{l}', f'{l},{-l}', f'{-l},{-l}'])
 

def make_maze(length): 
    " Function creating a fixed intricate maze. "

    length = 6

    def get_coords(last_pos):
        if last_pos == 'origin':
            last_pos = '0,0'
        x, y = last_pos.split(',')
        x, y = int(x), int(y)
        return x, y    

    def get_name(last_pos, dir):
        x, y = get_coords(last_pos)
        x_offset = {'u': 0, 'd': 0, 'r': 1, 'l': -1}
        y_offset = {'u': 1, 'd': -1, 'r': 0, 'l': 0}
        x, y = x+x_offset[dir], y+y_offset[dir]
        return f'{x},{y}'
    
    ext = {'l': 'left', 'r': 'right', 'u': 'up', 'd': 'down'}
    paths = ['rruulll', 'rrd', 'rullddrr']
    segments = []
    for p in paths:
        last_pos = 'origin'
        for dir in p:
            for i in range(length):
                new_pos = get_name(last_pos, dir)
                new_segment = {'anchor': last_pos, 'direction': ext[dir], 'name': new_pos}
                if new_segment not in segments:
                    segments.append(new_segment)
                last_pos = new_pos
        
    return Maze(*segments, goal_squares=[f'{-length},{2*length}'], start_squares=[f'{-length},{-length}'])


def get_primitive_dataset(env_name, distilled=True):
    """
    Fetches a dataset of task-agnostic expert trajectories in similar environments.
    Args:
        env_name (string): Name of the environment.
    """
    path = osp.join(DATASET_DIR, f'{env_name}_primitive_distilled.npy')
    if osp.isfile(path):
        return np.load(path, allow_pickle=True).item()    
    raise Exception(f'Dataset generation has not happened yet for {env_name}.')
