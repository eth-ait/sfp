# SFP: State-free Priors for Exploration in Off-Policy Reinforcement Learning
## [Paper](https://arxiv.org/abs/2205.13528) | [Project Page](https://eth-ait.github.io/sfp/)

This repository contains the code to [our paper](https://arxiv.org/abs/2205.13528) published in Transactions on Machine Learning Research (TMLR) 2022.

## Environment Setup

- A Mujoco installation is necessary to run most of our code. Installation details can be found [here](https://github.com/openai/mujoco-py).

- We recommend pipenv for creating and managing virtual environments (dependencies for other environment managers can be found in `Pipfile`). The virtual environment can be simply created and activated with

```
cd sfp
pipenv install
pipenv shell
```

## Generating Expert Trajectories

- To generate expert trajectories for a suite of environments

```
python scripts/collect_large_dataset.py --env suite_name
python scripts/distill_dataset.py --env suite_name
```
- Running this script once for the *point-maze* suite and once for the *meta-world* suite is sufficient for all experiments.
- The scripts saves trajectories in `~/.d4rl/datasets`.

## Training Prior Models

- Code for training prior models is contained in `flows`. After expert data is generated, a flow-based model can be trained with:
```
python flows/main.py --env suite_name
```
- By default, training metrics are logged to `~\logs` and the model is saved to `~\flows`.

## Training Downstream RL

- An agent can be trained on a downstream task with
```
python main.py --env env_name
```
- Results are once again logged to `~\logs`.

## Configuration

Default directories can be edited in `utils.py`.
All settings can be handled by editing `default_params.json` in the correct directory (either the main directory of `flows`).

### Flow Training
| Param         | Default           | Info  |
| ------------- | ------------- | -----:|
| env | "meta-world" | environment suite |
| mode | "action" | conditioning variable (also 'state', 'image' or 'action+state') |
| n_step | 1 | number of steps to condition upon |
| batch_size | 400 | input batch size for training |
| test_batch_size | 1000 | input batch size for testing |
| epochs | 100 | number of epochs to train for |
| lr | 0.0001 | learning rate |
| num_blocks | 6 | number of invertible blocks |
| num_hidden | 128 | number of hidden units |
| t_act | relu | translation activation |
| s_act | relu | shift activation |
| bn | True | batch normalization |
| shared | True | whether to share parameters in the coupling layers |
| pre_mlp | True | whether to preprocess actions before pushing through s/t networks |
| pre_mlp_units | 128 | units for preprocessing conditioning input |
| seed | 1 | random seed |
| num_threads | 8 | number of cpu threads |
| log_interval | 1000 | how many batches to wait before logging training status |
| one_step | False | enables learning single action distributions |
| debug | False | debugging flag |
| device | "cpu" | torch device |
| deterministic | False | use deterministic actor instead of density model |
| gaussian | false | learn a gaussian prior |
| data_ratio | 1.0 | fraction of data to be used for training |

### Downstream RL
| Param         | Default           | Info  |
| ------------- | ------------- | -----:|
| exp_name | "default" | experiment name |
| env | "reach-v2" | environment name |
| epochs | 50 | number of epochs |
| seed | 0 | seed |
| num_threads | 8 | number of threads to use |
| num_test_episodes | 10 | number of test episodes |
| save_freq | 1 | number of epochs between savings |
| debug | False | debug mode (logs to /tmp) |
| gpu | False | whether to use cuda acceleration |
| steps_per_epoch | 4000 | environment steps per epoch |
| start_steps | 10000 | initial steps of random exploration |
| update_after | 1000 | number of steps to start training |
| update_every | 50 | interval between training runs |
| max_ep_len | 1000 | maximum length of an episode |
| terminate_on_success | False | whether an episode should terminate when successful |
| batch_size | 100 | batch size for optimization |
| replay_size | 1000000 | size of the experience replay |
| gamma | 0.99 | discount rate |
| polyak | 0.995 | polyak rate |
| alpha | 0.2 | inverse of reward scale |
| lr | 0.001 | learning rate for neural networks |
| hid | 256 | units per hidden layer |
| l | 2 | number of hidden layers |
| visual | False | moves to visual RL |
| fix_env_seed | False | whether to  fix goal in sawyer envs |
| neg_rew | False | sets rewards to -1/0 instead of 0/1 |
| action_prior | "none" | action prior (e.g. 'parrot_state') |
| goal_cond | True | enables goal_conditioning |
| her | True | enables hindsight experience replay |
| replay_k | 4 | hindsight replay ratio |
| prioritize | False | enables PER |
| prioritize_alpha | 0.6 | alpha for PER |
| prioritize_beta | 0.4 | beta for PER |
| prioritize_epsilon | 1e-05 | epsilon for PER |
| clip_gradients | False | gradient clipping for Q-networks |
| sil | False | enables SIL |
| sil_m | 4 | how many imitation steps for a single RL update |
| sil_bs | 100 | SIL batch size |
| sil_weight | 1.0 | weight of SIL loss |
| sil_value_weight | 0.1 | relative weight of SIL value loss |
| n_step_rew | 10 | steps for Q-value backup |
| clip_rew | True | clip multi-step reward to [0, 1] range |
| use_prior | False | enables mixing SAC policy with prior |
| prior_model | "flow" | or 'deterministic', 'lscde', 'vae' |
| prior_cond | "action" | or 'state', 'action+state', 'state+goal' |
| prior_n_step | 1 | number of steps to condition prior on |
| prior_objective | True | modifies SAC objective to maximize mixing weight |
| prior_initial | True | whether to use prior for initial exploration |
| prior_schedule | 0 | number of steps for mixing coefficient scheduling |
| data_ratio | 1.0 | amount of data used for training prior |
| one_step | False | enables unconditional action prior |
| bc_epochs | 0 | number of behavioral cloning epochs to initialize policy |
| use_polyrl | false | enables SAC-PolyRL |
| rand_init_cond | true | conditions prior with uniformly sampled variables at beginning of episode |
| lambda_param | false | whether to learn lambda as a param or NN |
| beta | 0.0 | parameter encouraging high mixing weights |
| learn_lambda | true | if false, does not train lambda |
| lambda_init | 0.95 | initialization for lambda |
| prior_clamp" | 0.0 | restricts support of prior distribution |
| prior_smoothing | 0.0 | smooths out prior by mixing with an uniform distribution |
| epsilon | 1e-9 | gradient scaling for lambda function |
| sparsity | 14.0 | sparsity parameter for mujoco experiments |
| bias | 0.1 | state bias for mujoco experiments |
| permute_action | False | shuffles action dimension in mujoco |
| kl_reg | False | whether to use soft kl regularization |


----------

# Example Commands

## SFP on window-close (entire pipeline)

```
python scripts/collect_large_dataset.py --env meta-world
python scripts/distill_dataset.py --env meta-world
python flows/main.py --env meta-world
python main.py --env window-close-v2 --use_prior True --alpha 0.01
```

## Training Flow Model for PARROT on GPU

```
python main.py --mode image --env meta-world --shared true --pre_mlp_units 256 --num_blocks 4 --device cuda --bn false --lr 1e-05 --num_hidden 256
```

## Training Flow Model for PARROT-state
```
python main.py --mode state --env meta-world --shared true --pre_mlp_units 256 --num_blocks 4 --bn false --lr 1e-05 --num_hidden 256
```

## Training SAC on door-open

```
python main.py --env door-open-v2 --use_prior False
```

## Training SAC with BC on window-open

```
python main.py --env window-open-v2 --use_prior False --bc_epochs 10
```

## Training PARROT-state on window-open

```
python main.py --env window-open-v2 --use_prior False --action_prior parrot_state
```

## Training SFP on window-open (visual)
```
python main.py --env window-open-v2 --use_prior True --alpha 0.01 --visual True --replay_size 200000 --gpu True
```


## Citation
```
@article{bagatella2022sfp,
title={{SFP}: State-free Priors for Exploration in Off-Policy Reinforcement Learning},
author={Marco Bagatella and Sammy Joe Christen and Otmar Hilliges},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=qYNfwFCX9a},
}
```
