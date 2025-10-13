from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from torch import from_numpy
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm.auto import tqdm
from typing import Callable
import gymnasium as gym
from typing import Any, Dict
from argparse import ArgumentParser
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from train_rl import ProgressBarCallback, ProgressBarManager
from envs import make_env
from models import LSTMEncoder
import wandb
from wandb.integration.sb3 import WandbCallback
import time
from envs import SinglePush as StretchEnv, make_env
import pickle

"""Training Policy"""

N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 4  # Number of evaluations during the training 4
NUM_CPU = 16
N_TIMESTEPS = int(5000000)  # Training budget # 900,000
EVAL_FREQ = int(N_TIMESTEPS / (NUM_CPU  * N_EVALUATIONS))
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 2000)  # 15 minutes

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.001, log=True)
    batch_size = 2**trial.suggest_int("batch_exp", 10, 13)
    net_arch = trial.suggest_categorical("net_arch", ["big", "medium", "verybig", "max"])
    act_func = trial.suggest_categorical("act_func", ["relu", "tanh"])

    # Display true values
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("tau", tau)
    trial.set_user_attr("learning_rate", learning_rate)

    if net_arch == "big":
        net_arch = {"qf": [256, 256, 256, 256, 256], "pi": [256, 256, 256, 256, 256], "activation_fn":act_func}
    elif net_arch == "medium":
        net_arch = {"qf": [256, 256, 256, 256], "pi": [256, 256, 256, 256], "activation_fn":act_func}
    elif net_arch =="max":
        net_arch = {"qf": [512, 512, 512, 512, 512], "pi": [512, 512, 512, 512, 512], "activation_fn":act_func}
    else:
        net_arch = {"qf": [256, 256, 256], "pi": [256, 256, 256], "activation_fn":act_func}

    return {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": linear_schedule(learning_rate),
        "policy_kwargs": {
            "net_arch": net_arch
        },
        "batch_size":batch_size,
        # "action_weight":action_weight
    }

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    kwargs = {
        "policy": "MultiInputPolicy",
        "env": "singlepush",
        "device": "cuda",
    }

    # 1. Sample hyperparameters and update the keyword arguments
    kwargs.update(sample_sac_params(trial))

    reset_bank_path = "/share/portal/nlc62/stretch_sim_test/aggregate_dataset_nav_recover_mix_3.pickle"
    with open(reset_bank_path, "rb") as file:
        initial_states = pickle.load(file)
    initial_states = None
    kwargs["env"] = SubprocVecEnv(
        [make_env(
            ENV_ID, 
            max_episode_length=100, 
            frame_skip=20, 
            depth_rendering=False, 
            student=True, 
            initial_states=initial_states
        ) for i in range(NUM_CPU)]
    )
    
    # Create the RL model
    model = TQC(**kwargs)

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    num_cpu = 5
    eval_envs = SubprocVecEnv([make_env(ENV_ID, max_episode_length=500, frame_skip=40, depth_rendering=False, student=True) for i in range(num_cpu)])
    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`

    eval_callback = TrialEvalCallback(eval_envs, trial, N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True, verbose=False)

    nan_encountered = False
    try:
        # Train the model
        # with ProgressBarManager(N_TIMESTEPS) as progress_callback:
        model.learn(N_TIMESTEPS, callback=[eval_callback])
        eval_episodes = 100
        env = make_env(ENV_ID, max_episode_length=500, frame_skip=40, depth_rendering=False, student=True)()
        mean_rew, std_rew = evaluate_policy(
            model,
            env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )
        save_rew_threshold = 0
        print(f"Mean rew: {mean_rew}")
        mod_string = str(mean_rew).replace('.', '_')
        save_path = f"./student_model/retry_student_push_{mod_string}"
        model.save(save_path)
        print(f"Model saved to: {save_path}")
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    parser = ArgumentParser()
    parser.add_argument("--max_diff", type=int, default=1)
    args = parser.parse_args()
    max_diff = args.max_diff
    torch.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    with open("./hyperparameters2.txt", 'w') as file:
        file.write("Best trial:\n")
        file.write(f"  Value: {trial.value}\n")
        file.write("  Params: \n")
        for key, value in trial.params.items():
            file.write(f"    {key}: {value}\n")

        file.write("  User attrs:\n")
        for key, value in trial.user_attrs.items():
            file.write(f"    {key}: {value}\n")

    # Write report
    study.trials_dataframe().to_csv("study_results_sac_stretchsim.csv")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()
    fig1.savefig("fig1.png")
    fig2.savefig("fig2.png")