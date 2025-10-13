from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from tqdm.auto import tqdm
from typing import Callable
import gymnasium as gym
from argparse import ArgumentParser
import yaml
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
import wandb
from wandb.integration.sb3 import WandbCallback
import time
import sys
from utils.retry_utils import rollout_teacher, rollout_student, rollout_teacher_from_student
import pickle
from envs import SinglePush as StretchEnv, make_env

def plot_info_on_frame(pil_image, info, font_size=30):
    # TODO: this is a hard-coded path
    location = os.path.dirname(os.path.realpath(__file__))
    font = ImageFont.truetype(os.path.join(location, "arial.ttf"), font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array", "TimeLimit.truncated", "jnt_states"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y + 30*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1
    return np.array(pil_image)


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        verbose=1,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._prev = 0
        self.vid_dir = time.strftime("%Y%m%d-%H%M%S")
        os.mkdir(f"videos/{self.vid_dir}")

    def _on_step(self) -> bool:
        if self.n_calls // self._render_freq > self._prev:
            screens = []

            def grab_screens(_locals, _globals) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """

                screen = self._eval_env.render()
                image = np.uint8(screen)
                pil_image = Image.fromarray(image)
                info = _locals.get('info', {})

                plot_info_on_frame(pil_image, info)

                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(np.uint8(pil_image))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                screens, fps=int(1/(0.005*20))
            ).write_videofile(f"videos/{self.vid_dir}/{self._prev}.mp4")
            self._prev += 1
        return True

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(
        self,
    ):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--frame_skip", type=int, default=20)
    parser.add_argument("--num_cpu", type=int, default=16)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--tuning_config_path",
        type=str,
        default="./config/sac_hyperparameters.yaml",
    )
    # No / at the end
    parser.add_argument(
        "--save_dir", type=str, default="/share/portal/nlc62/CritiQ_ReTRy/retry_push"
    )
    parser.add_argument(
        "--model_name", type=str, default="retry_push_student"
    )
    parser.add_argument(
        "--teacher", type=str, default="/share/portal/nlc62/stretch_sim_test/teacher_model/teacher_push.zip"
    )
    parser.add_argument(
        "--teacher_run_name", type=str, default="retry_push_teacher", 
        help="Name of folder in videos where saved videos from teacher are saved",
    )
    parser.add_argument(
        "--student_run_name", type=str, default="retry_push_student",
        help="Name of folder in videos where saved videos from student are saved",
    )
    parser.add_argument("--total_students", type=int, default=100)
    parser.add_argument("--training_timesteps", type=int, default=3000000)
    parser.add_argument("--buffer_num_traj", type=int, default=1000)
    parser.add_argument("--load_saved", type=str, default=None)
    parser.add_argument("--teacher_env_id", type=str, default="singlepush")
    parser.add_argument("--student_env_id", type=str, default="singlepush")
    # blind pick
    # /share/portal/nlc62/benchmarks/Gymnasium-Robotics/model_checkpoints/blind_pick_tune/308.4138841496391
    # "FOFixedGripper2DBlind7cmPick-v0"
    # "POFixedGripper2DBlind7cmPick-v0"
    # block
    # /share/portal/nlc62/benchmarks/Gymnasium-Robotics/teacher_model/block_20.zip
    # "HandManipulateBlockRotateZDense-v1"
    # "HandManipulateBlockRotateZ_Noisy"
    # "HandManipulateBlockRotateZ_Image"
    # pen
    # /share/portal/nlc62/benchmarks/Gymnasium-Robotics/model_checkpoints/pen_tuning/-16.582218116625402
    # "HandManipulatePenRotateDense-v1"
    # "HandManipulatePenRotate_Noisy"
    # "HandManipulatePenRotate_Image-v1"
    args = parser.parse_args()

    teacher_model = TQC.load(args.teacher)
    teacher_sim = make_env(
        args.teacher_env_id, 
        max_episode_length=args.max_episode_steps, 
        frame_skip=args.frame_skip, 
        depth_rendering=False, 
        student=False, 
        initial_states=None, 
        seed=1
    )()
    # Initialize reset buffer with teacher states
    bank_D = rollout_teacher(
        args.max_episode_steps, 
        teacher_model, 
        args.teacher_run_name, 
        teacher_sim, 
        args.buffer_num_traj,
    )
    # sys.exit()
    # with open("./teacher_initial.pickle", "rb") as file:
    #     bank_D = pickle.load(file)

    student_env = args.student_env_id
    # Student training loop
    for ep in range(args.total_students):
        print(f"Episode: {ep}")
        with open(args.tuning_config_path) as f:
            hyperparameters = yaml.safe_load(f)
        
        # Train student with latest reset buffer
        vec_env = SubprocVecEnv([
            make_env(
                student_env, 
                max_episode_length=args.max_episode_steps, 
                frame_skip=args.frame_skip, 
                depth_rendering=False, 
                student=True, 
                initial_states=bank_D,
                seed=1,
            ) for i in range(args.num_cpu)
        ])
        hyperparameters["gamma"] = 1 - hyperparameters["gamma"]
        hyperparameters["learning_rate"] = linear_schedule(
            hyperparameters["learning_rate"]
        )
        model = TQC(
            "MultiInputPolicy",
            vec_env,
            verbose=args.verbose,
            device="cuda",
            tensorboard_log="tensorboard",
            **hyperparameters
        )
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": args.training_timesteps,
            "hyperparams": hyperparameters,
        }
        run = wandb.init(
            project="sim2real",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            name=args.model_name + "_" + student_env
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            verbose=2,
        )
        # Create the RewardCallback
        env = make_env(student_env, max_episode_length=args.max_episode_steps, frame_skip=args.frame_skip, depth_rendering=False, student=True)()
        video_callback = VideoRecorderCallback(env, render_freq=5000)

        checkpoint_callback = CheckpointCallback(
                                save_freq=args.training_timesteps//(10*args.num_cpu),
                                save_path=f"model_checkpoints/{args.model_name}",
                                name_prefix=args.model_name,
                                save_replay_buffer=True,
                            )
        with ProgressBarManager(args.training_timesteps) as progress_callback:
            reset_t = True
            model.learn(
                total_timesteps=args.training_timesteps,
                callback=[progress_callback, video_callback, wandb_callback, checkpoint_callback],
                reset_num_timesteps=reset_t
            )
            save_dir = args.save_dir
            model_name = args.model_name
            save_path = f"{save_dir}/{model_name}_{ep}"
            model.policy.save(f"{save_path}.pt")
            model.save(save_path)
        
        # Rollout the student
        student_sim = make_env(
            student_env, 
            max_episode_length=args.max_episode_steps, 
            frame_skip=args.frame_skip, 
            depth_rendering=False, 
            student=True, 
            initial_states=None, 
            seed=1
        )()
        rollout_save_path = f"./student_rollout_{student_env}_{ep}.pickle"
        bank_DL = rollout_student(
            args.max_episode_steps,
            model, 
            args.student_run_name, 
            student_sim, 
            args.buffer_num_traj,
            rollout_save_path,
        )

        # Sample random states and rollout the teacher from there
        teacher_corrections_save_path = f"./teacher_corrections_{student_env}_{ep}.pickle"
        teacher_corrections = rollout_teacher_from_student(
            bank_DL, 
            teacher_sim, 
            teacher_model, 
            args.max_episode_steps, 
            args.teacher_run_name, 
            args.buffer_num_traj,
            save_path=teacher_corrections_save_path,
        )

        # Add teacher correction states to a new reset buffer
        bank_D = teacher_corrections


    