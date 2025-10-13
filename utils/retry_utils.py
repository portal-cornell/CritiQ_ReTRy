# from envs import StretchMultiObjectEnv as StretchEnv, make_env
# from envs import SinglePush, make_env
# from envs import StretchNav, make_env
import mujoco
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
import sys
import torch
import os
from copy import deepcopy
from argparse import ArgumentParser
import os.path
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import TQC
# from models import StudentDrawerPolicy, TeacherPolicy, TeacherDrawerPolicy
from utils.reset_bank import ResetBank
# from envs.stretch_utils import linear_decay_sample, linear_gaussian_sample, get_obs_split
from tqdm import tqdm
from envs.stretch_utils import get_obs_split


def cap_values(image, p):
    percentile_value = np.percentile(image, p)
    capped_image = np.where(image > percentile_value, percentile_value, image)
    return capped_image

def print_obs(obs):
    for k in obs.keys():
        print(obs[k])
        print(f"{k}: {obs[k].shape}")

def plot_info_on_frame(pil_image, info, font_size=30):
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("/share/portal/hw575/vlmrm/src/vlmrm/cli/arial.ttf", font_size)
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

def save_episode_video(screens, run_name, file_name):
    moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                screens, fps=int(1/(0.005*20))
            ).write_videofile(f"videos/{run_name}/{file_name}.mp4")

# def teacher_run(sim, data, max_ep_length, model, run_name, ep_num, traj_ind):
#     # Reset the sim to the saved sim state
#     obs, info = sim.reset(data=data)
#     screens  = []
#     success = False
#     for i in range(max_ep_length):
#         teacher_ob, _ = get_obs_split(obs, 2)
#         with torch.no_grad():
#             a = model.predict(teacher_ob, deterministic=True)[0]
        
#         obs, r, done, trunc, info = sim.step(a)
#         screens+= [plot_info_on_frame(Image.fromarray(np.uint8(sim.render())), info, font_size=10)]
#         if done:
#             success = info["is_success"]
#             break
    
#     filename = str(success) + "_" + str(traj_ind) + "_" + str(ep_num)
#     save_episode_video(screens, run_name, filename)
#     return success

def run_episode(sim, model, traj_bank, max_ep_length, run_name, ep_num, is_student, reset_bank=None, save_video=True):
    if reset_bank is None:
        obs, info = sim.reset()
    else:
        sampled_bank = reset_bank.sample(pop=False)
        # sampled_index = linear_decay_sample(len(sampled_bank), 1) # sample uniform
        # sampled_index = linear_gaussian_sample(len(sampled_bank), 0.8)
        data = sampled_bank.sample(pop=False)
        # print(data.keys())
        obs, info = sim.reset()
        sim.unwrapped.set_state(data)
        obs = sim.unwrapped._get_obs()
    screens  = []
    success = False
    for i in range(max_ep_length):
        # Save the sim state
        sim_data = sim.unwrapped.get_state()
        sim_data_copy = deepcopy(sim_data)

        traj_bank.add(sim_data_copy)
        teacher_ob, student_ob = get_obs_split(obs, 1)

        # Take a step in traj
        # teacher_ob, student_ob = get_obs_split(obs, 2)
        if is_student:
            a = model.predict(student_ob, deterministic=True)[0]
        else:
            a = model.predict(teacher_ob, deterministic=True)[0]
        obs, r, done, trunc, info = sim.step(a)
        if save_video and ep_num < 10:
            screens+= [plot_info_on_frame(Image.fromarray(np.uint8(sim.render())), info, font_size=10)]
        success = info["is_success"]
        if done:
            break
    
    if save_video and ep_num < 10:
        filename = str(success) + "_" + str(ep_num)
        save_episode_video(screens, run_name, filename)

    return traj_bank, success

def create_dir(run_name):
    if os.path.exists(f"videos/{run_name}"):
        os.system(f"rm -rf videos/{run_name}")
    os.mkdir(f"videos/{run_name}")

def collect_traj(max_ep_length, model, run_name, sim, num_ep=1, is_student=False):
    """
    1) Run episode
    2) Save all states to bank
    3) If failed run binary search with teacher reset
    4) Add the state to the collective reset bank
    5) Return and save collective reset bank
    """
    create_dir(run_name)

    num_suc = 0
    bank = ResetBank()

    failed_traj_count = 0
    total_episodes = 0
    i = 0
    with tqdm(total=num_ep) as pbar:
        while i < num_ep:
            print(f"Collecting episode: {i}")
            traj_bank = ResetBank()
            traj_bank, success = run_episode(sim, model, traj_bank, max_ep_length, run_name, i, is_student)
            if is_student:
                if not success:
                    failed_traj_count += 1

                bank.extend(traj_bank)
                i += 1
                pbar.update(1)

            else:
                if success:
                    bank.extend(traj_bank)
                    i += 1
                    pbar.update(1)
                else:
                    failed_traj_count += 1

            total_episodes += 1
            len_bank = len(bank)
            print(f"Length of bank: {len_bank}")

    print(f"Failure rate: {failed_traj_count / total_episodes}")
    return bank

def rollout_from_bank(reset_bank, teacher_sim, teacher_model, max_ep_length, run_name, num_ep=1):
    """
    1) Sample reset from expert bank
    2) Run episode using sim and model
    3) Add each state in the episode to a separate bank
    4) Return the separate bank
    """
    create_dir(run_name)

    num_suc = 0
    rollout_bank = ResetBank()

    failed_traj_count = 0
    total_episodes = 0
    i = 0
    with tqdm(total=num_ep) as pbar:
        while i < num_ep:
            traj_bank = ResetBank()
            traj_bank, success = run_episode(
                teacher_sim, 
                teacher_model, 
                traj_bank, 
                max_ep_length, 
                run_name, 
                i, 
                is_student=False,
                reset_bank=reset_bank
            )
            if success:
                rollout_bank.extend(traj_bank)
                i += 1
                pbar.update(1)
            else:
                failed_traj_count += 1
            total_episodes += 1
            bank_len = len(rollout_bank)
            print(f"Length of bank: {bank_len}")
            num_suc += success

    print(f"Failure rate: {failed_traj_count/total_episodes}")
    return rollout_bank

def rollout_teacher(max_episode_steps, teacher_model, run_name, teacher_sim, num_ep):
    bank = collect_traj(
        max_episode_steps, 
        teacher_model, 
        run_name, 
        teacher_sim,  
        num_ep=num_ep,
    )

    with open("./teacher_initial.pickle", 'wb') as file:
        pickle.dump(bank, file)
    
    print("Saved expert rollout")
    return bank

def rollout_student(max_episode_steps, student_model, run_name, student_sim, num_ep, save_path):
    bank = collect_traj(
        max_episode_steps, 
        student_model, 
        run_name, 
        student_sim,  
        num_ep=num_ep,
        is_student=True,
    )

    with open(save_path, 'wb') as file:
        pickle.dump(bank, file)
    
    print("Saved student rollout")
    return bank

def rollout_teacher_from_student(reset_bank, teacher_sim, teacher_model, max_episode_steps, run_name, num_ep, save_path):
    new_bank = rollout_from_bank(
        reset_bank,
        teacher_sim,
        teacher_model,
        max_episode_steps,
        run_name=run_name,
        num_ep=num_ep,
    )
    with open(save_path, 'wb') as file:
        pickle.dump(new_bank, file)
    
    print("Saved expert rollouts from student states")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument(
        "--student_model_path", type=str
    )
    parser.add_argument(
        "--teacher_model_path", type=str
    )
    parser.add_argument("--frame_skip", type=int, default=40)

    parser.add_argument("--student_run_name", type=str)
    parser.add_argument("--teacher_run_name", type=str)
    parser.add_argument("--num_ep", type=int, default = 1)
    args = parser.parse_args()

    student_sim = StretchNav(max_episode_length=400, frame_skip=args.frame_skip, student=True)
    teacher_sim = StretchNav(max_episode_length=400, frame_skip=args.frame_skip, student=False)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    
    teacher_model = TQC.load(f"{args.teacher_model_path}")
    student_model = TQC.load(f"{args.student_model_path}")

    # bank_D = rollout_teacher(
    #     args.max_episode_steps, 
    #     teacher_model, 
    #     args.teacher_run_name, 
    #     teacher_sim, 
    #     args.num_ep,
    # )

    bank_DL = rollout_student(
        400,
        student_model, 
        args.student_run_name, 
        student_sim, 
        360,
        "./student_nav_recover_012.pickle",
    )

    with open("./student_nav_recover_012.pickle", "rb") as file:
        student_bank = pickle.load(file)

    print("Rollout from student state")
    rollout_teacher_from_student(
        student_bank, 
        teacher_sim, 
        teacher_model, 
        args.max_episode_steps, 
        args.teacher_run_name, 
        800,
        save_path="./expert_from_student_nav_recover_012.pickle",
    )

    with open("./teacher_nav_initial.pickle", "rb") as file:
        D_bank = pickle.load(file)

    with open("./expert_from_student_nav_recover_012.pickle", "rb") as file:
        DE_bank = pickle.load(file)
    
    with open("./aggregate_dataset_nav_recover_mix.pickle", "rb") as file:
        old_bank = pickle.load(file)
    print(len(D_bank))
    print(len(DE_bank))
    print(len(old_bank))
    bank = ResetBank()
    for i in range(1000):
        D_sampled_traj = old_bank.sample(pop=True)
        bank.extend(D_sampled_traj)
        
    for i in range(700):
        DE_sampled_traj = DE_bank.sample(pop=True)
        bank.extend(DE_sampled_traj)
    
    print(f"Length of aggregated dataset: {len(bank)}")
    with open("./aggregate_dataset_nav_recover_mix_3.pickle", 'wb') as file:
        pickle.dump(bank, file)


    


    