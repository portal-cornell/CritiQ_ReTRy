from envs import SinglePush as StretchEnv, make_env
import mujoco
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import open3d as o3d
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
from models.models import StudentPolicy, TeacherPolicy, StudentPushPolicy
from envs.stretch_utils import get_obs_split
from models.BCNetwork import BCNetwork
import yaml

def cap_values(image, p):
    percentile_value = np.percentile(image, p)
    # print(f"{percentile_value=}")
    capped_image = np.where(image > percentile_value, percentile_value, image)
    return capped_image

def print_obs(obs):
    goal_pos = obs["goal_pos"]
    jnt_states = obs["jnt_states"]
    pc = obs["depth_image"]
    print(f"{jnt_states=}")
    print(f"{goal_pos=}")
    # print(f"{goal_pos=}")

def plot_info_on_frame(pil_image, info, font_size=30):
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("/share/portal/hw575/vlmrm/src/vlmrm/cli/arial.ttf", font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = font_size  # Beginning of the y position of the text
    
    i = 0
    info = []
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array", "TimeLimit.truncated", "jnt_states"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y + 30*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1
    return np.array(pil_image)

def val_env(max_ep_length, frame_skip, model, run_name, sim, env_id, device, num_ep=10, reset_bank=None, record_video=True):
    if os.path.exists(f"videos/{run_name}"):
        os.system(f"rm -rf videos/{run_name}")
    os.mkdir(f"videos/{run_name}")

    mean_ep_rew = 0
    num_suc = 0
    video_freq = 1
    res = []
    grip_pos = [[],[],[]]
    save_dict = {}
    goal_count = []
    for i in range(num_ep):
        # obs, info = sim.reset()
        obs, info = sim.reset()
        # goal_count.append(sim.unwrapped.red_target)
        if reset_bank is not None:
            sampled_bank = reset_bank.sample(pop=False)
            data = sampled_bank.sample(pop=False)
            sim.unwrapped.set_state(data)
            obs = sim.unwrapped._get_obs()
        
        save_dict = {}
        # rotated_img = np.rot90(obs["depth_image"].copy(), k=1, axes=(1, 2))
        # rotated_img = rotated_img.reshape(rotated_img.shape[1], rotated_img.shape[2])
        # rotated_img = cap_values(rotated_img, 35)
        # normalized_img = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_images = [normalized_img]
        # print(obs)
        # 1/0
        screens  = []
        suc = False
        ep_reward = 0
        for j in range(max_ep_length):
            # a = model.predict(obs, deterministic=True)[0]
            
            # ob = {k: torch.tensor(v[np.newaxis, ...].copy(), dtype=torch.float32, device=device) for k, v in obs.items()}
            teacher_ob, student_ob = get_obs_split(obs, env_id) # 0: drawer, 1: push, 2: nav
            # print(student_ob.keys())
            # with torch.no_grad():
            #     a = model(student_ob).cpu().numpy()[0]
            a = model.predict(teacher_ob, deterministic=True)[0]
            # a = np.array([1, 0], dtype=np.float32)
            save_input = deepcopy(obs)
            save_input["action"] = a
            save_input = {k: v[np.newaxis, ...].copy() for k, v in save_input.items()}
            if len(save_dict.keys()) == 0:
                save_dict = save_input
            else:
                for k in save_dict.keys():
                    save_dict[k] = np.vstack((save_dict[k], save_input[k]))
            
            obs, r, done, trunc, info = sim.step(a)#sim.eval_step(a, n_step=frame_skip, get_frames=True)
            # print(done, trunc, info)
            # print("Goal pos: " + str(obs["goal_pos"]))
            # print("Joint State: " + str(obs["jnt_states"]))
            
            # rotated_img = np.rot90(obs["depth_image"].copy(), k=1, axes=(1, 2))
            # rotated_img = rotated_img.reshape(rotated_img.shape[1], rotated_img.shape[2])
            # rotated_img = cap_values(rotated_img, 25)
            # normalized_img = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # depth_images.append(normalized_img)
            # print_obs(obs)
            # print(f"Action: {a}")
            # print(info)
                
            if record_video:
                # screens+= [Image.fromarray(np.uint8(sim.render()))]
                screens+= [plot_info_on_frame(Image.fromarray(np.uint8(sim.render())), info, font_size=10)]#info["frames"]
            mean_ep_rew += r
            ep_reward += r
            res.append(r)
            if done:
                suc = info["is_success"]
                break
        
            
            # for k in range(3):
            #     grip_pos[k].append(obs["goal_pos"][k])
        # print(i)
        # with open("obs.pkl", "wb") as f:
        #     pickle.dump(save_dict, f)
            # print(j)
        # assert ep_reward < -10, ep_reward
        if record_video:
            file_name = str(suc) + "_" + str(int(ep_reward)) + "_" + str(i)
            moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                        screens, fps=int(1/(0.005*20))
                    ).write_videofile(f"videos/{run_name}/{file_name}.mp4")
        num_suc += suc
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
        # out = cv2.VideoWriter('sim_depth_images.mp4', fourcc, 5.0, (depth_images[0].shape[1], depth_images[0].shape[0]), isColor=False)

        # # Save rotated grayscale images as frames
        # for img in depth_images:
        #     # img_copy = np.uint8(img)
        #     out.write(img)

        # Release the VideoWriter and close all windows
        # out.release()

        # with open(f"./sim_inp_{i}.pickle", 'wb') as file:
        #     pickle.dump(save_dict, file)

    # sys.stdout = orig_stdout
    # f.close()
    # count_0, count_1, count_2, count_3 = 0, 0, 0, 0
    # for goal in goal_count:
    #     if goal == 0: count_0 += 1
    #     if goal == 1: count_1 += 1
    #     if goal == 2: count_2 += 1
    #     if goal == 3: count_3 += 1

    # print(f"Goal 0: {count_0}, Goal 1: {count_1}, Goal 2: {count_2}, Goal 3: {count_3}")
    print(f"mean_ep_reward: {mean_ep_rew/num_ep}")
    print(f"mean_suc: {num_suc/num_ep}")

    fig, ax = plt.subplots(2,2)
    l=['x', 'y', 'z']
    for i in  range(3):
        curr_ax = ax[i//2, i%2]
        ax[i//2, i%2].plot(range(len(grip_pos[i])), np.abs(grip_pos[i]))
        curr_ax.set_title(f"Delta to goal pos in {l[i]} dir")
        curr_ax.set(xlabel="Frames", ylabel="Delta")

    ax[1,1].plot(range(len(res)), -np.array(res))
    ax[1,1].set(xlabel="Frames", ylabel="l2 norm to goal")
    ax[1,1].set_title("l2 norm to goal pos")
    fig.suptitle("Simulation trajectory")
    fig.tight_layout()
    plt.savefig("traj.png")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="Evaluation Configuration File (YAML).")
    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as file:
            config_data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: {args.config_path} not found")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")

    # Load reset bank
    reset_bank_path = config_data["reset_bank_path"]
    reset_bank = None
    if len(reset_bank_path) > 0:
        with open(reset_bank_path, "rb") as file:
            reset_bank = pickle.load(file)

    # Load the sim env
    sim = make_env(
        config_data["env_id"], 
        max_episode_length=config_data["max_episode_steps"], 
        frame_skip=config_data["frame_skip"],
        student=config_data["student"], 
        initial_states=reset_bank, 
        seed=config_data["seed"],
    )()
    
    # Load model
    device = torch.device(config_data["device"])
    model = None
    if config_data["model_training"] == "rl":
        model = TQC.load(config_data["model_path"])
        
        # model = BCNetwork(input_dim=13, output_dim=3)
        # model = StudentPushPolicy()
        # model.load_state_dict(torch.load(args.model_path))
        # model = model.to(device)
        # model.eval()
    print(f"{model.observation_space.spaces=}")

    # teacher = TeacherPolicy(
    #     sim.observation_space, device, "gripper_obstacles", [256, 256, 256, 256]#, activation_fn=torch.nn.Tanh
    # )
    # teacher.load_from_sb3(args.model_path)
    # teacher = teacher.to(device)
    # teacher.eval()

    # student = StudentPolicy(device, sim.observation_space, "gripper_obstacles", [256, 256, 256, 256], vision_type=2)
    # student.load_state_dict(torch.load(args.model_path))
    # student = student.to(device)
    # student.eval()
    
    # with open("sim_inp_0.pickle", 'rb') as file:
    #     data = pickle.load(file)
    #     print(data["action"].shape)
    #     print(data["depth_image"].shape)
    #     print(data["pc"].shape)
    # sys.exit()
    reset_bank=None
    # with open("/share/portal/nlc62/stretch_sim_test/expert_push_initial.pickle", "rb") as file:
    #     reset_bank = pickle.load(file)
    val_env(
        config_data["max_episode_steps"], 
        config_data["frame_skip"], 
        model, 
        config_data["run_name"], 
        sim, 
        config_data["env_id"],
        device, 
        config_data["num_ep"], 
        reset_bank=reset_bank, 
        record_video=config_data["record_video"],
    )
    sim.close()