import numpy as np
from stable_baselines3.common.policies import BasePolicy
from torch import nn
import torch.nn.functional as F

def linear_decay_sample(size, decay_factor):
    """
    1 is uniform
    0 is weighted
    """
    if decay_factor > 1:
        decay_factor = 1
    if decay_factor < 0:
        decary_factor = 0
    # Create an array of weights that favors later indices
    weights = np.arange(1, size + 1)

    # Normalize weights to create a starting probability distribution
    probabilities = weights / weights.sum()

    # Create a uniform distribution
    uniform_probabilities = np.ones(size) / size

    # Linearly interpolate between the biased and uniform probabilities
    mixed_probabilities = (1 - decay_factor) * probabilities + decay_factor * uniform_probabilities

    # Normalize the mixed probabilities
    mixed_probabilities /= mixed_probabilities.sum()

    # Sample an index based on the mixed probabilities
    sampled_index = np.random.choice(size, p=mixed_probabilities)

    return sampled_index

def linear_gaussian_sample(size, bias):
    """
    1 is later in the trajectory
    -1 is earlier
    """
    if bias < -1:
        bias = -1
    if bias > 1:
        bias = 1

    mean = size // 2 + (bias * (size // 2))
    if mean > size - 1:
        mean = size - 1
    std_dev = size / 8
    indices = np.arange(size)

    weights = np.exp(-0.5 * ((indices - mean) / std_dev) ** 2)
    weights = weights / np.sum(weights)

    sampled_index = np.random.choice(size, p=weights)
    return sampled_index




def seq_obs(obs_list, sequence_len):
    """
    Takes the last sequence_len observations and concatenates them.
    """
    obs_list_len = len(obs_list)
    start_index = obs_list_len - sequence_len
    new_obs_list = obs_list[start_index:]
    key_list = obs_list[0].keys()
    final_obs = {}
    for key in key_list:
        final_obs[key] = []
    for obs in new_obs_list:
        for key in key_list:
            final_obs[key].append(obs[key])
    for f_key in final_obs.keys():
        final_obs[f_key] = np.array(final_obs[f_key])
    return final_obs

def get_history_obs(observation_history, task=0):
    """
    task:
    0: drawer
    1: push
    2: nav
    """
    if task == 0:
        return drawer_get_history_obs(observation_history)
    elif task == 1:
        return push_get_history_obs(observation_history)
    else:
        return nav_get_history_obs(observation_history)

def drawer_get_history_obs(observation_history):
    handle_0_tried, handle_1_tried, handle_2_tried = 0, 0, 0
    for obs in observation_history:
        if obs["handle_displacement_0"] > 0.07:
            handle_0_tried = 1
        if obs["handle_displacement_1"] > 0.07:
            handle_1_tried = 1
        if obs["handle_displacement_2"] > 0.07:
            handle_2_tried = 1
    return handle_0_tried, handle_1_tried, handle_2_tried

def push_get_history_obs(observation_history):
    target_done_0, target_done_1, target_done_2 = 0, 0, 0
    for obs in observation_history:
        if np.linalg.norm(obs["target_0"] - obs["red_box"]) < 0.06:
            target_done_0 = 1
        if np.linalg.norm(obs["target_1"] - obs["red_box"]) < 0.06:
            target_done_1 = 1
        if np.linalg.norm(obs["target_2"] - obs["red_box"]) < 0.06:
            target_done_2 = 1
    return target_done_0, target_done_1, target_done_2

def nav_get_history_obs(observation_history):
    target_done_0, target_done_1, target_done_2, target_done_3 = 0, 0, 0, 0
    for obs in observation_history:
        if np.linalg.norm(obs["target_0"]) < 1:
            target_done_0 = 1
        if np.linalg.norm(obs["target_1"]) < 1:
            target_done_1 = 1
        if np.linalg.norm(obs["target_2"]) < 1:
            target_done_2 = 1
        if np.linalg.norm(obs["target_3"]) < 1:
            target_done_3 = 1
    return target_done_0, target_done_1, target_done_2, target_done_3

def get_obs_split(obs, env):
    """
    env: StretchDrawer, singlepush, or StretchNav
    returns (dict): environment observation
    """
    if env == "StretchDrawer":
        return get_obs_split_drawer(obs)
    elif env == "singlepush":
        return get_obs_split_push(obs)
    elif env == "StretchNav":
        return get_obs_split_nav(obs)
    else:
        raise ValueError("Env not found")
        
def get_obs_split_nav(obs):
    teacher_obs_keys = [
        "base_rot", 
        "target_0", 
        "target_1",
        "target_2", 
        "target_3", 
        "target",
    ]
    student_obs_keys = [
        "base_rot", 
        "target_0", 
        "target_1",
        "target_2", 
        "target_3", 
        "target_history",
    ]
    teacher_obs = {key: obs[key] for key in teacher_obs_keys if key in obs}
    student_obs = {key: obs[key] for key in student_obs_keys if key in obs}
    return teacher_obs, student_obs

def get_obs_split_push(obs):
    teacher_obs_keys = [
        "jnt_states", 
        "target_0", 
        "target_1",
        "target_2", 
        "red_box",
        "red_target",
    ]
    student_obs_keys = [
        "jnt_states", 
        "target_0", 
        "target_1",
        "target_2", 
        "red_box",
        "red_history",
    ]
    teacher_obs = {key: obs[key] for key in teacher_obs_keys if key in obs}
    student_obs = {key: obs[key] for key in student_obs_keys if key in obs}
    return teacher_obs, student_obs

def get_obs_split_drawer(obs):
    teacher_obs_keys = [
        "jnt_states", 
        "delta_handle_pos_0", 
        "handle_displacement_0",
        "delta_handle_pos_1", 
        "handle_displacement_1",
        "delta_handle_pos_2", 
        "handle_displacement_2",
    ]
    student_obs_keys = [
        "jnt_states", 
        "student_handle_pos_0", 
        "handle_displacement_0",
        "handle_0_status",
        "student_handle_pos_1", 
        "handle_displacement_1",
        "handle_1_status",
        "student_handle_pos_2", 
        "handle_displacement_2",
        "handle_2_status",
    ]
    teacher_obs = {key: obs[key] for key in teacher_obs_keys if key in obs}
    student_obs = {key: obs[key] for key in student_obs_keys if key in obs}
    return teacher_obs, student_obs

class Trajectory:
    def __init__(self, seq_length):
        """
        traj (Dict)
            obs (Dict): keys of each observation has shape (traj length, feature size)
            actions (List): list of shape (traj length, action dim)
        """
        self.traj = {"obs": {}, "actions": []}
        self.size = 0 # traj_length
        self.seq_length = seq_length
        
    def add(self, obs, action):
        if len(self.traj["obs"].keys()) == 0:
            self.traj["obs"].update(obs)
        else:
            for key in self.traj["obs"].keys():
                k = np.concatenate((self.traj["obs"][key], obs[key]))
                self.traj["obs"][key] = k
            self.traj["actions"].extend(action)

        self.traj["actions"].append(action)
        self.size += 1
        return self.traj


class Buffer:
    def __init__(self):
        """
        Bank (List of Dict): Each list contains a saved reset state
            Keys:
                data stretch joint information
                handle positions
                goal drawer (0, 1, 2)
        """
        self.buffer = []
        

    def add_from_sim(self, data, goal):
        state_dict = {}
        for data_joint in self.data_joints:
            state_dict[data_joint] = data.joint(data_joint).qpos
        state_dict["goal"] = goal
        self.bank.append(state_dict)
        return self.bank
    
    def add(self, entry):
        self.bank.append(entry)
        return self.bank
    
    def extend(self, bank):
        self.bank.append(deepcopy(bank.bank))
        return self.bank

    def sample(self, ind=None, pop=True):
        if len(self.bank) == 0:
            print("Reset bank empty")
            return None
        if ind is None:
            ind = np.random.randint(0, len(self.bank))
        if pop:
            reset_state = self.bank.pop(ind)
        else:
            reset_state = self.bank[ind]
        return reset_state

    def __len__(self):
        return len(self.bank)

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self = pickle.load(file)
