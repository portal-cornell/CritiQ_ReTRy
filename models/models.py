import torch
import torch.nn.functional as F
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import List, Union
from gymnasium import Space
from gymnasium.spaces import Dict
import numpy as np

class LSTMEncoder(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param feat_name: Key for point observations (str)
    """

    def __init__(self, observation_space: Dict):
        hidden_dim = 256
        obs_size = sum(
            observation_space[i].shape[-1] for i in observation_space
        )
        super().__init__(observation_space, hidden_dim)
        
        self.lstm = nn.LSTM(input_size=obs_size, hidden_size=hidden_dim, num_layers=3, batch_first=True)

    def forward(self, obs_seq):
        if len(obs_seq["jnt_states"].shape) == 2:
            for key in obs_seq.keys():
                obs_seq[key] = obs_seq[key].unsqueeze(0)
        x = torch.cat(list(obs_seq.values()), dim=2)
        lstm_out, _ = self.lstm(x)
        # Take the last output of LSTM
        feat = lstm_out[:, -1, :]
        return feat

    def _predict(self, obs_seq, deterministic=False):
        with torch.no_grad():
            actions = self.forward(obs_seq)
            if deterministic:
                return actions.argmax(dim=-1).numpy()
            else:
                return actions.sample().numpy()
    
    def predict(self, obs, deterministic=False):
        return self._predict(obs, deterministic)

class DepthCNNEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        features_dim = (
            64
            + observation_space["jnt_states"].shape[0]
            + observation_space["goal_pos"].shape[0]
        )
        self.height, self.width = 58, 102
        super(DepthCNNEncoder, self).__init__(observation_space, features_dim)

        # Conv Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        # Linear Layers
        self.fc1 = nn.Linear(32 * self.height * self.width, 256)
        self.fc2 = nn.Linear(256, 64)
    
    def forward(self, ob: Dict):
        x = ob["depth_image"]
        x = x.unsqueeze(1)
        jnt_states = ob["jnt_states"]
        goal_pos = ob["goal_pos"]

        # Conv Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = x.view(-1, 32 * self.height * self.width)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.cat((x, jnt_states, goal_pos), dim=-1)

class PointNetEncoder(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param feat_name: Key for point observations (str)
    """

    def __init__(self, observation_space: Dict, feat_name: str):
        self.feat_name = feat_name
        self.remove_keys = [feat_name, "goal_pos", "depth_image", "pc", "gripper_obstacles"]
        features_dim = 64 + sum(
            observation_space[i].shape[0]
            * (i not in self.remove_keys)
            for i in observation_space
        )
        super().__init__(observation_space, features_dim)
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 64, 1)
        # self.conv3 = nn.Conv1d(64, 256, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(64)
        self.mlp = nn.Sequential(nn.Linear(67, 128), nn.ReLU(), nn.Linear(128, 64))
        # self.bn3 = nn.BatchNorm1d(256)

    def forward(self, ob: Dict):
        # print(f"{ob.keys()=}")
        x = ob[self.feat_name].transpose(1, 2)  # transform to B,3,N
        # print(f"{x.shape=}")
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = torch.cat((x.view(-1, 64), ob["goal_pos"]), dim=1)
        x = self.mlp(x)
        x = (torch.cat((x, *(v for k,v in ob.items() if k not in self.remove_keys)), dim=-1))
        return x


class TeacherPolicy(nn.Module):

    def __init__(
        self,
        observation_space: Dict,
        device: torch.DeviceObjType,
        feat_name: str,
        policy_layers: List[int],
        policy_layer_names: Union[List[str], str] = "sac",
        activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.policy_layer_names = policy_layer_names
        if self.policy_layer_names == "sac" or self.policy_layer_names == "tqc":
            self.policy_layer_names = [f"actor.latent_pi.{2*i}" for i in range(len(policy_layers))] + ["actor.mu"]
        self.device = device
        self.to(device)
        self.feat_name = feat_name
        self.remove_keys = [feat_name, "goal_pos", "depth_image", "pc"]
        features_dim = 64 + sum(
            observation_space[i].shape[0]
            * (i != feat_name and i != "goal_pos" and i != "depth_image" and i != "pc")
            for i in observation_space
        )
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 64, 1)
        # self.conv3 = nn.Conv1d(64, 256, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(64)
        self.mlp = nn.Sequential(nn.Linear(67, 128), nn.ReLU(), nn.Linear(128, 64))
        l = [nn.Linear(features_dim, policy_layers[0]), activation_fn()]
        for i in range(1, len(policy_layers)):
            l.append(nn.Linear(policy_layers[i - 1], policy_layers[i]))
            l.append(activation_fn())
        l.append(nn.Linear(policy_layers[-1], 6)) # action scale
        l.append(nn.Tanh())  # squash
        self.policy_mlp = nn.Sequential(*l)

    def forward(self, ob: Dict):
        x = ob[self.feat_name].transpose(1, 2)  # transform to B,3,N
        # print(f"{x.shape=}")
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = torch.cat((x.view(-1, 64), ob["goal_pos"]), dim=1)
        x = self.mlp(x)
        x = (torch.cat((x, *(v for k,v in ob.items() if k not in self.remove_keys)), dim=-1))
        return self.policy_mlp(x)

    def load_from_sb3(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        update_state_dict = {}
        for layer in state_dict:
            if layer.startswith("actor.features_extractor"):
                new_layer = layer.split("actor.features_extractor.")[1]
                update_state_dict[new_layer] = state_dict[layer]
            elif ".".join(layer.split(".")[:-1]) in self.policy_layer_names:
                new_layer = layer.split(".")
                wb = new_layer.pop()
                new_layer = ".".join(new_layer)
                update_state_dict[f"policy_mlp.{self.policy_layer_names.index(new_layer)*2}.{wb}"] = state_dict[layer]
        self.load_state_dict(update_state_dict)
        return self

class StudentPointNet(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param feat_name: Key for point observations (str)
    """

    def __init__(self, feat_name: str):
        self.feat_name = feat_name
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 64, 1)
        # self.conv3 = nn.Conv1d(64, 256, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(256)

    def forward(self, point_cloud):
        x = point_cloud.transpose(1, 2)  # transform to B,3,N
        # print(f"{x.shape=}")
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, 64)

class CNNVisionBackbone(nn.Module):
    def __init__(self, device, height, width):
        super().__init__()
        self.height, self.width = height, width
        # Conv Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(1,8)
        self.gn2 = nn.GroupNorm(1,16)
        self.gn3 = nn.GroupNorm(1,32)

        # Linear Layers
        self.fc1 = nn.Linear(32 * self.height * self.width, 256)
        self.fc2 = nn.Linear(256, 256)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = F.relu(x)
        
        x = x.view(-1, 32 * self.height * self.width)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentPolicy(nn.Module):
    def __init__(self, device, observation_space, feat_name, policy_layers, vision_type, activation_fn=nn.ReLU):
        
        super().__init__()
        features_dim = 64 + sum(
            observation_space[i].shape[0]
            * (i != feat_name and i != "goal_pos" and i != "depth_image" and i != "pc")
            for i in observation_space
        )
        self.feat_name = feat_name
        self.remove_keys = [feat_name, "goal_pos", "depth_image", "pc"]
        self.device = device
        self.height, self.width = observation_space["depth_image"].shape[-2:]

        if vision_type == 0:
            resnet_size = 18
            norm_cfg = {"name": "group_norm", "num_groups": 1}
            self.vision_backbone = ResNet(resnet_size, norm_cfg)
            self.vision_output_dim = 1000
        elif vision_type == 1:
            self.vision_backbone = CNNVisionBackbone(device, self.height, self.width)
            self.vision_output_dim = 256
        else: 
            self.vision_backbone = StudentPointNet(feat_name="pc")
            self.vision_output_dim = 64


        self.fc3 = nn.Linear(self.vision_output_dim + 3, 128)
        self.fc4 = nn.Linear(128, 64)

        # [256, 256, 256, 256]
        l = [nn.Linear(features_dim, policy_layers[0]), activation_fn()]
        for i in range(1, len(policy_layers)):
            l.append(nn.Linear(policy_layers[i - 1], policy_layers[i]))
            l.append(activation_fn())
        l.append(nn.Linear(policy_layers[-1], 6)) # action scale
        l.append(nn.Tanh())  # squash
        self.policy_mlp = nn.Sequential(*l)

    
    def forward(self, ob: Dict):
        pc = ob["pc"]
        goal_pos = ob["goal_pos"]
        
        x = self.vision_backbone(pc)

        x = torch.cat((x, goal_pos), dim=-1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = torch.cat((x, *(v for k,v in ob.items() if k not in self.remove_keys)), dim=-1)
        x = self.policy_mlp(x)
        return x

    def load_policy_from_teacher(self, teacher_model: nn.Module):
        new_state = {k:v for k,v in teacher_model.state_dict().items() if "policy_mlp" in k}
        self.load_state_dict(new_state, strict=False)
        return self

class TeacherDrawerPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(TeacherDrawerPolicy, self).__init__()
        # [256, 256, 256, 256]
        policy_layers = [256, 256, 256, 256]
        self.device = torch.device("cuda")
        self.policy_layer_names = [f"actor.latent_pi.{2*i}" for i in range(len(policy_layers))] + ["actor.mu"]
        l = [nn.Linear(input_dim, policy_layers[0]), nn.Tanh()]
        for i in range(1, len(policy_layers)):
            l.append(nn.Linear(policy_layers[i - 1], policy_layers[i]))
            l.append(nn.Tanh())
        l.append(nn.Linear(policy_layers[-1], output_dim)) # action scale
        l.append(nn.Tanh())  # squash
        print(l)
        self.policy_mlp = nn.Sequential(*l)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # self.mu = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, ob: Dict):
        x = torch.cat((ob["jnt_states"], ob["delta_target_pos"], ob["target_displacement"]), dim=-1)
        return self.policy_mlp(x)
    
    def load_from_sb3(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        update_state_dict = {}
        for layer in state_dict:
            if ".".join(layer.split(".")[:-1]) in self.policy_layer_names:
                new_layer = layer.split(".")
                wb = new_layer.pop()
                new_layer = ".".join(new_layer)
                update_state_dict[f"policy_mlp.{self.policy_layer_names.index(new_layer)*2}.{wb}"] = state_dict[layer]
        self.load_state_dict(update_state_dict)
        return self

class StudentPushPolicy(nn.Module):
    def __init__(self):
        super(StudentPushPolicy, self).__init__()
        input_dim = 13
        hidden_dim = 256
        output_dim = 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
    
    def forward(self, ob):
        x = torch.cat((
            ob["jnt_states"], 
            ob["target_0"], 
            ob["target_1"],
            ob["target_2"],
            ob["red_box"],
            ob["red_history"],
        ), dim=-1)
        return self.mlp(x)