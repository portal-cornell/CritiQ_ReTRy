from pickletools import read_decimalnl_short
import typing
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
import numpy as np
import sys
from numpy.typing import NDArray
import mujoco
import open3d as o3d
from envs.grocery_details import DIMS
from scipy.spatial.transform import Rotation
import os
from envs.stretch import *
import time
import mujoco.viewer
from envs import __file__ as base_path
from envs import stretch
from envs.stretch_utils import get_history_obs, get_obs_space


class StretchNav(BaseStretchEnv):
    st_filter_list = ["target"]
    teach_filter_list = []
    def __init__(
        self,
        max_episode_length=400,
        frame_skip=40,
        render_mode="rgb_array",
        camera_name="render_cam",
        seed=None,
        timestep=0.005,
        student=False,
        imitation_learning_training=False,
        initial_states=None
    ):
        """
        Initializes a Mujoco environment with a single fixed object on a table.

        Action Space:
            Num | Action           | Min | Max | Conversion
            0   | Base Velocity    | -1  | 1   | Scaled by 0.3
            1   | Base Angular Vel | ^   | ^   | ^             

        Observation Space: Dict
            Key            | Obs
            jnt_states     | base z axis rotation
            target         | delta from robot base to each potential goal
            target_history | student: visited potential goal, teacher: goal
        """

        self.initial_states = initial_states
        self.student = student
        self.imitation_learning_training = imitation_learning_training
        self.observation_space = get_obs_space("nav", self.student, self.imitation_learning_training)
        
        #init base env
        location = os.path.dirname(os.path.realpath(base_path))
        fname = os.path.join(location, f"scenes/stretch_nav.xml")
        action_mask = np.zeros(10, dtype=np.int32)
        # action_mask[1] = action_mask[2] = action_mask[5] = action_mask[6] = action_mask[7] = action_mask[8] = action_mask[9] = 0
        action_mask[0] = action_mask[1] = 1
        super().__init__(
            fname, # put params here if wanted
            self.observation_space,
            frame_skip,
            camera_name,
            render_mode,
            max_episode_length,
            seed=seed,
            timestep=timestep,
            action_dim=2,
            action_mask=action_mask,
        )
        
        self.body_names = [ "exterior", "room_1", "room_2", "room_3", "room_4"]
    
    def student_filter_list(self):
        return self.st_filter_list

    def reset_model(self, data = None):
        self.num_steps = 0
        self.target = np.random.randint(0, 4)
        self.target_history = np.zeros(4, dtype=np.float32)
        self.reset_tried_0 = 0
        self.reset_tried_1 = 0
        self.reset_tried_2 = 0
        self.reset_tried_3 = 0
        self.observation_history = []
        self.env_id = "nav"
        if data is not None:
            self.set_state(data)
        elif self.initial_states is not None:
            sampled_bank = self.initial_states.sample(pop=False)
            data = sampled_bank.sample(pop=False)
            self.set_state(data)
        else:
            rot = Rotation.from_euler('z', 180, degrees=True)
            x, y, z, w = rot.as_quat()

            super().reset_model(self.env_id, [0.63, 0.02, 3, 0., -np.pi / 2, -0.65, 0, 0, -4, 0, 0, w, x, y, z])

        # if not self.student:
        #     self.robot_pos = np.random.uniform(-5, 5, size=2)
        #     random_degree = np.random.uniform(0, 360)
        #     rot = Rotation.from_euler('z', random_degree, degrees=True)
        #     rot_quat = rot.as_quat()
        #     self.data.joint("base_link").qpos[3] = rot_quat[3]
        #     self.data.joint("base_link").qpos[4] = rot_quat[0]
        #     self.data.joint("base_link").qpos[5] = rot_quat[1]
        #     self.data.joint("base_link").qpos[6] = rot_quat[2]
        #     self.data.joint("base_link").qpos[:2] = self.robot_pos
        #     mujoco.mj_forward(self.model, self.data)
        #     while self.check_collision() and np.linalg.norm(self.data.body(f"room_{self.target}").xpos[:2] - self.robot_pos) > 1.5:
        #         self.robot_pos = np.random.uniform(-5, 5, size=2)
        #         self.data.joint("base_link").qpos[:2] = self.robot_pos
        #         mujoco.mj_forward(self.model, self.data)
        # else:
        #     self.robot_pos = np.array([0])
        return self._get_obs()

    def check_collision(self):
        for i, j in zip(self.data.contact.geom1, self.data.contact.geom2):
            body1 = self.data.body(self.model.body_rootid[self.model.geom_bodyid[i]]).name
            body2 = self.data.body(self.model.body_rootid[self.model.geom_bodyid[j]]).name
            if (body1=="base_link" and body2 in self.body_names) or (body2 == "base_link" and body1 in self.body_names):
                return True
        return False

    def close(self):
        super().close()

    def _get_info(self):
        return {"target": self.target, "step" : self.num_steps}

    def _get_obs(self):
        
        self.robot_pos = self.data.joint("base_link").qpos[:2]
        self.all_target_pos = [np.float32(self.data.body(f"room_{i}").xpos.copy())[:2] - self.robot_pos for i in range(4)]
        self.target_dists = list(map(np.linalg.norm, self.all_target_pos))

        for i in range(4):
            if self.target_dists[i] < 1: 
                self.target_history[i] = 1.
        l = []
        for i in range(4):
            l.append(self.target == i)
        one_hot = np.array(l, dtype=np.float32)
        
        base_quat = self.data.joint("base_link").qpos[3:].tolist()
        self.base_euler = Rotation.from_quat(base_quat[1:] + base_quat[:1]).as_euler('xyz')

        history = np.zeros(4, dtype=np.float32)
        target_tried_0, target_tried_1, target_tried_2, target_tried_3 = get_history_obs(self.observation_history, 2)
        if self.reset_tried_0 or target_tried_0 or (np.linalg.norm(self.all_target_pos[0][:2]) < 1):
            history[0] = 1.
            self.reset_tried_0 = 1
        if self.reset_tried_1 or target_tried_1 or (np.linalg.norm(self.all_target_pos[1][:2]) < 1):
            history[1] = 1.
            self.reset_tried_1 = 1
        if self.reset_tried_2 or target_tried_2 or (np.linalg.norm(self.all_target_pos[2][:2]) < 1):
            history[2] = 1.
            self.reset_tried_2 = 1
        if self.reset_tried_3 or target_tried_3 or (np.linalg.norm(self.all_target_pos[3][:2]) < 1):
            history[3] = 1.
            self.reset_tried_3 = 1
        
        if self.imitation_learning_training:
            obs = {
                "base_rot": np.array([self.base_euler[2]], dtype=np.float32),
                "target_0": np.array(self.all_target_pos[0], dtype=np.float32),
                "target_1": np.array(self.all_target_pos[1], dtype=np.float32),
                "target_2": np.array(self.all_target_pos[2], dtype=np.float32),
                "target_3": np.array(self.all_target_pos[3], dtype=np.float32),
                "target_history": history,
                "target": one_hot,
            }
        else:
            if self.student:
                obs = {
                    "base_rot": np.array([self.base_euler[2]], dtype=np.float32),
                    "target_0": np.array(self.all_target_pos[0], dtype=np.float32),
                    "target_1": np.array(self.all_target_pos[1], dtype=np.float32),
                    "target_2": np.array(self.all_target_pos[2], dtype=np.float32),
                    "target_3": np.array(self.all_target_pos[3], dtype=np.float32),
                    "target_history": history,
                }
            else:
                obs = {
                    "base_rot": np.array([self.base_euler[2]], dtype=np.float32),
                    "target_0": np.array(self.all_target_pos[0], dtype=np.float32),
                    "target_1": np.array(self.all_target_pos[1], dtype=np.float32),
                    "target_2": np.array(self.all_target_pos[2], dtype=np.float32),
                    "target_3": np.array(self.all_target_pos[3], dtype=np.float32),
                    "target": one_hot
                }

        self.observation_history.append(obs)
        return obs

    def get_state(self):
        return {
            "qpos": self.data.qpos.copy(),
            "ctrl": self.data.ctrl.copy(), 
            "target": self.target, 
            "history": self.target_history,
            "reset_tried_0": self.reset_tried_0,
            "reset_tried_1": self.reset_tried_1,
            "reset_tried_2": self.reset_tried_2,
            "reset_tried_3": self.reset_tried_3,
        }

    def set_state(self, state):
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.data.ctrl[:] = state["ctrl"]
        self.target = state["target"]
        self.target_history = state["history"]
        self.reset_tried_0 = state["reset_tried_0"]
        self.reset_tried_1 = state["reset_tried_1"]
        self.reset_tried_2 = state["reset_tried_2"]
        self.reset_tried_3 = state["reset_tried_3"]
        mujoco.mj_forward(self.model, self.data)

    def _get_success_ended(self):

        

        return int(self.target_history[self.target]) == 1, self.robot_fell

    def _get_reward(self, obs, act_info):
        """
        Rew = Actuator penalty + Table collision penalty + Item move penalty - d2goal + reach goal rew +
        (gripper open penalty if reached goal) + grasp reward + (target height penalty if grasped)
         + lift reward
        """
        col = self.check_collision()
        self.robot_fell = (np.abs(self.base_euler[0]) > 0.5) or (np.abs(self.base_euler[1]) > 0.5)
        if self.student: 
            action_rew = 0
            if act_info["act_norm"] < 0.25:
                action_rew = (act_info["act_norm"] - 1) * 0.5
    
            rew = -0.1 + action_rew
            if self.target_history[self.target]:
                rew = 10
            elif self.robot_fell: 
                rew = -50
            if col:
                rew -= 1
            return rew
        
        rew = -np.linalg.norm(self.all_target_pos[self.target]) / 10
        rew += -5*col
        self.robot_fell = (np.abs(self.base_euler[0]) > 0.5) or (np.abs(self.base_euler[1]) > 0.5)
        if self.robot_fell: return -50
        if (self.target_history[self.target]): return 100
        return rew

    def from_action_space(self, inp_action: NDArray) -> typing.Tuple[NDArray, typing.Dict[str, typing.Any]]:
        """
        Converts `action` to an actuator control and returns information about the action.

        Args:
            action: An action that is in the action space.

        Returns:
            actuator_control: The control that is copied to MjData.ctrl.
            action_info: A potentially empty dictionary with information about the action.
        """
        a = np.clip(inp_action, -1, 1)
        full_action = np.zeros(self.model.actuator_ctrlrange.shape[0], dtype=np.float32)
        full_action[self.action_indices] = a
        full_action[2:] *= 0.01
        full_action[:2] = (
            full_action[:2] * self._action_scale[:2] + self._mean_action[:2]
        )  # transform to actuator range

        action = np.copy(self.data.ctrl)
        action[2:] += full_action[2:]
        action[:2] = full_action[:2]
        
        return np.clip(action, self.low, self.high), {
            "act_norm": np.linalg.norm(a)
        }
    
    def passive_vis(self, model = None) -> None:
        import torch
        m = self.model
        d = self.data
        self._robot_id = self.model.body("base_link").id
        pcs = []
        # (d.geom("gripper_left_1"))
        with mujoco.viewer.launch_passive(m, d) as viewer:
            obs, _ = self.reset()
            start = time.time()
            cnt = 0
            while viewer.is_running() and time.time() - start < 300:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.

                obs = self._get_obs()
                print(self._get_reward(obs, None))
                print(self._get_success_ended())
                mujoco.mj_step(m,d)

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)