from pickletools import read_decimalnl_short
import typing
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
import numpy as np
import sys
from numpy.typing import NDArray
import mujoco
import open3d as o3d
from scipy.spatial.transform import Rotation
import os
from envs.stretch import *
import time 
import mujoco.viewer
from envs import __file__ as base_path
from envs import stretch
from envs.stretch_utils import get_history_obs
import pickle

class SinglePush(BaseStretchEnv):
    st_filter_list = ["red_target"]
    def __init__(
        self,
        max_episode_length=200,
        frame_skip=20,
        render_mode="rgb_array",
        camera_name="render_cam",
        seed=None,
        timestep=0.005,
        student=False, 
        initial_states=None
    ):
        """
        Initializes a Mujoco environment with a single fixed object on a table.

        Action Space:
            Num | Action           | Min | Max | Conversion
            0   | Arm ext          | -1  | 1   | Scaled by 0.01 and added to current position
            1   | Wrist yaw        | ^   | ^   | ^

        Observation Space: Dict
            Key           | Obs
            jnt_states    | joint_lift, joint_wrist_yaw, joint_gripper_finger_left_open, sum of arm l0-l4
            goal_pos      | delta from gripper centre to target position
            gripper_table | delta from gripper centre to table centre
        """
        self.student = student
        self.initial_states = initial_states
        self.joints = ["joint_wrist_yaw"]
        joint_state_shape = len(self.joints)+1 #  joint_yaw, joint_grip, wrist extension
        self.observation_space = Dict(
            {
                "jnt_states": Box(low=-np.inf, high=np.inf, shape=(joint_state_shape,)),
                "red_box": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "target_0": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "target_1": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "target_2": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "red_target": Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )

        if self.student:
            self.observation_space = Dict(
                {
                    "jnt_states": Box(low=-np.inf, high=np.inf, shape=(joint_state_shape,)),
                    "red_box": Box(low=-np.inf, high=np.inf, shape=(2,)),
                    "target_0": Box(low=-np.inf, high=np.inf, shape=(2,)),
                    "target_1": Box(low=-np.inf, high=np.inf, shape=(2,)),
                    "target_2": Box(low=-np.inf, high=np.inf, shape=(2,)),
                    "red_history": Box(low=-np.inf, high=np.inf, shape=(3,)),
                    # Uncomment if needed by imitation learning method
                    # "red_target": Box(low=-np.inf, high=np.inf, shape=(3,)), 
                }
            )
            

        #init base env
        location = os.path.dirname(os.path.realpath(base_path))
        fname = os.path.join(location, f"scenes/stretch_single.xml")
        action_mask = np.ones(10, dtype=np.int32)
        action_mask[1] = action_mask[2] = action_mask[5] = action_mask[6] = action_mask[7] = action_mask[8] = action_mask[9] = 0
        super().__init__(
            fname, # put params here if wanted
            self.observation_space,
            frame_skip,
            camera_name,
            render_mode,
            max_episode_length,
            seed=seed,
            timestep=timestep,
            action_dim=5,
            action_mask=action_mask,
        )
        
    def student_filter_list(self):
        return SinglePush.st_filter_list
    
    def reset_model(self, data = None):
        self.num_steps = 0
        self.red_target = np.random.randint(0, 3)
        self.box_fallen = False
        self.red_on_target = False
        self.observation_history = []
        self.reset_tried_0 = 0
        self.reset_tried_1 = 0
        self.reset_tried_2 = 0
        
        if self.initial_states is not None:
            sampled_bank = self.initial_states.sample(pop=False)
            data = sampled_bank.sample(pop=False)
            self.set_state(data)
        else:
            super().reset_model([0.64, 0.1, 0., 0., -np.pi / 2, -0.65, 0, 0, 0, 0.19, 0, 0, 0, 0, 1])

        for i in range(3):
            self.model.geom(f"target_{i}").rgba[:] = [0, self.red_target==i, self.red_target!=i, 1]
        on_target = 0
        
        if on_target and not self.student:
            places_x = [self.data.geom(f"target_{i}").xpos[0] for i in range(3)]
            places_y = [self.data.geom(f"target_{i}").xpos[1] for i in range(3)]
            places_x.pop(self.red_target)
            places_y.pop(self.red_target)
            choice_index = np.random.randint(0, len(places_x))
            self.data.joint("red_box").qpos[1] = places_y[choice_index]
            self.data.joint("red_box").qpos[0] = places_x[choice_index]
            
        return self._get_obs()

    def _get_joint_states(self):
        joint_states = []
        for i in self.joints:
            joint_states.append(self.data.joint(i).qpos[:2])
        joint_states.append(
            np.sum(
                [self.data.joint(f"joint_arm_l{i}").qpos[:2] for i in range(4)],
                keepdims=True,
            )[0]
        )
        return np.float32(np.concatenate(joint_states))

    def _initialize_simulation(self):

        super()._initialize_simulation()
        self._table_id = self.model.body("Table").id
        self._robot_id = self.model.body("base_link").id
        self._left_grip_geom_ids = [
            self.model.geom("gripper_left_0").id,
            self.model.geom("gripper_left_1").id,
            self.model.geom("rubber_tip_left").id,
        ]
        self._right_grip_geom_ids = [
            self.model.geom("gripper_right_0").id,
            self.model.geom("gripper_right_1").id,
            self.model.geom("rubber_tip_right").id,
        ]
    
        return self.model, self.data

    def close(self):
        super().close()

    def _get_info(self):
        return {"red on target":self.red_on_target}

    def _get_obs(self):

        qu = self.data.body("link_gripper_finger_left").xquat.tolist()
        self._gripper_frame = np.array([[0.,-1.,0.,],[0.,0.,-1.],[1.,0.,0.]]) @ Rotation.from_quat(qu[1:] + qu[:1]).as_matrix() 
        self.gripper_pos = np.float32(
            self.data.body("link_grasp_center").xpos
        ) 

        target_0 = np.float32(self.data.geom(f"target_0").xpos.copy()) - self.gripper_pos
        target_1 = np.float32(self.data.geom(f"target_1").xpos.copy()) - self.gripper_pos
        target_2 = np.float32(self.data.geom(f"target_2").xpos.copy()) - self.gripper_pos

        self.red_target_pos = np.float32(self.data.geom(f"target_{self.red_target}").xpos.copy()) - self.gripper_pos
        self.red_box = np.float32(self.data.body("red_box").xpos.copy()) - self.gripper_pos
        l = []
        for i in range(3):
            l.append(self.red_target == i)
        one_hot = np.array(l, dtype=np.float32)
        history = np.zeros(3, dtype=np.float32)
        target_tried_0, target_tried_1, target_tried_2 = get_history_obs(self.observation_history, 1)
        if self.reset_tried_0 or target_tried_0 or (np.linalg.norm(target_0[:2] - self.red_box[:2]) < 0.06):
            history[0] = 1.
            self.reset_tried_0 = 1
        if self.reset_tried_1 or target_tried_1 or (np.linalg.norm(target_1[:2] - self.red_box[:2]) < 0.06):
            history[1] = 1.
            self.reset_tried_1 = 1
        if self.reset_tried_2 or target_tried_2 or (np.linalg.norm(target_2[:2] - self.red_box[:2]) < 0.06):
            history[2] = 1.
            self.reset_tried_2 = 1

        obs = {
            "jnt_states": self._get_joint_states(),
            "target_0" : target_0[:2],
            "target_1" : target_1[:2],
            "target_2" : target_2[:2],
            "red_box" : self.red_box[:2],
            "red_target": one_hot,
            # "green_target": one_hot if self.red_target != 0 else np.logical_not(one_hot).astype(np.float32)
        }
        if self.student:
            obs = {
                "jnt_states": self._get_joint_states(),
                "target_0" : target_0[:2],
                "target_1" : target_1[:2],
                "target_2" : target_2[:2],
                "red_box" : self.red_box[:2],
                "red_history": history,
                # "red_target": one_hot,
            }
        self.observation_history.append(obs)
        return obs

    def get_state(self):
        return {
            "qpos": self.data.qpos.copy(), 
            "ctrl": self.data.ctrl.copy(), 
            "goal": self.red_target, 
            "reset_tried_0": self.reset_tried_0,
            "reset_tried_1": self.reset_tried_1,
            "reset_tried_2": self.reset_tried_2,
        }
    
    def set_state(self, state):
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.data.ctrl[:] = state["ctrl"]
        self.red_target = state["goal"]
        self.reset_tried_0 = state["reset_tried_0"]
        self.reset_tried_1 = state["reset_tried_1"]
        self.reset_tried_2 = state["reset_tried_2"]
        mujoco.mj_forward(self.model, self.data)

    def _check_both_gripper_col(self, bodies, geoms, target, left, right):
        left_col, right_col = False, False
        for body in bodies:
            for geom in geoms:
                if body == target and geom in left:
                    left_col = True
                if body == target and geom in right:
                    right_col = True
        return left_col, right_col

    def _get_success_ended(self):
        
        self.both_correct =  (self.red_dist < 0.06)# and (self.green_dist < 0.06)

        return self.both_correct , self.box_fallen
   
    def _get_reward(self, obs, act_info):
        """
        Rew = Actuator penalty + Table collision penalty + Item move penalty - d2goal + reach goal rew + 
        (gripper open penalty if reached goal) + grasp reward + (target height penalty if grasped)
         + lift reward
        """
        self.red_dist = np.linalg.norm((self.red_box - self.red_target_pos)[:2] )
        if self.student: 
            self.box_fallen = np.abs(self.red_box[-1]) > 0.2
            if self.box_fallen: return -20
            elif self.red_dist < 0.06: return 20
            return -0.1
        
        # Move red to target position
        self.box_fallen = np.abs(self.red_box[-1]) > 0.2 # or np.abs(self.green_box[-1]) > 0.2
        if self.box_fallen: return -200 # box fell off
        rew = 0

        rew += -np.linalg.norm(self.red_box) - 3 * max(self.red_dist - 0.06, 0) #- self.red_dist
        red_on_target = self.red_dist < 0.06

        rew += 200*(not self.red_on_target)*(red_on_target)
        self.red_on_target |= red_on_target
        return rew
    
    def passive_vis(self, model = None) -> None:
        import torch
        m = self.model
        d = self.data
        self._robot_id = self.model.body("base_link").id
        pcs = []
   
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
                mujoco.mj_step(m,d)
                
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

