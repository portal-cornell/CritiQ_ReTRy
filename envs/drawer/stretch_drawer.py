import typing
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
import numpy as np
import sys
from numpy.typing import NDArray
from envs.drawer.gen_env_drawer import gen_arrangement
import mujoco
import open3d as o3d
from scipy.spatial.transform import Rotation
import os
from envs.stretch import *
import time 
import mujoco.viewer
from envs.stretch_utils import seq_obs, get_history_obs
from copy import deepcopy
from envs import __file__ as base_path

class StretchDrawer(BaseStretchEnv):

    def __init__(
        self,
        max_episode_length=200,
        frame_skip=20,
        render_mode="rgb_array",
        camera_name="render_cam",
        seed=None,
        timestep=0.005,
        student=False,
        initial_states=None,
    ):
        """
        Initializes a Mujoco environment with a single fixed object on a table.

        Action Space:
            Num | Action           | Min | Max | Conversion
            0   | Lift pos         | -1  | 1   | Scaled by 0.01 and added to current position
            1   | Arm ext          | ^   | ^   | ^
            2   | Wrist yaw        | ^   | ^   | ^
            3   | Gripper          | ^   | ^   | ^

        Observation Space: Dict
            Key                 | Obs
            jnt_states          | joint_lift, joint_wrist_yaw, joint_gripper_finger_left_open, sum of arm l0-l4
            delta_handle_pos    | delta xyz (meters) from gripper centre to each drawer with opt one-hot for goal drawer
            handle_displacement | scalar value for state of drawer (meters)
        """
        joint_state_shape = 4 # joint_lift, joint_yaw, joint_grip, wrist extension
        self.observation_space = Dict(
            {
                "jnt_states": Box(low=-np.inf, high=np.inf, shape=(4,)),
                "delta_handle_pos_0": Box(low=-np.inf, high=np.inf, shape=(4,)),
                "handle_displacement_0": Box(low=-np.inf, high=np.inf, shape=(1,)),
                "delta_handle_pos_1": Box(low=-np.inf, high=np.inf, shape=(4,)),
                "handle_displacement_1": Box(low=-np.inf, high=np.inf, shape=(1,)),
                "delta_handle_pos_2": Box(low=-np.inf, high=np.inf, shape=(4,)),
                "handle_displacement_2": Box(low=-np.inf, high=np.inf, shape=(1,)),
            }
        )

        self.student_obs = student
        self.student_info = {}
        self.initial_states = initial_states

        if self.student_obs:
            self.observation_space = Dict()
            self.observation_space["jnt_states"] = Box(low=-np.inf, high=np.inf, shape=(4,))
            self.observation_space["student_handle_pos_0"] = Box(low=-np.inf, high=np.inf, shape=(3,))
            self.observation_space["handle_displacement_0"] = Box(low=-np.inf, high=np.inf, shape=(1,))
            self.observation_space["handle_0_status"] = Box(low=-np.inf, high=np.inf, shape=(1,))
            self.observation_space["student_handle_pos_1"] = Box(low=-np.inf, high=np.inf, shape=(3,))
            self.observation_space["handle_displacement_1"] = Box(low=-np.inf, high=np.inf, shape=(1,))
            self.observation_space["handle_1_status"] = Box(low=-np.inf, high=np.inf, shape=(1,))
            self.observation_space["student_handle_pos_2"] = Box(low=-np.inf, high=np.inf, shape=(3,))
            self.observation_space["handle_displacement_2"] = Box(low=-np.inf, high=np.inf, shape=(1,))
            self.observation_space["handle_2_status"] = Box(low=-np.inf, high=np.inf, shape=(1,))
            # Uncomment if needed by imitation learning methods
            # self.observation_space["delta_handle_pos_0"] = Box(low=-np.inf, high=np.inf, shape=(4,))
            # self.observation_space["delta_handle_pos_1"] = Box(low=-np.inf, high=np.inf, shape=(4,))
            # self.observation_space["delta_handle_pos_2"] = Box(low=-np.inf, high=np.inf, shape=(4,))
     
        #init base env
        location = os.path.dirname(os.path.realpath(base_path))
        fname = os.path.join(location, f"scenes/scene_600_400_700_1902111393.xml")
        action_mask = np.ones(10)
        action_mask[0] = action_mask[1] = action_mask[6] = action_mask[7] = action_mask[8] = action_mask[9] = 0
        super().__init__(
            fname, # put params here if wanted
            self.observation_space,
            frame_skip,
            camera_name,
            render_mode,
            max_episode_length,
            seed=seed,
            timestep=timestep,
            action_dim=4,
            action_mask=action_mask,
        )
    
    def reset_model(self, data = None):
        self.num_steps = 0
        self.reset_opened_0 = 0
        self.reset_opened_1 = 0
        self.reset_opened_2 = 0
        self.chosen_drawer = np.random.randint(0, 3)
        self.observation_history = []
        if self.initial_states is not None:
            sampled_bank = self.initial_states.sample(pop=False)
            data = sampled_bank.sample(pop=False)
            self.set_state(data)
        else:
            super().reset_model(data)
        return self._get_obs()

    def get_state(self):
        return {
            "qpos": self.data.qpos.copy(), 
            "ctrl": self.data.ctrl.copy(), 
            "goal": self.chosen_drawer, 
            "reset_tried_0": self.reset_tried_0,
            "reset_tried_1": self.reset_tried_1,
            "reset_tried_2": self.reset_tried_2,
        }

    def set_state(self, state):
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.data.ctrl[:] = state["ctrl"]
        self.chosen_drawer = state["goal"]
        self.reset_tried_0 = state["reset_tried_0"]
        self.reset_tried_1 = state["reset_tried_1"]
        self.reset_tried_2 = state["reset_tried_2"]
        mujoco.mj_forward(self.model, self.data)

    def _get_drawers(self, chosen_drawer, one_hot=False):
        pos_0 = self.data.body("handle0").xpos.copy()
        pos_1 = self.data.body("handle1").xpos.copy()
        pos_2 = self.data.body("handle2").xpos.copy()
        if chosen_drawer == 0:
            target_pos = self.data.body("handle0").xpos.copy()
            if one_hot:
                pos_0, pos_1, pos_2 = np.ones(4), np.zeros(4), np.zeros(4)
                pos_0[:3] = self.data.body("handle0").xpos.copy()
                pos_1[:3] = self.data.body("handle1").xpos.copy()
                pos_2[:3] = self.data.body("handle2").xpos.copy()
        elif chosen_drawer == 1:
            target_pos = self.data.body("handle1").xpos.copy()
            if one_hot:
                pos_0, pos_1, pos_2 = np.zeros(4), np.ones(4), np.zeros(4)
                pos_0[:3] = self.data.body("handle0").xpos.copy()
                pos_1[:3] = self.data.body("handle1").xpos.copy()
                pos_2[:3] = self.data.body("handle2").xpos.copy()
        else:
            target_pos = self.data.body("handle2").xpos.copy()
            if one_hot:
                pos_0, pos_1, pos_2 = np.zeros(4), np.zeros(4), np.ones(4)
                pos_0[:3] = self.data.body("handle0").xpos.copy()
                pos_1[:3] = self.data.body("handle1").xpos.copy()
                pos_2[:3] = self.data.body("handle2").xpos.copy()
        
        return target_pos, pos_0, pos_1, pos_2

    def _get_drawer_displacement(self, chosen_drawer):
        if chosen_drawer == 0:
            target_displacement = self.data.joint("handle0").qpos.copy()
        elif chosen_drawer == 1:
            target_displacement = self.data.joint("handle1").qpos.copy()
        else:
            target_displacement = self.data.joint("handle2").qpos.copy()
        displacement_0 = self.data.joint("handle0").qpos.copy()
        displacement_1 = self.data.joint("handle1").qpos.copy()
        displacement_2 = self.data.joint("handle2").qpos.copy()
        return target_displacement, displacement_0, displacement_1, displacement_2

    def _get_joint_states(self):
        joint_states = []
        for i in JNT_NAMES:
            joint_states.append(self.data.joint(i).qpos[:2])
        joint_states.append(
            np.sum(
                [self.data.joint(f"joint_arm_l{i}").qpos[:2] for i in range(4)],
                keepdims=True,
            )[0]
        )
        return np.float32(np.concatenate(joint_states))

    def _get_student_info(self):
        _, pos_0, pos_1, pos_2 = self._get_drawers(self.chosen_drawer)
        _, dis_0, dis_1, dis_2 = self._get_drawer_displacement(self.chosen_drawer)
        gripper_pos = (
            self.data.body("rubber_tip_left").xpos
            + self.data.body("rubber_tip_right").xpos
        ) / 2
        obs = {}
        obs["jnt_states"] = self._get_joint_states()
        obs["delta_handle_pos_0"] = np.float32(pos_0 - gripper_pos)
        obs["handle_displacement_0"] = np.float32(dis_0)
        obs["delta_handle_pos_1"] = np.float32(pos_1 - gripper_pos)
        obs["handle_displacement_1"] = np.float32(dis_1)
        obs["delta_handle_pos_2"] = np.float32(pos_2 - gripper_pos)
        obs["handle_displacement_2"] = np.float32(dis_2)
        return obs

    def _initialize_simulation(self):

        super()._initialize_simulation()
        self._drawer_id = self.model.body("cabinet").id
        self._table_id = self.model.body("Table").id
        self._handle0_id = self.model.body("handle0").id
        self._handle1_id = self.model.body("handle1").id
        self._handle2_id = self.model.body("handle2").id
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

    def _get_obs(self):
        base_link = self.data.joint("base_link").qpos
        w, x, y, z = base_link[3:]

        qu = self.data.body("link_gripper_finger_left").xquat.tolist()
        self._gripper_frame = np.array([[0.,-1.,0.,],[0.,0.,-1.],[1.,0.,0.]]) @ Rotation.from_quat(qu[1:] + qu[:1]).as_matrix() 
        self.gripper_pos = (
            self.data.body("rubber_tip_left").xpos
            + self.data.body("rubber_tip_right").xpos
        ) / 2

        if self.student_obs:
            self.target_pos, self.handle_pos_0, self.handle_pos_1, self.handle_pos_2 = self._get_drawers(self.chosen_drawer, one_hot=False)
            t, a, b, c = self._get_drawers(self.chosen_drawer, one_hot=True)
            a[:3] -= self.gripper_pos
            b[:3] -= self.gripper_pos
            c[:3] -= self.gripper_pos
        else:
            self.target_pos, self.handle_pos_0, self.handle_pos_1, self.handle_pos_2 = self._get_drawers(self.chosen_drawer, one_hot=True)

        self.target_displacement, self.displacement_0, self.displacement_1, self.displacement_2 = self._get_drawer_displacement(self.chosen_drawer)
        joint_states = []
        for i in JNT_NAMES:
            joint_states.append(self.data.joint(i).qpos[:2])
        joint_states.append(
            np.sum(
                [self.data.joint(f"joint_arm_l{i}").qpos[:2] for i in range(4)],
                keepdims=True,
            )[0]
        )
        handle_0_tried, handle_1_tried, handle_2_tried = get_history_obs(self.observation_history)
        if self.reset_opened_0 == 1 or handle_0_tried == 1 or self.displacement_0 > 0.07:
            self.reset_opened_0 = 1
            handle_0_tried = 1
        if self.reset_opened_1 == 1 or handle_1_tried == 1 or self.displacement_1 > 0.07:
            self.reset_opened_1 = 1
            handle_1_tried = 1
        if self.reset_opened_2 == 1 or handle_2_tried == 1 or self.displacement_2 > 0.07:
            self.reset_opened_2 = 1
            handle_2_tried = 1
            
        self.handle_pos_0[:3] -= self.gripper_pos
        self.handle_pos_1[:3] -= self.gripper_pos
        self.handle_pos_2[:3] -= self.gripper_pos

        obs = {
            "jnt_states": np.float32(np.concatenate(joint_states)),
            "delta_handle_pos_0": np.float32(self.handle_pos_0),
            "handle_displacement_0": np.float32(self.displacement_0),
            "delta_handle_pos_1": np.float32(self.handle_pos_1),
            "handle_displacement_1": np.float32(self.displacement_1),
            "delta_handle_pos_2": np.float32(self.handle_pos_2),
            "handle_displacement_2": np.float32(self.displacement_2),
        }

        if self.student_obs:
            obs = {}
            obs["jnt_states"] = np.float32(np.concatenate(joint_states))
            obs["student_handle_pos_0"] = np.float32(self.handle_pos_0)
            obs["handle_displacement_0"] = np.float32(self.displacement_0)
            obs["handle_0_status"] = np.float32(np.array([handle_0_tried]))
            obs["student_handle_pos_1"] = np.float32(self.handle_pos_1)
            obs["handle_displacement_1"] = np.float32(self.displacement_1)
            obs["handle_1_status"] = np.float32(np.array([handle_1_tried]))
            obs["student_handle_pos_2"] = np.float32(self.handle_pos_2)
            obs["handle_displacement_2"] = np.float32(self.displacement_2)
            obs["handle_2_status"] = np.float32(np.array([handle_2_tried]))
            obs["delta_handle_pos_0"] = np.float32(a)
            obs["delta_handle_pos_1"] = np.float32(b)
            obs["delta_handle_pos_2"] = np.float32(c)

        self.observation_history.append(obs)
            
        return obs

    def _check_infeasible(self):
        roll_check = False
        yaw_check = False
        base_link = self.data.joint("base_link").qpos
        w, x, y, z = base_link[3:]

        yaw_z = np.arctan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        )  # yaw (z-rot)
        angular_diff = np.abs(yaw_z - np.pi)
        angular_diff = min(angular_diff, 2 * np.pi - angular_diff)
        
        if angular_diff >= np.pi / 8:
            yaw_check = True

        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        if np.abs(roll) > 0.05:
            roll_check = True
        
        return yaw_check or roll_check
        

    def _check_handle_col(self, bodies, geoms, handle_ids, left, right):
        left_col, right_col = False, False
        for handle_id in handle_ids:
            left_handle_col, right_handle_col = self._check_both_gripper_col(
                bodies,
                geoms,
                handle_id,
                left,
                right,
            )
            left_col = left_col or left_handle_col
            right_col = right_col or right_handle_col
        return left_col, right_col

    def _check_both_gripper_col(self, bodies, geoms, target, left, right):
        left_col, right_col = False, False
        for body in bodies:
            for geom in geoms:
                if body == target and geom in left:
                    left_col = True
                if body == target and geom in right:
                    right_col = True
        return left_col, right_col

    def _contact_drawer_checking(self):
        left_drawer, left_handle = False, False
        right_drawer, right_handle = False, False
        for i, j in zip(self.data.contact.geom1, self.data.contact.geom2):
            
            body1 = self.model.geom_bodyid[i]
            body2 = self.model.geom_bodyid[j]
            root_body1 = self.model.body_rootid[self.model.geom_bodyid[i]]
            root_body2 = self.model.body_rootid[self.model.geom_bodyid[j]]
            
            left_drawer_col, right_drawer_col = self._check_both_gripper_col(
                [root_body1, root_body2],
                [i, j],
                self._drawer_id,
                self._left_grip_geom_ids,
                self._right_grip_geom_ids,
            )
            left_handle_col, right_handle_col = self._check_handle_col(
                [body1, body2],
                [i, j],
                [self._handle0_id, self._handle1_id, self._handle2_id],
                self._left_grip_geom_ids,
                self._right_grip_geom_ids,
            )
            left_drawer = left_drawer or left_drawer_col
            right_drawer = right_drawer or right_drawer_col
            left_handle = left_handle or left_handle_col
            right_handle = right_handle or right_handle_col
        left = left_drawer and (not left_handle)
        right = right_drawer and (not right_handle)
        return left or right
    
    def _contact_drawer_handle(self, bodies, roots, drawer_id, handles, robot_id):
        """
        Checks if the contact is between robot and drawer but not handles
        """
        root1, root2 = roots[0], roots[1]
        body1, body2 = bodies[0], bodies[1]
        result = False
        if root1 == robot_id:
            if root2 == drawer_id and body2 not in handles:
                return True
        if root2 == robot_id:
            if root1 == drawer_id and body1 not in handles:
                return True
        return False

    def _get_contact_force(self):
        total_force = np.zeros(3)
        left_gripper_handle_col = False
        right_gripper_handle_col = False
        
        for n in range(self.data.ncon):
            contact = self.data.contact[n]
            # If root body id is robot and other is drawer but not handle
            i, j = contact.geom1, contact.geom2
            body1 = self.model.geom_bodyid[i]
            body2 = self.model.geom_bodyid[j]
            root_body1 = self.model.body_rootid[self.model.geom_bodyid[i]]
            root_body2 = self.model.body_rootid[self.model.geom_bodyid[j]]
            
            result = self._contact_drawer_handle(
                [body1, body2],
                [root_body1, root_body2],
                self._drawer_id,
                [self._handle0_id, self._handle1_id, self._handle2_id],
                self._robot_id,
            )
            if result:
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, n, contact_force)
                total_force += contact_force[:3]
        magnitude = np.sum(np.abs(total_force))
        # print(f"Total force: {total_force}, {magnitude} at step {self.num_steps}")
        return magnitude

    def _get_success_ended(self):
        infeasible = self._check_infeasible()
        success = False
        target, handle0, handle1_pos, handle2_pos = self._get_drawer_displacement(self.chosen_drawer)
        goal_drawer_opened = target[0] > 0.07

        if self.chosen_drawer == 2:
            success = goal_drawer_opened
        elif self.chosen_drawer == 1:
            success = goal_drawer_opened and (handle2_pos[0] <= 0.025)
        else:
            success = goal_drawer_opened and (handle1_pos[0] <= 0.025) and (handle2_pos[0] <= 0.025)
        return success, infeasible
    
    def _get_reward(self, obs, act_info):
        """
        Sparse reward with success and infeasibility
        """
        # Sparse reward
        reward = -0.1
        success, infeasible = self._get_success_ended()
        if infeasible:
            reward = -10
        elif success: 
            reward = 10
        return reward
    
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
                
                # print(self._init_pos)
                mujoco.mj_step(m,d)
                
                # input()
                # if time.time() - a > 5:
                #     # self.reset_scene()
                #     a = time.time()
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)