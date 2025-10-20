import typing
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
from numpy.typing import NDArray
import mujoco

JNT_NAMES = ["joint_lift", "joint_wrist_yaw", "joint_gripper_finger_left_open"]


class BaseStretchEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        # "render_fps":250
    }

    def __init__(
        self,
        model_path: typing.Union[
            str,
            typing.Callable[
                [], typing.Tuple[str, typing.Dict[str, typing.Any]]
            ],
        ],
        observation_space: Space,
        frame_skip: int,
        camera_name: str,
        render_mode: str = "rgb_array",
        max_episode_length: int = 200,
        width: int = 640,
        height: int = 480,
        seed: typing.Optional[int] = None,
        timestep: float = 0.005,
        action_dim: int = 5,
        action_mask : typing.Optional[NDArray] = None,
    ):
        """
        Base class for a Stretch RE1 environment.

        Action Space:
            Num | Action           | Min | Max | Conversion
            0   | Base Velocity    | -1  | 1   | Scaled by 0.3
            1   | Base Angular Vel | ^   | ^   | ^
            2   | Lift pos         | ^   | ^   | Scaled by 0.01 and added to current position
            3   | Arm ext          | ^   | ^   | ^
            4   | Wrist yaw        | ^   | ^   | ^
            5   | Gripper          | ^   | ^   | ^

        Args:
            model_path: Path to the MuJoCo Model or a callable that returns a path to the Mujoco Model and kwargs that are set as attributes.
            frame_skip: Number of MuJoCo simulation steps per gym `step()`.
            observation_space: The observation space of the environment.
            camera_name: The name of the camera used for rendering.
            render_mode: The `render_mode` used by default is rgb_array.
            max_episode_length: The number of env.step() before trunc is set to True. By default it is 1000.
            width: The maximum pixel width of any camera. By default it is 640.
            height: The maximum pixel height of any camera. By default it is 480.
            seed: Seed for the numpy random number generator. By default it is None.
            timestep: The time between Mujoco simulation steps. By default it is 0.005.
        """
        if seed is None:
            self.seed = int(np.random.rand() * (2**32 - 1))
        else:
            self.seed = seed
        np.random.seed(self.seed)
        self.action_mask = action_mask
        self.action_dim = int(action_mask.sum())
        if self.action_mask is None:
            self.action_mask = np.array([
                True, False, True, True, True, True, False, False, False, False
            ])
        self.num_steps = 0
        self.max_episode_length = max_episode_length
        if isinstance(model_path, typing.Callable):
            model_path, attrs = model_path()
            for name in attrs:
                setattr(self, name, attrs[name])
        self.model_path = model_path

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
            width=width,
            height=height,
        )
        self.action_indices = np.where(np.arange(1, self.model.actuator_ctrlrange.shape[0] + 1) * self.action_mask) # +1 so that it doesn't ignore 0 index
        # print(self.action_indices)
        self.model.opt.timestep = timestep  # set mujoco timestep between frames

    def _set_action_space(self) -> Space:
        """
        Sets action space to a 6 dim Box bounded from -1 to 1. Also sets self.low and self.high to the
        model's actuator bounds.
        """
        self.action_space = Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.low, self.high = (
            self.model.actuator_ctrlrange.copy().astype(np.float32).T
        )
        self._mean_action = (
            self.high + self.low
        ) / 2  # use if velocity control
        self._action_scale = (self.high - self.low) / 2
        return self.action_space

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
            full_action[:2] * self._action_scale[:2] * 0.3 + self._mean_action[:2]
        )  # transform to actuator range

        action = np.copy(self.data.ctrl)
        action[2:] += full_action[2:]
        action[:2] = full_action[:2]
        
        return np.clip(action, self.low, self.high), {
            "act_norm": np.abs(np.clip(action, self.low, self.high)[0])
        }

    def step(self, action):
        a, action_info = self.from_action_space(action)
        
        self.do_simulation(a, self.frame_skip)
        self.num_steps += 1

        obs = self._get_obs()
        r = self._get_reward(obs, action_info)
        done, ended = self._get_success_ended()
        trunc = self.num_steps >= self.max_episode_length
        info = dict({"r":r, "is_success": done}, **self._get_info()) #concat dicts
        return obs, r, done or ended, trunc, info
    
    def eval_step(
        self, action, n_step=1, get_frames=True
    ) -> typing.Tuple[
        typing.Any, float, bool, bool, typing.Dict[str, typing.Any]
    ]:
        """
        Performs the equivalent of an env.step(), but with a frame_skip of n_step. If `get_frames` is True then
        it returns n_step frames.

        Args:
            action: The action.
            n_step: The number of steps. By default it is 1.
            get_frames: Whether to render at each time step.

        Returns:
            observation: The observation from the environment.
            reward: The reward.
            done: Whether the robot reached the goal.
            trunc: Whether num_steps >= max_episode_length
            info: A dictionary with information
        """
        a, action_info = self.from_action_space(action)
        frames = []
        for _ in range(n_step):
            self.do_simulation(a, 1)
            if get_frames:
                frames.append(self.render())

        self.num_steps += 1
        obs = self._get_obs()

        r = self._get_reward(obs, action_info)
        done, ended = self._get_success_ended()
        trunc = self.num_steps >= self.max_episode_length
        info = dict({"r":r, "is_success": done}, **self._get_info()) #concat dicts

        if get_frames:
            info["frames"] = frames
        return obs, r, done, trunc, info

    def reset(
        self, seed: int = None, **kwargs
    ) -> typing.Tuple[typing.Any, typing.Dict[str, typing.Any]]:
        """
        Resets environment to the original configuration.

        Args:
            seed: The seed for numpy rng.
            kwargs: kwargs to pass to reset_model

        Returns:
            ob: Observation from the enviroment.
            info: Information form the environment.
        """
        super().reset(seed=seed)

        self._reset_simulation()

        info = self._get_reset_info()
        self.num_steps = 0
        ob = self.reset_model(**kwargs)
        if self.render_mode == "human":
            self.render()
        return ob, info

    def _initialize_simulation(self):
        """
        Initializes mujoco related parameters. 

        Returns:
            model: Mujoco MjModel
            data: Mujoco MjData
        """
        self.width = 640
        self.height = 480
        model, data = super()._initialize_simulation()

        self.model = model
        self.data = data
        return self.model, self.data

    def reset_model(self, env_id: str = None, stretch_pos : NDArray = None) -> typing.Any:
        """
        Resets robot state.
        Args:
            stretch_pos: Initial joint states by default it is None and the initial state is 
            randomly generated.

        Returns:
            ob: Observation from the enviroment.
        """

        if stretch_pos is not None:
            self._lift_pos, self._arm_ext, self._wrist_yaw, self._gripper_jnt, self._head_pan, self._head_tilt, self._wrist_pitch, self._wrist_roll, *self.base_pos = stretch_pos
            if env_id == "nav":
                box_pos = np.zeros(7)
                box_pos[6] = 1
                if self.target == 0:
                    box_pos[:2] = [-3.75, 3.75]
                elif self.target == 1:
                    box_pos[:2] = [3.25, -3.25]
                elif self.target == 2:
                    box_pos[:2] = [2.75, 2.75]
                else:
                    box_pos[:2] = [-2.75, -3.25]
                self.data.joint("red_box").qpos = box_pos
        else:    
        # self.data.mocap_pos[0:3] = self.target_pos
            self._lift_pos = 0.58 + np.random.uniform(0.05,0.15) # TODO: Make this range larger and decrease lower bound
            self._arm_ext = np.random.uniform(0.01, 0.03)

            self._head_pan = -np.pi / 2 + np.random.uniform(-0.05, 0.05)
            self._head_tilt = -0.65 + np.random.uniform(-0.1, 0.1)
            self._gripper_jnt = np.random.uniform(0.0075, 0.015)
            self._wrist_yaw = np.random.uniform(-0.03,0.03)

            stretch_x_noise = np.random.uniform(-0.02, 0.02)
            stretch_y_noise = 0.035 + np.random.uniform(0, 0.015) 
            self.base_pos = np.zeros(7)
            self.base_pos[6] = 1
            self.base_pos[0] = stretch_x_noise
            self.base_pos[1] = stretch_y_noise
            self._wrist_pitch = np.random.uniform(-0.05, 0.05)
            self._wrist_roll = np.random.uniform(-0.05, 0.05)

        self.data.joint("base_link").qpos = self.base_pos
        self.data.joint("joint_lift").qpos[0] = self._lift_pos
        self.data.joint("joint_head_pan").qpos[0] = self._head_pan
        self.data.joint("joint_head_tilt").qpos[0] = self._head_tilt

        if env_id == "drawer":
            box_pos = np.zeros(7)
            box_pos[1] = 0.81
            box_pos[6] = 1
            if self.chosen_drawer == 0:
                box_pos[2] = 0.71
            elif self.chosen_drawer == 1:
                box_pos[2] = 0.75
            else:
                box_pos[2] = 0.815
            self.data.joint("red_box").qpos = box_pos
        # self.data.joint("joint_wrist_yaw").qpos[0] = self._wrist_yaw
        # self.data.joint("joint_gripper_slide").qpos[0] = self._gripper_jnt
        # self.data.joint("joint_wrist_pitch").qpos[0] = self._wrist_pitch
        # self.data.joint("joint_wrist_roll").qpos[0] = self._wrist_roll
        # for i in range(4):
        #     self.data.joint(f"joint_arm_l{i}").qpos[0] = self._arm_ext / 4
        mujoco.mj_forward(self.model, self.data)

        action = np.array(
                [
                    0,
                    0,
                    self._lift_pos,
                    self._arm_ext,
                    self._wrist_yaw,
                    self._gripper_jnt,
                    self._head_pan,
                    self._head_tilt,
                    self._wrist_pitch,
                    self._wrist_roll
                ]
            ) 
        # override action mask for camera
        # action[6] = self._head_pan
        # action[7] = self._head_tilt

        self.do_simulation(
            action,
            100,
        )  # make sure objects are stable and joints reach pos
        
        return self._get_obs()

    def _get_obs(self) -> typing.Any:
        """
        Returns the current observation.
        Returns:
            ob: Observation from the enviroment.
        """
        raise NotImplementedError()

    def _get_reward(self, obs, act_info) -> float:
        """
        Returns the reward for the current state and action

        Args:
            obs: The current observation
            act_info: Information regarding the current action

        Returns:
            reward: The reward.
        """
        raise NotImplementedError()
    
    def _get_success_ended(self, **kwargs) -> typing.Tuple[bool, bool]:
        """
        Returns whether the robot has reached a goal state and whether the goal state was a success or not. 

        Args:
            kwargs: Arguments required for computing the state
        
        Returns: 
            success: Whether the episode was a success.
            ended: Whether the episode ended, but was not a success. 
        """
        raise NotImplementedError()

    def _get_info(self) -> typing.Dict[str, typing.Any]:
        """
        Returns info regarding the step. 

        Returns:
            info: Dictionary with info
        """
        return {}