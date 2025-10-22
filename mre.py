from typing import Union

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import REGISTERED_SCENE_BUILDERS
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)

from rsl_code.registration import register_env
from utils.tesollo_delto import DeltoRightGenPop


@register_env("MRE-v0",
              max_episode_steps=300)
class MRE(BaseEnv):
    SUPPORTED_ROBOTS = ["delto_right_gen_pop"]
    agent: Union[DeltoRightGenPop]

    def __init__(
        self,
        *args,
        robot_uids="delto_right_gen_pop",
        gamma=0.9,
        num_envs=1024,
        max_episode_steps=300,
        scene_builder_cls: Union[str, SceneBuilder] = "ReplicaCAD",
        **kwargs,
    ):
        self.gamma = gamma
        self.cur_step = 0
        self.action_avg = None
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self.normalized_start_qpos = None

        if isinstance(scene_builder_cls, str):
            scene_builder_cls = REGISTERED_SCENE_BUILDERS[
                scene_builder_cls
            ].scene_builder_cls
        self.scene_builder: SceneBuilder = scene_builder_cls(self)
        self.build_config_idxs = [0] * num_envs
        self.init_config_idxs = [0] * num_envs

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=0,
            num_envs=num_envs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self._set_episode_rng(seed, options.get("env_idx", torch.arange(self.num_envs)))

        obs, info = super().reset(seed=seed, options=options)

        if seed is not None:
            if isinstance(seed, list):
                seed = seed[0]
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return obs, info


    def step(self, action, *args, **kwargs):
        self.cur_step += 1

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        action = action.clone()
        action = action.to(self.device)

        total_action = action
        total_action[:, 6:] += self.normalized_start_qpos[..., 6:]

        normalized_wrist_pos = torch.zeros((self.num_envs, 6), device=self.device)

        normalized_wrist_pos = (
            (normalized_wrist_pos - self.limits[:6, 0])
            / (self.limits[:6, 1] - self.limits[:6, 0])
        ) * 2 - 1
        total_action[:, :6] += normalized_wrist_pos

        weight = self.gamma ** (self.elapsed_steps + 1)
        total_action =  self.normalized_start_qpos * weight[..., None]

        total_action = torch.clamp(total_action, min=-1.0, max=1.0)

        obs, reward, terminated, truncated, info = super().step(
            total_action, *args, **kwargs
        )

        return obs, reward, terminated, truncated, info

    def _load_agent(self, options: dict):
        """Load the agent (robot) into the environment."""
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=[0, 0, 0, 0]))

        self.limits = []
        for control in self.agent.controller.controllers.values():
            self.limits.append(control._get_joint_limits())
        self.limits = np.concatenate(self.limits, axis=0)
        self.limits = torch.tensor(self.limits).to(self.device)

        dist = self.limits[6:, 1] - self.limits[6:, 0]
        zero_pos_normed = - self.limits[6:, 0] / dist
        self.tesollo_start = (zero_pos_normed - 1) * 2
        self.tesollo_start = self.tesollo_start.reshape(1, -1)

    def _load_scene(self, options: dict):
        """Load the scene."""
        # Build scene - check if scene builder uses build_configs
        if hasattr(self.scene_builder, 'build_configs') and self.scene_builder.build_configs is not None:
            self.scene_builder.build(
                self.build_config_idxs
                if self.build_config_idxs is not None
                else self.scene_builder.sample_build_config_idxs()
            )
        else:
            self.scene_builder.build()

        # Collect all objects
        self.all_objects = {}
        if hasattr(self.scene_builder, 'movable_objects'):
            for obj in self.scene_builder.movable_objects.values():
                name = obj.name.split("_", 1)[1].split("-", 1)[0] if "_" in obj.name else obj.name
                self.all_objects[name] = obj

        # Collect all articulations
        self.all_articulations = {}
        if hasattr(self.scene_builder, 'articulations'):
            for art in self.scene_builder.articulations.values():
                name = art.name.split("_", 1)[1].split("-", 1)[0] if "_" in art.name else art.name
                self.all_articulations[name] = art

    def _load_lighting(self, options: dict):
        """Load lighting for the scene."""
        if hasattr(self.scene_builder, 'builds_lighting') and self.scene_builder.builds_lighting:
            return
        self._scene.set_ambient_light([0.4, 0.4, 0.4])
        return super()._load_lighting(options)

    def _after_control_step(self):
        """Perform actions after the control step."""
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    
    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs)


    def compute_normalized_dense_reward(self, obs, action, info):
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs=obs, action=action, info=info) 

    @property
    def _default_sim_config(self):
        """Get default simulation configuration."""
        return SimConfig(
            sim_freq=120,
            control_freq=60,
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(2048, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(2048, self.num_envs) * 2,
                found_lost_pairs_capacity=45 * 10 ** 7,  # 2**27
            ),
            scene_config=SceneConfig(
                gravity=np.array([0.0, 0.0, -9.81]),
                bounce_threshold=2.0,
                solver_position_iterations=8,
                solver_velocity_iterations=0,
            ),
            default_materials_config=DefaultMaterialsConfig(
                dynamic_friction=1,
                static_friction=1,
                restitution=0,
            ),
        )

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([-1.3, -1.0, 1.3], [-2.0, -1.0, 1])
        return CameraConfig(
            "render_camera",
            room_camera_pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def _set_start_joints(self, env_idx):
        """Set the initial joint positions for the environment."""
        num_envs = env_idx.shape[0] if isinstance(env_idx, torch.Tensor) else 1
        start_qpos = torch.zeros(
            (num_envs, self.agent.action_space.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        start_qpos[:, 6:] = torch.zeros_like(self.limits[6:, 0])
        start_qpos[:, 10] = -1
        start_qpos[:, 18] = 1
        start_qpos[:, 23] = 1

        start_qpos = torch.clamp(
            start_qpos, min=self.limits[:, 0], max=self.limits[:, 1]
        )

        self.agent.reset(start_qpos)
        self.agent.robot.set_pose(sapien.Pose())

        self.normalized_start_qpos[env_idx] = (
            (start_qpos - self.limits[:, 0]) / (self.limits[:, 1] - self.limits[:, 0])
        ) * 2 - 1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize an episode for the given environment indices."""
        with torch.device(self.device):
            # Initialize action_avg on first call
            if self.action_avg is None:
                self.action_avg = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.normalized_start_qpos = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                self.action_avg[env_idx] = torch.zeros_like(
                    self.agent.robot.qpos[env_idx], dtype=torch.float32
                )

            self._set_start_joints(env_idx)

            # Reset objects to their initial poses if any exist
            if hasattr(self, 'all_objects'):
                for obj in self.all_objects.values():
                    if obj.initial_pose.raw_pose.shape[0] == 1:
                        raw_pose_tensor = obj.initial_pose.raw_pose[0]
                    else:
                        raw_pose_tensor = obj.initial_pose.raw_pose[env_idx]

                    obj.set_pose(Pose.create_from_pq(p=raw_pose_tensor[..., :3], q=raw_pose_tensor[..., 3:]))
                    obj.set_linear_velocity(torch.zeros((len(env_idx), 3), device=self.device))
                    obj.set_angular_velocity(torch.zeros((len(env_idx), 3), device=self.device))

            # Reset articulations to their initial poses if any exist
            if hasattr(self, 'all_articulations'):
                for art in self.all_articulations.values():
                    art.set_pose(art.initial_pose)
                    art.set_qpos(
                        torch.zeros((len(env_idx), art.dof[0].item()), device=self.device)
                    )
                    art.set_qvel(
                        torch.zeros((len(env_idx), art.dof[0].item()), device=self.device)
                    )
                    art.set_qf(
                        torch.zeros((len(env_idx), art.dof[0].item()), device=self.device)
                    )