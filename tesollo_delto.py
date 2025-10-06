import sapien
import torch
import os
import numpy as np
from copy import deepcopy
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from transforms3d.euler import euler2quat


from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose

ASSET_ROOT_DIR = "assets/tesollo_delto"

MODE_CONFIGS = {
    "dex_sim": {
        "disable_self_collisions": True,
        "joint_stiffness": 2e2,
        "joint_damping": 5,
        "joint_force_limit": 5e1,
        "arm_joint_stiffness": 4e2,
        "arm_joint_damping": 1e1,
        "arm_joint_force_limit": 5e1,
    },
    "arm_sim": {
        "disable_self_collisions": True,
        "joint_stiffness": 1e2,
        "joint_damping": 1,
        "joint_force_limit": 5e1,
        "arm_joint_stiffness": 2e2,
        "arm_joint_damping": 2e1,
        "arm_joint_force_limit": 5e1,
    },
    "real": {
        "disable_self_collisions": False,
        "joint_stiffness": 2e2,
        "joint_damping": 5,
        "joint_force_limit": 5e1,
        # "arm_joint_stiffness": 4e2,
        # "arm_joint_stiffness": 1e2,
        "arm_joint_stiffness": 8e1,
        "arm_joint_damping": 1e1,
        "arm_joint_force_limit": 5e1,
        # "arm_joint_stiffness": 1e3,
        # "arm_joint_damping": 1e2,
        # "arm_joint_force_limit": 1e2,
    },
}


class DeltoBaseGenPop(BaseAgent):

    def __init__(self, *args, **kwargs):
        # # HACK: cannot pass argument to BaseAgent
        # self.mode = os.environ.get("DELTO_CONTROL_MODE")
        # if self.mode not in MODE_CONFIGS:
        #     raise ValueError(f"Invalid mode: {self.mode}, should be one of {list(MODE_CONFIGS.keys())}")

        # for param, value in MODE_CONFIGS[self.mode].items():
        #     print(f"{param}: {value}")
        #     setattr(self, param, value)

        self.hand_dof = 20

        super().__init__(*args, **kwargs)

    # def get_link_pose(self, link_name):
    #     return next((l.pose for l in self.robot.links if l.name == link_name), None)

    @property
    def base_pose(self):
        return vectorize_pose(self.base_link[0].pose, device=self.device)

    # @property
    # def palm_pose(self):
    #     return self.get_link_pose(self.palm_link)

    @property
    def tip_poses(self):
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        
        return torch.stack(tip_poses, dim=-2) 

    @property
    def finger_qpos(self):
        return torch.stack([joint.qpos for joint in self.robot.active_joints]).to(self.device)
    
    def _after_init(self):
        for link in self.robot.links:
            link.disable_gravity = True

        self.tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            self.finger_tip_links,
        )
        # self.wrist_links = sapien_utils.get_objs_by_names(
        #     self.robot.get_links(), ["wrist"]
        # )
        self.base_link = sapien_utils.get_objs_by_names(
            self.robot.get_links(), [self.floating_base_link]
        )



@register_agent()
class DeltoRightGenPop(DeltoBaseGenPop):
    uid = "delto_right_gen_pop"
    urdf_path = f"{ASSET_ROOT_DIR}/urdf/dg5f_right.urdf"
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            k: dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1 
            )
            for k in [f"rj_dg_{i}_{j}" for i in range(1, 6) for j in [1, 2, 3, 4, 'tip']]
        },
    )
    floating_base_link = "rl_dg_mount"
    palm_link = "rl_dg_palm"
    finger_tip_links = [f"rl_dg_{i}_tip" for i in range(1, 6)]

    hand_link_names = [f'rl_dg_{i}_{j}' for i in range(1, 6) for j in [1, 2, 3, 4, 'tip']]
    hand_joint_names = [f"rj_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 5)]

    trans_joint_names = [
        "dummy_x_translation_joint",
        "dummy_y_translation_joint",
        "dummy_z_translation_joint",
    ]
    rot_joint_names = [
        "dummy_x_rotation_joint",
        "dummy_y_rotation_joint",
        "dummy_z_rotation_joint",
    ]

    hand_init_height = 0.5

    @property
    def _controller_configs(self):
        trans_pd = PDJointPosControllerConfig(
            self.trans_joint_names,
            lower=[-10, -10, -10],
            upper=[10, 10, 10],
            stiffness=2000,
            damping=100,
            force_limit=1000,
            normalize_action=True,
        )
        rot_pd = PDJointPosControllerConfig(
            self.rot_joint_names,
            lower=-4 * np.pi,
            upper=4 * np.pi,
            stiffness=2000,
            damping=100,
            force_limit=1000,
            normalize_action=True,
        )
        finger_pd = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=20 * [10],
            damping=20 * [0.3],
            force_limit=20 * [10],
            normalize_action=True,
        )

        controller_dict = dict(
            pd_joint_pos=dict(trans=trans_pd, rot=rot_pd, finger=finger_pd),
        )
        return deepcopy(controller_dict)
