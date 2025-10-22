import gymnasium as gym
import numpy as np
import sapien

import torch
import torch.nn.functional as F

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from mre import MRE

from tqdm import trange
import argparse
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

from einops import repeat, rearrange

NUM_STEPS = 300

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a policy on a ManiSkill environment"
    )

    parser.add_argument(
        "--env_name",
        type=str,
        default="MRE-v0",
        help="Environment name to train on",
    )

    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
    )

    return parser.parse_args()

def main(args):
    env_kwargs = dict(
        num_envs=args.num_envs,
    )

    env: BaseEnv = gym.make(
        args.env_name,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        **env_kwargs
    )
    
    if args.record:
        env = RecordEpisode(env, 
                            "videos/", 
                            info_on_video=False, 
                            save_trajectory=False,
                            )

    action_dims = 26
    
    actions = torch.zeros((args.num_envs, NUM_STEPS, 26), device="cuda" if args.num_envs > 1 else "cpu") 

    obs, _ = env.reset(options=dict(reconfigure=True))

    for step in trange(NUM_STEPS):
        action = actions[:, step, :]
        env.step(action)

    env.close()


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
