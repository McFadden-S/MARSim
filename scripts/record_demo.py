"""
Record a demo GIF of MARSim gameplay.

Runs a single episode with trained agents and captures Pygame frames
into an animated GIF suitable for the README.

Usage::

    python scripts/record_demo.py --friendly models/friendly_agents-final --enemy models/enemy_agents-final --output docs/demo.gif

Requires: Pillow (pip install Pillow)
"""

import argparse
import sys

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

import io
import numpy as np
import torch
import pygame

from MARSim.PPO_Policy import PPO
from MARSim.map_generator import Battlefield
from MARSim.envs import make_MARSim
from MARSim.grid_config import GridConfig, AgentType
from MARSim.a_star_policy import AStarAgent
from MARSim.utils import build_obs_tensor


def capture_frame(screen) -> Image.Image:
    """Capture the current Pygame screen as a PIL Image."""
    raw = pygame.image.tostring(screen, "RGB")
    size = screen.get_size()
    return Image.frombytes("RGB", size, raw)


def record_demo(
    friendly_ckpt: str,
    enemy_ckpt: str,
    output_path: str,
    ugv_action_skip: int = 3,
    max_frames: int = 300,
    frame_duration_ms: int = 100,
):
    bf = Battlefield()
    env = make_MARSim(GridConfig(num_agents=50, size=50, density=0.0, map=bf.map))

    r = env.grid_config.obs_radius
    obs_dim = 2 + (2 * r + 1) ** 2
    act_dim = len(env.grid_config.MOVES)

    friendly_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
    enemy_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
    friendly_policy.load(friendly_ckpt)
    enemy_policy.load(enemy_ckpt)

    agent_types = env.grid_config.agent_types
    B = env.grid_config.num_agents

    is_friendly_uav = torch.tensor([t == AgentType.FRIENDLY_UAV for t in agent_types], dtype=torch.bool)
    is_enemy_uav = torch.tensor([t == AgentType.ENEMY_UAV for t in agent_types], dtype=torch.bool)
    is_ugv = torch.tensor([t.is_ugv for t in agent_types], dtype=torch.bool)

    fr_idx = is_friendly_uav.nonzero(as_tuple=False).squeeze(-1)
    en_idx = is_enemy_uav.nonzero(as_tuple=False).squeeze(-1)
    ugv_idx = is_ugv.nonzero(as_tuple=False).squeeze(-1)

    obs = env.reset(display_graphics=True)
    ugv_agent = AStarAgent()

    terminated = [False] * B
    truncated = [False] * B
    frames = []
    steps = 0

    # Capture initial frame
    screen = env.unwrapped.graphics.screen
    if screen:
        frames.append(capture_frame(screen))

    while not (all(terminated) or all(truncated)) and steps < max_frames:
        steps += 1
        obs_tensor = build_obs_tensor(env, obs)
        joint_actions = torch.zeros(B, dtype=torch.long)

        if fr_idx.numel() > 0:
            fr_actions, _, _, _ = friendly_policy.step(obs_tensor[fr_idx])
            joint_actions[fr_idx] = fr_actions.view(-1)
        if en_idx.numel() > 0:
            en_actions, _, _, _ = enemy_policy.step(obs_tensor[en_idx])
            joint_actions[en_idx] = en_actions.view(-1)
        if ugv_idx.numel() > 0:
            for k in ugv_idx.tolist():
                if steps % ugv_action_skip == 0:
                    joint_actions[k] = int(ugv_agent.act(obs[k]))

        obs, rewards, terminated, truncated, infos = env.step(joint_actions.tolist())

        screen = env.unwrapped.graphics.screen
        if screen:
            frames.append(capture_frame(screen))

        if all(terminated) or all(truncated):
            break

    pygame.quit()

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        print(f"Saved {len(frames)} frames to {output_path}")
    else:
        print("No frames captured.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a MARSim demo GIF")
    parser.add_argument("--friendly", default="models/friendly_agents-final")
    parser.add_argument("--enemy", default="models/enemy_agents-final")
    parser.add_argument("--output", default="docs/demo.gif")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-duration", type=int, default=100, help="Frame duration in ms")
    args = parser.parse_args()

    record_demo(
        friendly_ckpt=args.friendly,
        enemy_ckpt=args.enemy,
        output_path=args.output,
        max_frames=args.max_frames,
        frame_duration_ms=args.frame_duration,
    )
