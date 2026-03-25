"""
Shared utility functions for MARSim training and evaluation scripts.
"""

import numpy as np
import torch


def build_obs_tensor(env, obs):
    """
    Convert a list of per-agent observation dicts into a single float32 tensor.

    Each agent's observation is flattened to::

        [xy (2), ugv_relative_xy (2), agents_patch ((2r+1)^2), team_agents_patch ((2r+1)^2)]

    giving a final shape of ``[num_agents, 2 + 2 + 2*(2r+1)^2]``.

    Args:
        env: A MARSim environment (used to read grid_config).
        obs: List of observation dicts, one per agent.

    Returns:
        torch.Tensor of shape [num_agents, obs_dim] with dtype float32.
    """
    r = env.grid_config.obs_radius
    patch_size = (2 * r + 1) ** 2
    num_agents = env.grid_config.num_agents
    obs_dim = 2 + 2 + 2 * patch_size

    obs_list = []
    for i in range(num_agents):
        xy = np.asarray(obs[i]["xy"], dtype=np.float32)                         # (2,)
        ugv_rel = np.asarray(obs[i]["ugv_relative_xy"], dtype=np.float32)       # (2,)
        own_agents = obs[i]["agents"].astype(np.float32).reshape(-1)            # (patch_size,)
        team_agents = obs[i]["team_agents"].astype(np.float32).reshape(-1)      # (patch_size,)
        vec = np.concatenate([xy, ugv_rel, own_agents, team_agents], axis=0)
        assert vec.shape[0] == obs_dim, (
            f"Agent {i}: expected obs dim {obs_dim}, got {vec.shape[0]}"
        )
        obs_list.append(vec)

    return torch.from_numpy(np.stack(obs_list, axis=0))  # [B, obs_dim]
