"""
Shared utility functions for MARSim training and evaluation scripts.
"""

import numpy as np
import torch


def build_obs_tensor(env, obs):
    """
    Convert a list of per-agent observation dicts into a single float32 tensor.

    Each agent's observation is flattened to:
        [xy (2,), agents_patch ((2r+1)^2,)]
    giving a final shape of [num_agents, 2 + (2r+1)^2].

    Args:
        env: A MARSim environment (used to read grid_config).
        obs: List of observation dicts, one per agent.  Each dict must
             contain 'xy' (array-like, shape (2,)) and 'agents' (2D array
             of shape (2r+1, 2r+1)).

    Returns:
        torch.Tensor of shape [num_agents, obs_dim] with dtype float32.
    """
    r = env.grid_config.obs_radius
    patch_size = (2 * r + 1) ** 2
    num_agents = env.grid_config.num_agents

    obs_list = []
    for i in range(num_agents):
        xy = np.asarray(obs[i]["xy"], dtype=np.float32)             # (2,)
        patch = obs[i]["agents"].astype(np.float32).reshape(-1)     # ((2r+1)^2,)
        vec = np.concatenate([xy, patch], axis=0)
        assert vec.shape[0] == 2 + patch_size, (
            f"Agent {i}: expected obs dim {2 + patch_size}, got {vec.shape[0]}"
        )
        obs_list.append(vec)

    return torch.from_numpy(np.stack(obs_list, axis=0))  # [B, obs_dim]
