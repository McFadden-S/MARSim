"""
MARSim training script.

Trains PPO agents for friendly and enemy UAVs in a multi-agent resupply
scenario.  UGVs use a deterministic A* pathfinding policy.  Model
checkpoints are saved periodically to the ``models/`` directory.
"""

import numpy as np
import torch
from tqdm import tqdm

from MARSim.PPO_Policy import PPO
from MARSim.map_generator import Battlefield
from MARSim.envs import make_MARSim
from MARSim.grid_config import GridConfig, AgentType
from MARSim.a_star_policy import AStarAgent
from MARSim.utils import build_obs_tensor


def main():
    # --- Environment setup ---
    bf = Battlefield()
    env = make_MARSim(
        grid_config=GridConfig(num_agents=50, size=50, density=0.0, map=bf.map)
    )

    # UGVs only act every N steps to simulate slower ground movement
    UGV_ACTION_SKIP = 3

    # --- Policy setup ---
    r = env.grid_config.obs_radius
    obs_dim = 2 + (2 * r + 1) ** 2
    act_dim = len(env.grid_config.MOVES)

    friendly_agents = PPO(observation_shape=obs_dim, action_shape=act_dim)
    enemy_agents = PPO(observation_shape=obs_dim, action_shape=act_dim)

    # --- Agent type masks (constant for the lifetime of the env) ---
    agent_types = env.grid_config.agent_types
    B = env.grid_config.num_agents

    is_friendly_uav = torch.tensor([t == AgentType.FRIENDLY_UAV for t in agent_types], dtype=torch.bool)
    is_enemy_uav    = torch.tensor([t == AgentType.ENEMY_UAV    for t in agent_types], dtype=torch.bool)
    is_ugv          = torch.tensor([t.is_ugv                    for t in agent_types], dtype=torch.bool)

    # --- Training loop ---
    NUM_UPDATES = 100
    EPISODES_PER_UPDATE = 10

    for update_idx in tqdm(range(NUM_UPDATES), desc="Training"):
        show = (update_idx == NUM_UPDATES - 1)

        # Save checkpoints every 10 updates
        if update_idx > 0 and update_idx % 10 == 0:
            friendly_agents.save(f"models/friendly_agents-{update_idx}")
            enemy_agents.save(f"models/enemy_agents-{update_idx}")

        # Rollout buffers (time-major)
        fr_obs_buf, fr_act_buf, fr_logp_buf, fr_val_buf = [], [], [], []
        fr_rew_buf, fr_done_buf = [], []
        en_obs_buf, en_act_buf, en_logp_buf, en_val_buf = [], [], [], []
        en_rew_buf, en_done_buf = [], []

        # Bootstrap tensors (overwritten each step, used for GAE)
        fr_last_obs = fr_last_done = None
        en_last_obs = en_last_done = None
        step_counter = 0

        for _ in range(EPISODES_PER_UPDATE):
            terminated = [False] * B
            truncated = [False] * B
            obs = env.reset(display_graphics=show)
            ugv_agent = AStarAgent()

            while not (all(terminated) or all(truncated)):
                step_counter += 1
                obs_tensor = build_obs_tensor(env, obs)

                fr_idx = is_friendly_uav.nonzero(as_tuple=False).squeeze(-1)
                en_idx = is_enemy_uav.nonzero(as_tuple=False).squeeze(-1)
                ugv_idx = is_ugv.nonzero(as_tuple=False).squeeze(-1)

                # --- Friendly UAV actions (PPO) ---
                fr_actions = torch.empty(0, dtype=torch.long)
                fr_logp = torch.empty(0, dtype=torch.float32)
                fr_value = torch.empty(0, dtype=torch.float32)
                if fr_idx.numel() > 0:
                    fr_actions, fr_logp, _, fr_value = friendly_agents.step(obs_tensor[fr_idx])

                # --- Enemy UAV actions (PPO) ---
                en_actions = torch.empty(0, dtype=torch.long)
                en_logp = torch.empty(0, dtype=torch.float32)
                en_value = torch.empty(0, dtype=torch.float32)
                if en_idx.numel() > 0:
                    en_actions, en_logp, _, en_value = enemy_agents.step(obs_tensor[en_idx])

                # --- UGV actions (A* planner, with action skip) ---
                ugv_actions_np = []
                if ugv_idx.numel() > 0:
                    for k in ugv_idx.tolist():
                        if step_counter % UGV_ACTION_SKIP == 0:
                            ugv_actions_np.append(int(ugv_agent.act(obs[k])))
                        else:
                            ugv_actions_np.append(0)
                ugv_actions = (
                    torch.tensor(ugv_actions_np, dtype=torch.long)
                    if ugv_actions_np
                    else torch.empty(0, dtype=torch.long)
                )

                # --- Assemble joint action vector ---
                joint_actions = torch.empty(B, dtype=torch.long)
                if fr_idx.numel() > 0: joint_actions[fr_idx] = fr_actions.view(-1)
                if en_idx.numel() > 0: joint_actions[en_idx] = en_actions.view(-1)
                if ugv_idx.numel() > 0: joint_actions[ugv_idx] = ugv_actions.view(-1)

                # --- Step environment ---
                obs_next, rewards, terminated, truncated, infos = env.step(joint_actions.tolist())
                rew_tensor = torch.tensor(rewards, dtype=torch.float32)
                done_tensor = torch.tensor(terminated, dtype=torch.bool)

                # --- Store rollout data ---
                if fr_idx.numel() > 0:
                    fr_obs_buf.append(obs_tensor[fr_idx])
                    fr_act_buf.append(fr_actions.view(-1))
                    fr_logp_buf.append(fr_logp.view(-1))
                    fr_val_buf.append(fr_value.view(-1))
                    fr_rew_buf.append(rew_tensor[fr_idx])
                    fr_done_buf.append(done_tensor[fr_idx])

                if en_idx.numel() > 0:
                    en_obs_buf.append(obs_tensor[en_idx])
                    en_act_buf.append(en_actions.view(-1))
                    en_logp_buf.append(en_logp.view(-1))
                    en_val_buf.append(en_value.view(-1))
                    en_rew_buf.append(rew_tensor[en_idx])
                    en_done_buf.append(done_tensor[en_idx])

                # --- Update bootstrap tensors ---
                next_obs_tensor = build_obs_tensor(env, obs_next)
                if fr_idx.numel() > 0:
                    fr_last_obs = next_obs_tensor[fr_idx]
                    fr_last_done = done_tensor[fr_idx]
                if en_idx.numel() > 0:
                    en_last_obs = next_obs_tensor[en_idx]
                    en_last_done = done_tensor[en_idx]

                obs = obs_next
                if all(terminated) or all(truncated):
                    break

        # --- Stack rollout buffers and run PPO updates ---
        def stack_or_none(lst):
            return torch.stack(lst) if lst else None

        fr_data = [stack_or_none(b) for b in [fr_obs_buf, fr_act_buf, fr_logp_buf, fr_val_buf, fr_rew_buf, fr_done_buf]]
        en_data = [stack_or_none(b) for b in [en_obs_buf, en_act_buf, en_logp_buf, en_val_buf, en_rew_buf, en_done_buf]]

        if fr_data[0] is not None:
            friendly_agents.update(
                obs=fr_data[0], actions=fr_data[1], logprobs=fr_data[2],
                rewards=fr_data[4], dones=fr_data[5].to(torch.float32),
                values=fr_data[3], last_observation=fr_last_obs,
                last_done=fr_last_done.to(torch.float32),
                num_updates=NUM_UPDATES, update=update_idx,
                is_nan_padding=False, num_steps=step_counter,
            )

        if en_data[0] is not None:
            enemy_agents.update(
                obs=en_data[0], actions=en_data[1], logprobs=en_data[2],
                rewards=en_data[4], dones=en_data[5].to(torch.float32),
                values=en_data[3], last_observation=en_last_obs,
                last_done=en_last_done.to(torch.float32),
                num_updates=NUM_UPDATES, update=update_idx,
                is_nan_padding=False, num_steps=step_counter,
            )

    # Save final checkpoints
    friendly_agents.save("models/friendly_agents-final")
    enemy_agents.save("models/enemy_agents-final")


if __name__ == "__main__":
    main()
