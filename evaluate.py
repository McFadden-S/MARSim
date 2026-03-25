"""
MARSim evaluation framework.

Loads trained PPO checkpoints and runs evaluation episodes, collecting
per-episode metrics and printing a summary table.  Optionally renders
the last K episodes for visual inspection.

Usage::

    python evaluate.py                                        # defaults
    python evaluate.py --episodes 50 --render 5               # custom
    python evaluate.py --friendly models/friendly_agents-90 \\
                       --enemy    models/enemy_agents-90
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from MARSim.PPO_Policy import PPO
from MARSim.map_generator import Battlefield
from MARSim.envs import make_MARSim
from MARSim.grid_config import GridConfig, AgentType
from MARSim.a_star_policy import AStarAgent
from MARSim.utils import build_obs_tensor


def evaluate(
    num_episodes: int = 20,
    render_last_k: int = 0,
    ugv_action_skip: int = 3,
    friendly_ckpt: str = "models/friendly_agents-final",
    enemy_ckpt: str = "models/enemy_agents-final",
    save_metrics: str = None,
):
    """
    Run evaluation episodes and collect metrics.

    Args:
        num_episodes: Number of evaluation episodes to run.
        render_last_k: Render the last K episodes with Pygame.
        ugv_action_skip: UGV acts every N steps (simulates slower speed).
        friendly_ckpt: Path to the friendly PPO checkpoint (without .pth).
        enemy_ckpt: Path to the enemy PPO checkpoint (without .pth).
        save_metrics: If set, save per-episode metrics to this JSON file.

    Returns:
        Dictionary of aggregated evaluation metrics.
    """
    # --- Build environment ---
    bf = Battlefield()
    env = make_MARSim(
        grid_config=GridConfig(num_agents=50, size=50, density=0.0, map=bf.map)
    )

    r = env.grid_config.obs_radius
    obs_dim = 2 + 2 + 2 * (2 * r + 1) ** 2
    act_dim = len(env.grid_config.MOVES)

    # --- Load trained policies ---
    friendly_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
    enemy_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
    friendly_policy.load(friendly_ckpt)
    enemy_policy.load(enemy_ckpt)

    # --- Agent type masks ---
    agent_types = env.grid_config.agent_types
    B = env.grid_config.num_agents

    is_friendly_uav = torch.tensor([t == AgentType.FRIENDLY_UAV for t in agent_types], dtype=torch.bool)
    is_enemy_uav    = torch.tensor([t == AgentType.ENEMY_UAV    for t in agent_types], dtype=torch.bool)
    is_ugv          = torch.tensor([t.is_ugv                    for t in agent_types], dtype=torch.bool)

    fr_idx = is_friendly_uav.nonzero(as_tuple=False).squeeze(-1)
    en_idx = is_enemy_uav.nonzero(as_tuple=False).squeeze(-1)
    ugv_idx = is_ugv.nonzero(as_tuple=False).squeeze(-1)

    # --- Evaluation metrics ---
    metrics = {
        "friendly_reward": [],
        "enemy_reward": [],
        "steps": [],
        "friendly_alive": [],
        "enemy_alive": [],
        "ugv_reached_goal": [],
        "ugv_destroyed": [],
    }

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        show = ep >= num_episodes - render_last_k

        try:
            obs = env.reset(display_graphics=show)
        except TypeError:
            obs, _ = env.reset()

        ugv_agent = AStarAgent()
        terminated = [False] * B
        truncated = [False] * B

        team_rew_friendly = 0.0
        team_rew_enemy = 0.0
        steps = 0
        ugv_reached_goal = False
        ugv_destroyed = False

        while not (all(terminated) or all(truncated)):
            steps += 1
            obs_tensor = build_obs_tensor(env, obs)

            # --- Get actions from each policy ---
            joint_actions = torch.zeros(B, dtype=torch.long)

            if fr_idx.numel() > 0:
                fr_actions, _, _, _ = friendly_policy.step(obs_tensor[fr_idx])
                joint_actions[fr_idx] = fr_actions.view(-1)

            if en_idx.numel() > 0:
                en_actions, _, _, _ = enemy_policy.step(obs_tensor[en_idx])
                joint_actions[en_idx] = en_actions.view(-1)

            # --- Feed drone obstacle data to UGV A* agent ---
            drone_data = env.unwrapped.get_drone_obstacle_data()
            ugv_agent.update_from_drones(drone_data)

            if ugv_idx.numel() > 0:
                for k in ugv_idx.tolist():
                    if steps % ugv_action_skip == 0:
                        joint_actions[k] = int(ugv_agent.act(obs[k]))
                    else:
                        joint_actions[k] = 0

            # --- Step environment ---
            obs, rewards, terminated, truncated, infos = env.step(joint_actions.tolist())

            rew_tensor = torch.tensor(rewards, dtype=torch.float32)
            if fr_idx.numel() > 0:
                team_rew_friendly += rew_tensor[fr_idx].sum().item()
            if en_idx.numel() > 0:
                team_rew_enemy += rew_tensor[en_idx].sum().item()

            if all(terminated) or all(truncated):
                break

        # --- Episode-end stats ---
        alive = torch.tensor([infos[i].get("is_active", True) for i in range(B)], dtype=torch.bool)
        friendly_alive = int(alive[fr_idx].sum().item()) if fr_idx.numel() > 0 else 0
        enemy_alive = int(alive[en_idx].sum().item()) if en_idx.numel() > 0 else 0

        # Check UGV outcome
        for k in ugv_idx.tolist():
            if not infos[k].get("is_active", True):
                # UGV was deactivated — check if it reached goal
                if env.unwrapped.was_on_goal and env.unwrapped.was_on_goal[k]:
                    ugv_reached_goal = True
                else:
                    ugv_destroyed = True

        metrics["friendly_reward"].append(team_rew_friendly)
        metrics["enemy_reward"].append(team_rew_enemy)
        metrics["steps"].append(steps)
        metrics["friendly_alive"].append(friendly_alive)
        metrics["enemy_alive"].append(enemy_alive)
        metrics["ugv_reached_goal"].append(ugv_reached_goal)
        metrics["ugv_destroyed"].append(ugv_destroyed)

        print(
            f"  Ep {ep+1:3d}/{num_episodes} | steps={steps:4d} | "
            f"fr_rew={team_rew_friendly:8.1f} | en_rew={team_rew_enemy:8.1f} | "
            f"fr_alive={friendly_alive:2d} | en_alive={enemy_alive:2d} | "
            f"ugv={'GOAL' if ugv_reached_goal else 'DESTROYED' if ugv_destroyed else 'ALIVE'}"
        )

    # --- Summary ---
    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "num_episodes": num_episodes,
        "avg_steps": _mean(metrics["steps"]),
        "avg_friendly_reward": _mean(metrics["friendly_reward"]),
        "avg_enemy_reward": _mean(metrics["enemy_reward"]),
        "avg_friendly_alive": _mean(metrics["friendly_alive"]),
        "avg_enemy_alive": _mean(metrics["enemy_alive"]),
        "ugv_success_rate": _mean([int(x) for x in metrics["ugv_reached_goal"]]),
        "ugv_destruction_rate": _mean([int(x) for x in metrics["ugv_destroyed"]]),
    }

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for key, val in summary.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            print(f"  {label:30s}: {val:10.3f}")
        else:
            print(f"  {label:30s}: {val}")
    print("=" * 60)

    # --- Save metrics ---
    if save_metrics:
        os.makedirs(os.path.dirname(save_metrics) or ".", exist_ok=True)
        with open(save_metrics, "w") as f:
            json.dump({"summary": summary, "episodes": metrics}, f, indent=2)
        print(f"\nMetrics saved to {save_metrics}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MARSim agents")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--render", type=int, default=0, help="Render last K episodes")
    parser.add_argument("--ugv-skip", type=int, default=3, help="UGV action skip interval")
    parser.add_argument("--friendly", type=str, default="models/friendly_agents-final",
                        help="Friendly checkpoint path (without .pth)")
    parser.add_argument("--enemy", type=str, default="models/enemy_agents-final",
                        help="Enemy checkpoint path (without .pth)")
    parser.add_argument("--save-metrics", type=str, default=None,
                        help="Save per-episode metrics to JSON file")
    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        render_last_k=args.render,
        ugv_action_skip=args.ugv_skip,
        friendly_ckpt=args.friendly,
        enemy_ckpt=args.enemy,
        save_metrics=args.save_metrics,
    )
