"""
MARSim evaluation framework.

Loads trained PPO checkpoints and runs evaluation episodes, collecting
per-episode metrics and printing a summary table.  Optionally renders
the last K episodes for visual inspection.

Supports sweeping across training checkpoints to produce learning curves::

    python evaluate.py --sweep --episodes 100 --save-metrics results/sweep.json

Usage::

    python evaluate.py                                        # defaults
    python evaluate.py --episodes 50 --render 5               # custom
    python evaluate.py --friendly models/friendly_agents-90 \\
                       --enemy    models/enemy_agents-90
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm

from MARSim.PPO_Policy import PPO
from MARSim.map_generator import Battlefield
from MARSim.envs import make_MARSim
from MARSim.grid_config import GridConfig, AgentType
from MARSim.a_star_policy import AStarAgent
from MARSim.utils import build_obs_tensor


def _find_checkpoints(model_dir="models"):
    """Discover paired friendly/enemy checkpoints saved every N updates."""
    pattern = os.path.join(model_dir, "friendly_agents-*.pth")
    pairs = []
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)  # friendly_agents-10.pth
        m = re.match(r"friendly_agents-(\d+)\.pth", basename)
        if not m:
            continue
        update_idx = int(m.group(1))
        enemy_path = os.path.join(model_dir, f"enemy_agents-{update_idx}.pth")
        if os.path.exists(enemy_path):
            pairs.append((
                update_idx,
                os.path.join(model_dir, f"friendly_agents-{update_idx}"),
                os.path.join(model_dir, f"enemy_agents-{update_idx}"),
            ))
    # Also include final checkpoint
    final_fri = os.path.join(model_dir, "friendly_agents-final.pth")
    final_ene = os.path.join(model_dir, "enemy_agents-final.pth")
    if os.path.exists(final_fri) and os.path.exists(final_ene):
        # Determine the update index for final (one past the last numbered)
        max_idx = max((p[0] for p in pairs), default=0)
        pairs.append((
            max_idx + 10,  # assume final is 10 past last checkpoint
            os.path.join(model_dir, "friendly_agents-final"),
            os.path.join(model_dir, "enemy_agents-final"),
        ))
    pairs.sort(key=lambda t: t[0])
    return pairs


def evaluate_single(
    env, friendly_policy, enemy_policy, agent_types, B,
    fr_idx, en_idx, ugv_idx,
    num_episodes=20, render_last_k=0, ugv_action_skip=3,
):
    """Run evaluation episodes with pre-loaded policies. Returns metrics dict."""
    metrics = {
        "friendly_reward": [],
        "enemy_reward": [],
        "steps": [],
        "friendly_alive": [],
        "enemy_alive": [],
        "ugv_reached_goal": [],
        "ugv_destroyed": [],
    }

    for ep in range(num_episodes):
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

            joint_actions = torch.zeros(B, dtype=torch.long)

            if fr_idx.numel() > 0:
                fr_actions, _, _, _ = friendly_policy.step(obs_tensor[fr_idx])
                joint_actions[fr_idx] = fr_actions.view(-1)

            if en_idx.numel() > 0:
                en_actions, _, _, _ = enemy_policy.step(obs_tensor[en_idx])
                joint_actions[en_idx] = en_actions.view(-1)

            drone_data = env.unwrapped.get_drone_obstacle_data()
            ugv_agent.update_from_drones(drone_data)

            if ugv_idx.numel() > 0:
                for k in ugv_idx.tolist():
                    if steps % ugv_action_skip == 0:
                        joint_actions[k] = int(ugv_agent.act(obs[k]))
                    else:
                        joint_actions[k] = 0

            obs, rewards, terminated, truncated, infos = env.step(joint_actions.tolist())

            rew_tensor = torch.tensor(rewards, dtype=torch.float32)
            if fr_idx.numel() > 0:
                team_rew_friendly += rew_tensor[fr_idx].sum().item()
            if en_idx.numel() > 0:
                team_rew_enemy += rew_tensor[en_idx].sum().item()

            if all(terminated) or all(truncated):
                break

        alive = torch.tensor([infos[i].get("is_active", True) for i in range(B)], dtype=torch.bool)
        friendly_alive = int(alive[fr_idx].sum().item()) if fr_idx.numel() > 0 else 0
        enemy_alive = int(alive[en_idx].sum().item()) if en_idx.numel() > 0 else 0

        for k in ugv_idx.tolist():
            if not infos[k].get("is_active", True):
                if env.unwrapped.grid.on_goal(k):
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

    return metrics


def evaluate(
    num_episodes: int = 20,
    render_last_k: int = 0,
    ugv_action_skip: int = 3,
    friendly_ckpt: str = "models/friendly_agents-final",
    enemy_ckpt: str = "models/enemy_agents-final",
    save_metrics: str = None,
):
    """Run evaluation episodes against a single checkpoint pair."""
    bf = Battlefield()
    env = make_MARSim(
        grid_config=GridConfig(num_agents=50, size=50, density=0.0, map=bf.map)
    )

    r = env.grid_config.obs_radius
    obs_dim = 2 + 2 + 2 * (2 * r + 1) ** 2
    act_dim = len(env.grid_config.MOVES)

    friendly_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
    enemy_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
    friendly_policy.load(friendly_ckpt)
    enemy_policy.load(enemy_ckpt)

    agent_types = env.grid_config.agent_types
    B = env.grid_config.num_agents

    is_friendly_uav = torch.tensor([t == AgentType.FRIENDLY_UAV for t in agent_types], dtype=torch.bool)
    is_enemy_uav    = torch.tensor([t == AgentType.ENEMY_UAV    for t in agent_types], dtype=torch.bool)
    is_ugv          = torch.tensor([t.is_ugv                    for t in agent_types], dtype=torch.bool)

    fr_idx = is_friendly_uav.nonzero(as_tuple=False).squeeze(-1)
    en_idx = is_enemy_uav.nonzero(as_tuple=False).squeeze(-1)
    ugv_idx = is_ugv.nonzero(as_tuple=False).squeeze(-1)

    metrics = evaluate_single(
        env, friendly_policy, enemy_policy, agent_types, B,
        fr_idx, en_idx, ugv_idx,
        num_episodes=num_episodes, render_last_k=render_last_k,
        ugv_action_skip=ugv_action_skip,
    )

    # Print per-episode results
    for ep in range(num_episodes):
        ugv_status = (
            "GOAL" if metrics["ugv_reached_goal"][ep]
            else "DESTROYED" if metrics["ugv_destroyed"][ep]
            else "ALIVE"
        )
        print(
            f"  Ep {ep+1:3d}/{num_episodes} | "
            f"steps={metrics['steps'][ep]:4d} | "
            f"fr_rew={metrics['friendly_reward'][ep]:8.1f} | "
            f"en_rew={metrics['enemy_reward'][ep]:8.1f} | "
            f"fr_alive={metrics['friendly_alive'][ep]:2d} | "
            f"en_alive={metrics['enemy_alive'][ep]:2d} | "
            f"ugv={ugv_status}"
        )

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

    if save_metrics:
        os.makedirs(os.path.dirname(save_metrics) or ".", exist_ok=True)
        with open(save_metrics, "w") as f:
            json.dump({"summary": summary, "episodes": metrics}, f, indent=2)
        print(f"\nMetrics saved to {save_metrics}")

    return summary


def evaluate_sweep(
    num_episodes: int = 100,
    ugv_action_skip: int = 3,
    save_metrics: str = "results/sweep.json",
):
    """Evaluate every cached checkpoint pair and save a training curve."""
    pairs = _find_checkpoints()
    if not pairs:
        print("No checkpoint pairs found in models/")
        return

    print(f"Found {len(pairs)} checkpoint pairs: updates {[p[0] for p in pairs]}")

    bf = Battlefield()
    env = make_MARSim(
        grid_config=GridConfig(num_agents=50, size=50, density=0.0, map=bf.map)
    )

    r = env.grid_config.obs_radius
    obs_dim = 2 + 2 + 2 * (2 * r + 1) ** 2
    act_dim = len(env.grid_config.MOVES)

    agent_types = env.grid_config.agent_types
    B = env.grid_config.num_agents

    is_friendly_uav = torch.tensor([t == AgentType.FRIENDLY_UAV for t in agent_types], dtype=torch.bool)
    is_enemy_uav    = torch.tensor([t == AgentType.ENEMY_UAV    for t in agent_types], dtype=torch.bool)
    is_ugv          = torch.tensor([t.is_ugv                    for t in agent_types], dtype=torch.bool)

    fr_idx = is_friendly_uav.nonzero(as_tuple=False).squeeze(-1)
    en_idx = is_enemy_uav.nonzero(as_tuple=False).squeeze(-1)
    ugv_idx = is_ugv.nonzero(as_tuple=False).squeeze(-1)

    sweep_results = {
        "updates": [],
        "avg_friendly_reward": [],
        "avg_enemy_reward": [],
        "avg_steps": [],
        "avg_friendly_alive": [],
        "avg_enemy_alive": [],
        "ugv_success_rate": [],
        "ugv_destruction_rate": [],
    }

    for update_idx, fri_ckpt, ene_ckpt in tqdm(pairs, desc="Sweep"):
        friendly_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
        enemy_policy = PPO(observation_shape=obs_dim, action_shape=act_dim)
        friendly_policy.load(fri_ckpt)
        enemy_policy.load(ene_ckpt)

        metrics = evaluate_single(
            env, friendly_policy, enemy_policy, agent_types, B,
            fr_idx, en_idx, ugv_idx,
            num_episodes=num_episodes, ugv_action_skip=ugv_action_skip,
        )

        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        sweep_results["updates"].append(update_idx)
        sweep_results["avg_friendly_reward"].append(_mean(metrics["friendly_reward"]))
        sweep_results["avg_enemy_reward"].append(_mean(metrics["enemy_reward"]))
        sweep_results["avg_steps"].append(_mean(metrics["steps"]))
        sweep_results["avg_friendly_alive"].append(_mean(metrics["friendly_alive"]))
        sweep_results["avg_enemy_alive"].append(_mean(metrics["enemy_alive"]))
        sweep_results["ugv_success_rate"].append(
            _mean([int(x) for x in metrics["ugv_reached_goal"]]))
        sweep_results["ugv_destruction_rate"].append(
            _mean([int(x) for x in metrics["ugv_destroyed"]]))

        print(
            f"  Update {update_idx:4d} | "
            f"fr_rew={sweep_results['avg_friendly_reward'][-1]:8.1f} | "
            f"en_rew={sweep_results['avg_enemy_reward'][-1]:8.1f} | "
            f"ugv_success={sweep_results['ugv_success_rate'][-1]:.2f} | "
            f"ugv_destroyed={sweep_results['ugv_destruction_rate'][-1]:.2f}"
        )

    if save_metrics:
        os.makedirs(os.path.dirname(save_metrics) or ".", exist_ok=True)
        with open(save_metrics, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep metrics saved to {save_metrics}")

    return sweep_results


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
    parser.add_argument("--sweep", action="store_true",
                        help="Evaluate all cached checkpoints for a training curve")
    args = parser.parse_args()

    if args.sweep:
        evaluate_sweep(
            num_episodes=args.episodes,
            ugv_action_skip=args.ugv_skip,
            save_metrics=args.save_metrics or "results/sweep.json",
        )
    else:
        evaluate(
            num_episodes=args.episodes,
            render_last_k=args.render,
            ugv_action_skip=args.ugv_skip,
            friendly_ckpt=args.friendly,
            enemy_ckpt=args.enemy,
            save_metrics=args.save_metrics,
        )
