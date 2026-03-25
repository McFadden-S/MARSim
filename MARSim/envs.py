"""
Core MARSim Gymnasium environment.

Implements the multi-agent grid world where friendly and enemy UAVs/UGVs
interact.  The environment handles observation generation, reward shaping,
and four different collision resolution systems.
"""

from typing import Optional
from collections import defaultdict

import gymnasium

from MARSim.grid import Grid
from MARSim.grid_config import (
    GridConfig,
    AgentType,
    REWARD_STEP_PENALTY,
    REWARD_DETECTION,
    REWARD_FRIENDLY_UAV_KILLED,
    REWARD_ENEMY_KILL_BONUS,
    REWARD_UGV_SUCCESS,
    REWARD_UGV_FAILURE,
    REWARD_UGV_ENEMY_BONUS,
    REWARD_UGV_ENEMY_PENALTY,
    REWARD_DEFENCE_SCALE,
    REWARD_DEFENCE_THRESHOLD,
)
from MARSim.wrappers.multi_time_limit import MultiTimeLimit
from MARSim.graphics import PygameRenderer
from MARSim.map_generator import Battlefield


class MARSim(gymnasium.Env):
    """
    Multi-agent resupply environment.

    **Observation space** (per agent):
        - ``obstacles``: (2r+1, 2r+1) binary grid of nearby terrain.
        - ``agents``:    (2r+1, 2r+1) type-encoded agent positions.
        - ``xy``:        absolute (row, col) position.
        - ``target_xy``: goal position (or sentinel for UAVs).

    **Action space**: Discrete(5) — stay / N / S / W / E.

    **Reward structure** (see ``grid_config.py`` for constant values):
        - Per-step penalty to encourage efficiency.
        - Detection bonuses for spotting opponents within obs_radius.
        - Defence shaping for friendly UAVs keeping enemies away from UGV.
        - Large terminal rewards/penalties when the UGV is destroyed or
          reaches its goal.
    """

    def __init__(self, grid_config: GridConfig = GridConfig(num_agents=2)):
        self.grid: Optional[Grid] = None
        self.grid_config = grid_config
        self.was_on_goal: Optional[list[bool]] = None
        self.agent_types = list(self.grid_config.agent_types)

        # Gymnasium spaces
        self.action_space = gymnasium.spaces.Discrete(len(self.grid_config.MOVES))
        full_size = self.grid_config.obs_radius * 2 + 1
        self.observation_space = gymnasium.spaces.Dict(
            obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

        # Rendering
        self.graphics: Optional[PygameRenderer] = None
        self.display_graphics: bool = False

        # Reward-shaping bookkeeping (reset each episode)
        self._visited_cells: Optional[list[set]] = None
        self._detected_opponents: Optional[list[set]] = None

    def get_num_agents(self) -> int:
        return self.grid_config.num_agents

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: list):
        """
        Execute a joint action for all agents.

        Returns:
            observations, rewards, terminated, truncated, infos
        """
        assert len(action) == self.grid_config.num_agents

        # Snapshot active status before movement to detect destruction events
        prev_active = list(self.grid.is_active.values())

        self.move_agents(action)
        self.update_was_on_goal()

        rewards, terminated = self._compute_rewards(prev_active)

        if self.display_graphics:
            self.graphics.draw_state(
                agents_xy=self.grid.positions_xy,
                targets_xy=self.grid.finishes_xy,
                agent_types=self.agent_types,
                is_active=[not t for t in terminated],
            )

        return self._obs(), rewards, terminated, [False] * self.grid_config.num_agents, self._get_infos()

    def _compute_rewards(self, prev_active: list[bool]):
        """
        Compute shaped rewards and termination flags for all agents.

        Reward components:
            1. Step penalty — constant negative per timestep.
            2. Detection — bonuses for spotting opponents within obs_radius.
            3. Defence — proximity-based shaping for friendly UAVs.
            4. Destruction events — bonuses/penalties when agents are killed.
            5. Terminal — large team-wide rewards when UGV succeeds or fails.
        """
        num = self.grid_config.num_agents
        detection_radius = self.grid_config.obs_radius

        rewards: list[float] = []
        terminated: list[bool] = []
        ugv_destroyed = False
        ugv_succeeded = False

        for idx in range(num):
            reward = REWARD_STEP_PENALTY
            agent_type = self.grid_config.agent_types[idx]

            if self.grid.is_active[idx]:
                pos = self.grid.positions_xy[idx]

                # --- Detection rewards ---
                reward += self._detection_reward(idx, agent_type, pos, detection_radius)

                # --- Defence shaping (friendly UAVs only) ---
                if agent_type == AgentType.FRIENDLY_UAV:
                    reward += self._defence_reward()

            # --- Destruction events ---
            if prev_active[idx] and not self.grid.is_active[idx]:
                if agent_type == AgentType.FRIENDLY_UGV:
                    if not self.grid.on_goal(idx):
                        ugv_destroyed = True
                    else:
                        ugv_succeeded = True
                elif agent_type == AgentType.FRIENDLY_UAV:
                    reward += REWARD_FRIENDLY_UAV_KILLED
                else:
                    reward += REWARD_ENEMY_KILL_BONUS

            rewards.append(reward)
            terminated.append(not self.grid.is_active[idx])

        # --- Team-wide terminal rewards ---
        if ugv_destroyed:
            self._apply_terminal_rewards_impl(rewards, terminated, success=False)
        if ugv_succeeded:
            self._apply_terminal_rewards_impl(rewards, terminated, success=True)

        return rewards, terminated

    def _detection_reward(self, agent_idx: int, agent_type: AgentType,
                          pos: tuple, detection_radius: int) -> float:
        """Reward for detecting opponents within the observation radius."""
        reward = 0.0
        for other_idx in range(self.grid_config.num_agents):
            if other_idx == agent_idx or not self.grid.is_active[other_idx]:
                continue

            other_type = self.grid_config.agent_types[other_idx]
            ox, oy = self.grid.positions_xy[other_idx]
            manhattan = abs(pos[0] - ox) + abs(pos[1] - oy)

            if manhattan > detection_radius:
                continue
            if other_idx in self._detected_opponents[agent_idx]:
                continue

            # Friendly UAV detecting an enemy UAV
            if agent_type == AgentType.FRIENDLY_UAV and other_type == AgentType.ENEMY_UAV:
                reward += REWARD_DETECTION

            # Enemy UAV detecting a friendly agent (distance-decayed)
            if agent_type == AgentType.ENEMY_UAV and other_type.is_friendly:
                if manhattan == 0 or detection_radius == 0:
                    reward += 1.0
                else:
                    reward += 1.0 / (manhattan / detection_radius)

        return reward

    def _defence_reward(self) -> float:
        """
        Proximity-based reward for friendly UAVs: positive when enemies
        are far from the UGV, negative when dangerously close.
        """
        ugv_indices = [
            i for i, t in enumerate(self.grid_config.agent_types)
            if t == AgentType.FRIENDLY_UGV and self.grid.is_active[i]
        ]
        enemy_indices = [
            i for i, t in enumerate(self.grid_config.agent_types)
            if t == AgentType.ENEMY_UAV and self.grid.is_active[i]
        ]
        if not ugv_indices or not enemy_indices:
            return 0.0

        ugv_pos = self.grid.positions_xy[ugv_indices[0]]
        min_dist = min(
            abs(ugv_pos[0] - self.grid.positions_xy[e][0])
            + abs(ugv_pos[1] - self.grid.positions_xy[e][1])
            for e in enemy_indices
        )

        # Normalise by maximum possible Manhattan distance (corner-to-corner)
        max_dist = max(1, 2 * (self.grid.config.size - 1))
        s = min_dist / float(max_dist)

        k = REWARD_DEFENCE_SCALE
        thresh = REWARD_DEFENCE_THRESHOLD

        if s > thresh:
            return k * s
        else:
            # Inside danger zone — increasingly negative as s -> 0
            return -k * 10 * (1.0 - (s / thresh))

    def _apply_terminal_rewards_impl(self, rewards: list[float],
                                     terminated: list[bool], success: bool):
        """Instance-method version that applies team-wide terminal rewards."""
        for idx in range(self.grid_config.num_agents):
            atype = self.grid_config.agent_types[idx]
            if atype.is_friendly:
                rewards[idx] += REWARD_UGV_SUCCESS if success else REWARD_UGV_FAILURE
            else:
                rewards[idx] += REWARD_UGV_ENEMY_PENALTY if success else REWARD_UGV_ENEMY_BONUS
            terminated[idx] = True

    # ------------------------------------------------------------------
    # Grid initialisation and goal tracking
    # ------------------------------------------------------------------

    def _initialize_grid(self):
        """Create a fresh grid (and generate a new map if none was provided)."""
        if self.grid_config.map is None:
            bf = Battlefield()
            self.grid_config.map = bf.map
        self.grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [
            self.grid.on_goal(i) and self.grid.is_active[i]
            for i in range(self.grid_config.num_agents)
        ]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        display_graphics: bool = False,
    ):
        """Reset the environment for a new episode."""
        self._initialize_grid()
        self.update_was_on_goal()
        self.display_graphics = display_graphics

        if self.display_graphics:
            if self.graphics is None:
                self.graphics = PygameRenderer(
                    width=self.grid.config.size,
                    height=self.grid.config.size,
                    obs_radius=self.grid_config.obs_radius,
                )
            self.graphics.draw_static(self.grid.obstacles)
            self.graphics.draw_state(
                agents_xy=self.grid.positions_xy,
                targets_xy=self.grid.finishes_xy,
                agent_types=self.agent_types,
                is_active=[True] * self.grid_config.num_agents,
            )

        if seed is not None:
            self.grid.seed = seed

        # Reset per-episode reward-shaping state
        self._visited_cells = [set() for _ in range(self.grid_config.num_agents)]
        self._detected_opponents = [set() for _ in range(self.grid_config.num_agents)]

        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()

    # ------------------------------------------------------------------
    # Observations and info
    # ------------------------------------------------------------------

    def _obs(self) -> list[dict]:
        """Build a list of observation dicts, one per agent."""
        return [
            {
                "obstacles": self.grid.get_obstacles_for_agent(i),
                "agents": self.grid.get_positions(i),
                "xy": self.grid.positions_xy[i],
                "target_xy": self.grid.finishes_xy[i],
            }
            for i in range(self.grid_config.num_agents)
        ]

    def _get_infos(self) -> list[dict]:
        return [
            {"is_active": self.grid.is_active[i]}
            for i in range(self.grid_config.num_agents)
        ]

    # ------------------------------------------------------------------
    # Movement and collision systems
    # ------------------------------------------------------------------

    def _revert_action(self, agent_idx, used_cells, cell, actions):
        """Recursively revert an agent's move and cascade to displaced agents."""
        actions[agent_idx] = 0
        used_cells[cell].remove(agent_idx)
        new_cell = self.grid.positions_xy[agent_idx]
        if new_cell in used_cells and len(used_cells[new_cell]) > 0:
            used_cells[new_cell].append(agent_idx)
            return self._revert_action(
                used_cells[new_cell][0], used_cells, new_cell, actions
            )
        else:
            used_cells.setdefault(new_cell, []).append(agent_idx)
        return actions, used_cells

    def move_agents(self, actions):
        """
        Resolve movement for all agents using the configured collision system.

        Supported systems:
            - ``priority``: agents move sequentially; first-come-first-served.
            - ``block_both``: if two agents target the same cell, neither moves.
            - ``soft``: edge-swap and cell-conflict detection with cascading
              reversions.
            - ``uav_collision``: UAVs can collide and destroy each other
              across teams; UGVs are vulnerable to enemy UAV kamikazes.
        """
        system = self.grid.config.collision_system

        if system == "priority":
            self._move_priority(actions)
        elif system == "block_both":
            self._move_block_both(actions)
        elif system == "soft":
            self._move_soft(actions)
        elif system == "uav_collision":
            self._move_uav_collision(actions)
        else:
            raise ValueError(f"Unknown collision system: {system}")

    def _move_priority(self, actions):
        """Simple sequential movement — earlier indices have priority."""
        for i in range(self.grid_config.num_agents):
            if self.grid.is_active[i]:
                self.grid.move(i, actions[i])

    def _move_block_both(self, actions):
        """If two agents target the same cell, both are blocked."""
        used_cells = {}
        agents_xy = self.grid.get_agents_xy()

        for idx, (x, y) in enumerate(agents_xy):
            if self.grid.is_active[idx]:
                dx, dy = self.grid_config.MOVES[actions[idx]]
                target = (x + dx, y + dy)
                used_cells[target] = "blocked" if target in used_cells else "visited"
                used_cells[(x, y)] = "blocked"

        for idx in range(self.grid_config.num_agents):
            if self.grid.is_active[idx]:
                x, y = agents_xy[idx]
                dx, dy = self.grid_config.MOVES[actions[idx]]
                if used_cells.get((x + dx, y + dy)) != "blocked":
                    self.grid.move(idx, actions[idx])

    def _move_soft(self, actions):
        """Edge-swap + cell-conflict detection with cascading reversions."""
        used_cells = {}
        used_edges = {}
        agents_xy = self.grid.get_agents_xy()

        # Record intended moves
        for idx, (x, y) in enumerate(agents_xy):
            if self.grid.is_active[idx]:
                dx, dy = self.grid.config.MOVES[actions[idx]]
                used_cells.setdefault((x + dx, y + dy), []).append(idx)
                used_edges[x, y, x + dx, y + dy] = [idx]
                if dx != 0 or dy != 0:
                    used_edges.setdefault((x + dx, y + dy, x, y), []).append(idx)

        # Revert edge swaps
        for idx, (x, y) in enumerate(agents_xy):
            if self.grid.is_active[idx]:
                dx, dy = self.grid.config.MOVES[actions[idx]]
                if len(used_edges[x, y, x + dx, y + dy]) > 1:
                    used_cells[x + dx, y + dy].remove(idx)
                    used_cells.setdefault((x, y), []).append(idx)
                    actions[idx] = 0

        # Revert cell conflicts and obstacle collisions
        for idx in reversed(range(len(agents_xy))):
            x, y = agents_xy[idx]
            if self.grid.is_active[idx]:
                dx, dy = self.grid.config.MOVES[actions[idx]]
                if (
                    len(used_cells[x + dx, y + dy]) > 1
                    or self.grid.has_obstacle(x + dx, y + dy)
                ):
                    actions, used_cells = self._revert_action(
                        idx, used_cells, (x + dx, y + dy), actions
                    )

        # Execute surviving moves
        for idx in range(self.grid_config.num_agents):
            if self.grid.is_active[idx]:
                self.grid.move_without_checks(idx, actions[idx])

    def _move_uav_collision(self, actions):
        """
        UAV-aware collision system:
            1. Cross-team same-cell UAV collisions destroy both.
            2. Cross-team head-on swaps destroy both.
            3. Enemy UAV entering a friendly UGV's cell destroys both.
            4. Survivors move with priority semantics.
        """
        agents_xy = self.grid.get_agents_xy()
        atypes = self.grid_config.agent_types

        # Compute intended positions
        intended = []
        for i, (x, y) in enumerate(agents_xy):
            if not self.grid.is_active[i] or actions[i] == 0:
                intended.append((x, y))
            else:
                dx, dy = self.grid.config.MOVES[actions[i]]
                intended.append((x + dx, y + dy))

        doomed_uavs: set[int] = set()
        doomed_ugvs: set[int] = set()

        # 1) Same-cell collisions among UAVs (cross-team only)
        bucket: dict[tuple, list[int]] = defaultdict(list)
        for i, end in enumerate(intended):
            if self.grid.is_active[i] and atypes[i].is_uav:
                bucket[end].append(i)

        for end_cell, ids in bucket.items():
            if len(ids) >= 2:
                friendly = [i for i in ids if atypes[i] == AgentType.FRIENDLY_UAV]
                enemy = [i for i in ids if atypes[i] == AgentType.ENEMY_UAV]
                if friendly and enemy:
                    doomed_uavs.update(friendly)
                    doomed_uavs.update(enemy)

        # 2) Head-on swaps (cross-team only)
        start_bucket: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(agents_xy):
            if self.grid.is_active[i] and atypes[i].is_uav:
                start_bucket[s].append(i)

        for i in range(self.grid_config.num_agents):
            if not self.grid.is_active[i] or not atypes[i].is_uav:
                continue
            if intended[i] == agents_xy[i]:
                continue  # didn't move — can't swap
            for j in start_bucket.get(intended[i], []):
                if i == j or not self.grid.is_active[j] or not atypes[j].is_uav:
                    continue
                if intended[j] == agents_xy[i]:
                    # Cross-team swap
                    if atypes[i].is_friendly != atypes[j].is_friendly:
                        doomed_uavs.add(i)
                        doomed_uavs.add(j)

        # 3) Enemy UAV kamikaze into friendly UGV
        for i in range(self.grid_config.num_agents):
            if not (self.grid.is_active[i] and atypes[i] == AgentType.ENEMY_UAV):
                continue
            for j in range(self.grid_config.num_agents):
                if not (self.grid.is_active[j] and atypes[j] == AgentType.FRIENDLY_UGV):
                    continue
                if intended[i] == agents_xy[j] or intended[i] == intended[j]:
                    doomed_uavs.add(i)
                    doomed_ugvs.add(j)

        # Destroy all doomed agents before moving survivors
        for i in doomed_uavs | doomed_ugvs:
            if self.grid.is_active[i]:
                self.grid.hide_agent(i)
                self.grid.is_active[i] = False

        # Survivors move with priority semantics; UGVs reaching goal are hidden
        for i in range(self.grid_config.num_agents):
            if self.grid.is_active[i]:
                self.grid.move(i, actions[i])
                if self.grid.on_goal(i):
                    self.grid.hide_agent(i)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)


# ── Factory ──────────────────────────────────────────────────────────────────

def make_MARSim(grid_config: GridConfig) -> MultiTimeLimit:
    """
    Create a MARSim environment wrapped with a multi-agent time limit.

    Args:
        grid_config: Configuration for the grid, agents, and collision system.

    Returns:
        A ``MultiTimeLimit``-wrapped ``MARSim`` environment.
    """
    env = MARSim(grid_config=grid_config)
    return MultiTimeLimit(env, grid_config.max_episode_steps)
