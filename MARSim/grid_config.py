"""
Grid configuration and agent type definitions for MARSim.

All tunable constants (grid dimensions, reward values, collision mode, etc.)
live here so they can be adjusted in one place rather than scattered across
the codebase as magic numbers.
"""

from typing import Optional, Union, Literal
from enum import IntEnum


# ── Agent types ──────────────────────────────────────────────────────────────

class AgentType(IntEnum):
    """Identifiers for the four agent roles in MARSim."""
    FRIENDLY_UGV = 0
    FRIENDLY_UAV = 1
    ENEMY_UAV    = 2
    ENEMY_UGV    = 3

    @property
    def is_friendly(self) -> bool:
        return self in (AgentType.FRIENDLY_UAV, AgentType.FRIENDLY_UGV)

    @property
    def is_enemy(self) -> bool:
        return self in (AgentType.ENEMY_UAV, AgentType.ENEMY_UGV)

    @property
    def is_uav(self) -> bool:
        return self in (AgentType.FRIENDLY_UAV, AgentType.ENEMY_UAV)

    @property
    def is_ugv(self) -> bool:
        return self in (AgentType.FRIENDLY_UGV, AgentType.ENEMY_UGV)


def default_agent_types(num_agents: int) -> list:
    """
    Build the default agent roster: 1 friendly UGV + the remaining slots
    split evenly between friendly and enemy UAVs (extra slot to friendly
    if the split is odd).
    """
    remaining = num_agents - 1
    friendly_uavs = remaining // 2 + (remaining % 2)
    enemy_uavs = remaining // 2
    return (
        [AgentType.FRIENDLY_UGV]
        + [AgentType.FRIENDLY_UAV] * friendly_uavs
        + [AgentType.ENEMY_UAV] * enemy_uavs
    )


# ── Reward constants ─────────────────────────────────────────────────────────
# Centralised here so reward shaping can be tuned without touching envs.py.

REWARD_STEP_PENALTY       = -0.1    # per-step cost to encourage efficiency
REWARD_DETECTION          =  1.0    # friendly UAV detects an enemy UAV
REWARD_FRIENDLY_UAV_KILLED = -1.0   # friendly UAV is destroyed
REWARD_ENEMY_KILL_BONUS   = 10.0    # enemy UAV destroys an opponent
REWARD_UGV_SUCCESS        = 50.0    # UGV reaches its goal (team bonus)
REWARD_UGV_FAILURE        = -50.0   # UGV destroyed before goal (team penalty)
REWARD_UGV_ENEMY_BONUS    = 100.0   # enemy team bonus when UGV is destroyed
REWARD_UGV_ENEMY_PENALTY  = -100.0  # enemy team penalty when UGV succeeds
REWARD_DEFENCE_SCALE      =  0.05   # scale of the UGV-defence proximity term
REWARD_DEFENCE_THRESHOLD  =  0.10   # normalised distance danger threshold


# ── Dummy target sentinel ────────────────────────────────────────────────────
# UAVs have no goal position; this sentinel value marks their target as "none".
DUMMY_TARGET = (-4, -4)


# ── Grid configuration ───────────────────────────────────────────────────────

class GridConfig:
    """
    Central configuration object shared by the environment, grid, and
    generator.  All runtime parameters are collected here.

    Attributes:
        MOVES: Action-index-to-delta mapping.
               0 = stay, 1 = north, 2 = south, 3 = west, 4 = east.
        num_agents: Total number of agents (all types combined).
        size: Side length of the square grid (before the artificial border).
        density: Obstacle density for random map generation (0.0 = use
                 provided map).
        obs_radius: Radius of each agent's local observation window.
        collision_system: One of 'block_both', 'priority', 'soft',
                          'uav_collision'.
        max_episode_steps: Hard step limit per episode (used by TimeLimit
                           wrapper).
        map: Pre-built grid (list-of-lists or string) or ``None`` for
             random generation.
    """

    MOVES: list = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
    FREE: Literal[0] = 0
    OBSTACLE: Literal[1] = 1
    empty_outside: bool = True

    def __init__(self, **kwargs):
        self.on_target: Literal["finish", "nothing", "restart"] = "finish"
        self.seed: Optional[int] = None
        self.density: float = 0.3
        self.obs_radius: int = 5
        self.agents_xy: Optional[list] = None
        self.targets_xy: Optional[list] = None

        self.num_agents: int = 10
        self.agent_types: Optional[list] = None

        self.possible_agents_xy: Optional[list] = None
        self.possible_targets_xy: Optional[list] = None
        self.collision_system: Literal[
            "block_both", "priority", "soft", "uav_collision"
        ] = "uav_collision"
        self.persistent: bool = False
        self.map: Optional[Union[list, str]] = None

        self.max_episode_steps: int = 200

        # Override defaults with any kwargs supplied by the caller
        self.__dict__.update(kwargs)

        # Auto-generate the agent roster if not explicitly provided
        if self.agent_types is None:
            self.agent_types = default_agent_types(self.num_agents)
