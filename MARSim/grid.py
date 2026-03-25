"""
Grid state management for MARSim.

Handles agent placement, observation generation, movement validation,
collision detection, and the artificial border that pads the map so that
observation windows never go out-of-bounds.
"""

from copy import deepcopy
import math
import warnings

import numpy as np

from MARSim.generator import generate_positions_and_targets_fast, generate_from_possible_positions
from MARSim.grid_config import GridConfig, AgentType, DUMMY_TARGET


class Grid:
    """
    Manages the 2D grid world: obstacle layout, agent positions, and
    per-agent observations.

    On construction the grid places agents on the map according to the
    configuration, wraps the map with an artificial border, and prepares
    an occupancy matrix for fast collision checks.
    """

    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries: int = 10):
        self.config = grid_config
        self.rnd = np.random.default_rng(grid_config.seed)

        # Parse the map into a numpy obstacle array (0 = free, 1 = wall)
        self.obstacles = np.array([np.array(line) for line in self.config.map]).astype(np.int32)

        # --- Agent placement ---
        if grid_config.targets_xy and grid_config.agents_xy:
            self._place_from_explicit(grid_config)
        elif grid_config.possible_agents_xy and grid_config.possible_targets_xy:
            self.starts_xy, self.finishes_xy = generate_from_possible_positions(self.config)
        else:
            self._place_by_team_sides()

        # Retry if placement failed
        if len(self.starts_xy) != len(self.finishes_xy):
            for attempt in range(num_retries):
                if len(self.starts_xy) == len(self.finishes_xy):
                    warnings.warn(f"Created valid configuration after {attempt} attempts.", Warning, stacklevel=2)
                    break
                self.starts_xy, self.finishes_xy = generate_positions_and_targets_fast(self.obstacles, self.config)

        if not self.starts_xy or not self.finishes_xy or len(self.starts_xy) != len(self.finishes_xy):
            raise OverflowError(
                "Can't create task. Check grid_config (density, num_agents, map)."
            )

        if add_artificial_border:
            self._add_artificial_border()

        # Build occupancy matrix
        filled_positions = np.zeros(self.obstacles.shape)
        for x, y in self.starts_xy:
            filled_positions[x, y] = 1

        self.positions = filled_positions
        self.positions_xy = self.starts_xy
        self._initial_xy = deepcopy(self.starts_xy)
        self.is_active = {i: True for i in range(self.config.num_agents)}

    # ------------------------------------------------------------------
    # Placement strategies
    # ------------------------------------------------------------------

    def _place_from_explicit(self, gc: GridConfig):
        """Use explicitly provided agents_xy / targets_xy."""
        self.starts_xy = gc.agents_xy

        if isinstance(gc.targets_xy[0][0], (list, tuple)):
            self.finishes_xy = [seq[0] for seq in gc.targets_xy]
        else:
            self.finishes_xy = gc.targets_xy

        if len(self.starts_xy) != len(self.finishes_xy):
            raise IndexError("agents_xy and targets_xy must have the same length.")
        if gc.num_agents > len(self.starts_xy):
            raise IndexError(f"Not enough positions to place {gc.num_agents} agents.")

        self.starts_xy = self.starts_xy[:gc.num_agents]
        self.finishes_xy = self.finishes_xy[:gc.num_agents]

        # Clear obstacles under start/finish positions
        for sx, sy in self.starts_xy:
            if self.config.map is not None and self.obstacles[sx, sy] == gc.OBSTACLE:
                warnings.warn(f"Obstacle on start ({sx}, {sy}), replacing with free cell.", Warning, stacklevel=2)
            self.obstacles[sx, sy] = gc.FREE
        for fx, fy in self.finishes_xy:
            if self.config.map is not None and self.obstacles[fx, fy] == gc.OBSTACLE:
                warnings.warn(f"Obstacle on finish ({fx}, {fy}), replacing with free cell.", Warning, stacklevel=2)
            self.obstacles[fx, fy] = gc.FREE

    def _place_by_team_sides(self):
        """
        Spawn friendlies on the left third, enemies on the right third,
        with targets on the opposing side.  UAVs get a dummy target.
        """
        H, W = self.obstacles.shape
        FREE = self.config.FREE

        left_mask = np.indices((H, W))[1] < W // 3
        right_mask = np.indices((H, W))[1] >= 2 * W // 3

        free = self.obstacles == FREE
        free_left = np.argwhere(free & left_mask)
        free_right = np.argwhere(free & right_mask)

        agent_types = list(self.config.agent_types)

        FRIENDLY_KINDS = {AgentType.FRIENDLY_UAV, AgentType.FRIENDLY_UGV}
        ENEMY_KINDS = {AgentType.ENEMY_UAV, AgentType.ENEMY_UGV}

        friendly_ids = [i for i, t in enumerate(agent_types) if t in FRIENDLY_KINDS]
        enemy_ids = [i for i, t in enumerate(agent_types) if t in ENEMY_KINDS]
        other_ids = [i for i in range(len(agent_types)) if i not in friendly_ids and i not in enemy_ids]

        def sample_free(coords: np.ndarray, k: int) -> list[tuple]:
            if coords.shape[0] < k:
                raise OverflowError(f"Not enough free cells to place {k} agents (available={coords.shape[0]}).")
            idx = self.rnd.choice(coords.shape[0], size=k, replace=False)
            return [tuple(map(int, coords[i])) for i in idx]

        # Sample starts
        friendly_starts = sample_free(free_left, len(friendly_ids)) if friendly_ids else []
        enemy_starts = sample_free(free_right, len(enemy_ids)) if enemy_ids else []

        # Remove used cells so finishes don't overlap
        if friendly_starts:
            used = set(friendly_starts)
            free_left = np.array([c for c in free_left if tuple(c) not in used], dtype=int)
        if enemy_starts:
            used = set(enemy_starts)
            free_right = np.array([c for c in free_right if tuple(c) not in used], dtype=int)

        # Sample finishes on the OPPOSITE side
        friendly_finishes = sample_free(free_right, len(friendly_ids)) if friendly_ids else []
        enemy_finishes = sample_free(free_left, len(enemy_ids)) if enemy_ids else []

        # Handle agents that don't fit neatly into friendly/enemy
        other_starts, other_finishes = [], []
        if other_ids:
            other_starts = sample_free(
                free_left if free_left.shape[0] >= len(other_ids) else free_right,
                len(other_ids),
            )
            used = set(other_starts)
            free_left = np.array([c for c in free_left if tuple(c) not in used], dtype=int)
            other_finishes = sample_free(
                free_right if free_right.shape[0] >= len(other_ids) else free_left,
                len(other_ids),
            )

        # Assemble in original agent order
        starts_xy = [None] * self.config.num_agents
        finishes_xy = [None] * self.config.num_agents

        for k, aid in enumerate(friendly_ids):
            starts_xy[aid] = friendly_starts[k]
            finishes_xy[aid] = friendly_finishes[k] if agent_types[aid].is_ugv else DUMMY_TARGET

        for k, aid in enumerate(enemy_ids):
            starts_xy[aid] = enemy_starts[k]
            finishes_xy[aid] = enemy_finishes[k] if agent_types[aid].is_ugv else DUMMY_TARGET

        for k, aid in enumerate(other_ids):
            starts_xy[aid] = other_starts[k]
            finishes_xy[aid] = other_finishes[k] if agent_types[aid].is_ugv else DUMMY_TARGET

        if any(p is None for p in starts_xy) or any(p is None for p in finishes_xy):
            raise OverflowError("Failed to assign starts/finishes for all agents.")

        self.starts_xy, self.finishes_xy = starts_xy, finishes_xy

    # ------------------------------------------------------------------
    # Artificial border
    # ------------------------------------------------------------------

    def _add_artificial_border(self):
        """
        Pad the map with an obs_radius-wide border so that observation
        windows never go out-of-bounds.  The border is surrounded by
        a 1-cell-thick obstacle wall.
        """
        gc = self.config
        r = gc.obs_radius

        if gc.empty_outside:
            padded = np.zeros(np.array(self.obstacles.shape) + r * 2)
        else:
            padded = self.rnd.binomial(1, gc.density, np.array(self.obstacles.shape) + r * 2)

        H, W = padded.shape
        # Draw border walls
        padded[r - 1, r - 1:W - r + 1] = gc.OBSTACLE
        padded[r - 1:H - r + 1, r - 1] = gc.OBSTACLE
        padded[H - r, r - 1:W - r + 1] = gc.OBSTACLE
        padded[r - 1:H - r + 1, W - r] = gc.OBSTACLE
        # Place original map inside
        padded[r:H - r, r:W - r] = self.obstacles

        self.obstacles = padded
        self.starts_xy = [(x + r, y + r) for x, y in self.starts_xy]
        self.finishes_xy = [(x + r, y + r) for x, y in self.finishes_xy]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_obstacles(self, ignore_borders: bool = False) -> np.ndarray:
        """Return a copy of the obstacle grid, optionally without the border."""
        gc = self.config
        if ignore_borders:
            return self.obstacles[gc.obs_radius:-gc.obs_radius, gc.obs_radius:-gc.obs_radius].copy()
        return self.obstacles.copy()

    def get_agents_xy(self, only_active: bool = False, ignore_borders: bool = False) -> list:
        """Return a list of (x, y) positions for all (or active-only) agents."""
        positions = deepcopy(self.positions_xy)
        if only_active:
            active_ids = [i for i, active in self.is_active.items() if active]
            positions = [pos for idx, pos in enumerate(positions) if idx in active_ids]
        if ignore_borders:
            r = self.config.obs_radius
            positions = [[x - r, y - r] for x, y in positions]
        return positions

    @staticmethod
    def to_relative(coordinates, offset):
        """Convert absolute coordinates to offsets from initial positions."""
        result = deepcopy(coordinates)
        for idx in range(len(result)):
            x, y = result[idx]
            dx, dy = offset[idx]
            result[idx] = x - dx, y - dy
        return result

    def get_agents_xy_relative(self):
        return self.to_relative(self.positions_xy, self._initial_xy)

    def get_targets_xy_relative(self):
        return self.to_relative(self.finishes_xy, self._initial_xy)

    def get_obstacles_for_agent(self, agent_id: int) -> np.ndarray:
        """
        Return the local obstacle patch for an agent.
        UAVs see clear sky (all zeros); UGVs see the actual terrain.
        """
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        patch = self.obstacles[x - r:x + r + 1, y - r:y + r + 1]

        if self.config.agent_types[agent_id].is_uav:
            return np.zeros_like(patch).astype(np.float32)
        return patch.astype(np.float32)

    def get_positions(self, agent_id: int, pad_value: float = -1.0) -> np.ndarray:
        """
        Return a (2r+1, 2r+1) array centered on *agent_id* showing nearby
        agent positions encoded by type:

            FRIENDLY_UAV -> 1.0, ENEMY_UAV -> 2.0,
            FRIENDLY_UGV -> 3.0, ENEMY_UGV -> 4.0

        Out-of-bounds cells are filled with *pad_value*.
        """
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        size = 2 * r + 1

        pos = np.full((size, size), pad_value, dtype=np.float32)

        # Compute valid ranges in the global occupancy grid
        x_min, x_max = max(0, x - r), min(self.positions.shape[0], x + r + 1)
        y_min, y_max = max(0, y - r), min(self.positions.shape[1], y + r + 1)

        # Local slice offsets
        lx0 = x_min - (x - r)
        lx1 = lx0 + (x_max - x_min)
        ly0 = y_min - (y - r)
        ly1 = ly0 + (y_max - y_min)

        local_patch = self.positions[x_min:x_max, y_min:y_max].copy()

        # Overlay agent-type-specific values for active agents in the patch
        TYPE_VALUES = {
            AgentType.FRIENDLY_UAV: 1.0,
            AgentType.ENEMY_UAV:    2.0,
            AgentType.FRIENDLY_UGV: 3.0,
            AgentType.ENEMY_UGV:    4.0,
        }

        for idx, (ax, ay) in enumerate(self.positions_xy):
            if not self.is_active[idx]:
                continue
            if x_min <= ax < x_max and y_min <= ay < y_max:
                atype = self.config.agent_types[idx]
                local_patch[ax - x_min, ay - y_min] = TYPE_VALUES.get(atype, 0.5)

        pos[lx0:lx1, ly0:ly1] = local_patch
        return pos

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_agent_to_cell(self, agent_id: int, x: int, y: int):
        """Forcibly teleport an agent to (x, y).  Raises if target is occupied."""
        if self.positions[self.positions_xy[agent_id]] == self.config.FREE:
            raise KeyError(f"Agent {agent_id} is not on the map.")
        self.positions[self.positions_xy[agent_id]] = self.config.FREE
        if self.obstacles[x, y] != self.config.FREE or self.positions[x, y] != self.config.FREE:
            raise ValueError(f"Can't force agent to blocked position ({x}, {y}).")
        self.positions_xy[agent_id] = (x, y)
        self.positions[x, y] = self.config.OBSTACLE

    def has_obstacle(self, x: int, y: int) -> bool:
        return self.obstacles[x, y] == self.config.OBSTACLE

    def move_without_checks(self, agent_id: int, action: int):
        """Move an agent without obstacle/occupancy validation (used by soft collision)."""
        x, y = self.positions_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        self.positions[x, y] = self.config.FREE
        self.positions[x + dx, y + dy] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x + dx, y + dy)

    def _inside_artificial_border(self, x: int, y: int) -> bool:
        """Return True if (x, y) is strictly inside the artificial border ring."""
        r = self.config.obs_radius
        H, W = self.obstacles.shape
        return (r <= x < H - r) and (r <= y < W - r)

    def move(self, agent_id: int, action: int):
        """
        Move an agent according to *action*, respecting obstacles and borders.
        UAVs ignore terrain obstacles but respect the border.
        UGVs respect both obstacles and other agents' positions.
        """
        x, y = self.positions_xy[agent_id]
        dx, dy = self.config.MOVES[action]

        if self.config.agent_types[agent_id].is_uav:
            # UAVs fly over terrain — only the artificial border stops them
            nx, ny = x + dx, y + dy
            if self._inside_artificial_border(nx, ny):
                self.positions[x, y] = self.config.FREE
                x, y = nx, ny
                self.positions[x, y] = self.config.OBSTACLE
            self.positions_xy[agent_id] = (x, y)
            return

        # Ground vehicles — check both obstacles and occupancy
        if self.obstacles[x + dx, y + dy] == self.config.FREE:
            if self.positions[x + dx, y + dy] == self.config.FREE:
                self.positions[x, y] = self.config.FREE
                x += dx
                y += dy
                self.positions[x, y] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x, y)

    # ------------------------------------------------------------------
    # Goal and active-status helpers
    # ------------------------------------------------------------------

    def on_goal(self, agent_id: int) -> bool:
        return self.positions_xy[agent_id] == self.finishes_xy[agent_id]

    def hide_agent(self, agent_id: int) -> bool:
        """Deactivate an agent and remove it from the occupancy grid."""
        if not self.is_active[agent_id]:
            return False
        self.is_active[agent_id] = False
        self.positions[self.positions_xy[agent_id]] = self.config.FREE
        return True

    def show_agent(self, agent_id: int) -> bool:
        """Reactivate a hidden agent.  Raises if the cell is already occupied."""
        if self.is_active[agent_id]:
            return False
        self.is_active[agent_id] = True
        if self.positions[self.positions_xy[agent_id]] == self.config.OBSTACLE:
            raise KeyError("The cell is already occupied.")
        self.positions[self.positions_xy[agent_id]] = self.config.OBSTACLE
        return True

    # ------------------------------------------------------------------
    # Cone masks and shared vision
    # ------------------------------------------------------------------

    @staticmethod
    def build_cone_masks(obs_radius, cone_half_angle_deg=45):
        """
        Precompute boolean visibility masks for each movement direction.

        Returns a dict mapping action index (0-4) to a (2r+1, 2r+1) bool array.
        Action 0 (stay / no previous direction) gives full visibility.
        """
        r = obs_radius
        size = 2 * r + 1
        dir_vectors = {
            0: None,       # stay → full view
            1: (-1, 0),    # North
            2: (1, 0),     # South
            3: (0, -1),    # West
            4: (0, 1),     # East
        }
        cos_thresh = math.cos(math.radians(cone_half_angle_deg))
        masks = {}
        for action, dv in dir_vectors.items():
            mask = np.zeros((size, size), dtype=bool)
            mask[r, r] = True  # own cell always visible
            if dv is None:
                mask[:] = True
                masks[action] = mask
                continue
            dx, dy = dv
            for i in range(size):
                for j in range(size):
                    ci, cj = i - r, j - r
                    if ci == 0 and cj == 0:
                        continue
                    mag = math.sqrt(ci * ci + cj * cj)
                    cos_angle = (dx * ci + dy * cj) / mag
                    if cos_angle >= cos_thresh:
                        mask[i, j] = True
            masks[action] = mask
        return masks

    def get_true_obstacles_for_agent(self, agent_id):
        """Return real obstacle patch regardless of agent type (for drone scouting)."""
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.obstacles[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)

    def build_global_agent_grid(self):
        """Build full-grid array encoding agent types for all active agents."""
        TYPE_VALUES = {
            AgentType.FRIENDLY_UAV: 1.0,
            AgentType.ENEMY_UAV:    2.0,
            AgentType.FRIENDLY_UGV: 3.0,
            AgentType.ENEMY_UGV:    4.0,
        }
        grid = np.zeros(self.obstacles.shape, dtype=np.float32)
        for idx, (ax, ay) in enumerate(self.positions_xy):
            if self.is_active[idx]:
                grid[ax, ay] = TYPE_VALUES.get(self.config.agent_types[idx], 0.5)
        return grid

    def build_team_visibility(self, team_indices, effective_directions, cone_masks):
        """Return a boolean mask of cells visible to any active team member."""
        r = self.config.obs_radius
        visible = np.zeros(self.obstacles.shape, dtype=bool)
        for idx in team_indices:
            if not self.is_active[idx]:
                continue
            x, y = self.positions_xy[idx]
            direction = effective_directions[idx]
            cone = cone_masks[direction]
            x0, x1 = x - r, x + r + 1
            y0, y1 = y - r, y + r + 1
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1 = min(visible.shape[0], x1)
            gy1 = min(visible.shape[1], y1)
            lx0, ly0 = gx0 - x0, gy0 - y0
            visible[gx0:gx1, gy0:gy1] |= cone[lx0:lx0 + (gx1 - gx0), ly0:ly0 + (gy1 - gy0)]
        return visible

    def get_patch_centered_on(self, agent_id, full_grid, pad_value=0.0):
        """Extract a (2r+1)x(2r+1) patch from *full_grid* centered on the agent."""
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        size = 2 * r + 1
        patch = np.full((size, size), pad_value, dtype=np.float32)
        x_min = max(0, x - r)
        x_max = min(full_grid.shape[0], x + r + 1)
        y_min = max(0, y - r)
        y_max = min(full_grid.shape[1], y + r + 1)
        lx0 = x_min - (x - r)
        ly0 = y_min - (y - r)
        patch[lx0:lx0 + (x_max - x_min), ly0:ly0 + (y_max - y_min)] = \
            full_grid[x_min:x_max, y_min:y_max]
        return patch
