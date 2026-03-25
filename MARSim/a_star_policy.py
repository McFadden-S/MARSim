"""
A* pathfinding policy for UGV navigation in MARSim.

Provides:
    - ``GridMemory``: Dynamically growing agent-centric memory with three
      cell states (UNKNOWN / FREE / BLOCK).
    - ``AStarAgent``: Complete UGV controller that incrementally learns the
      map, plans paths with A*, and falls back to frontier exploration.
    - ``BatchAStarAgent``: Multi-agent wrapper maintaining per-index state.
"""

import numpy as np
from heapq import heappush, heappop
from collections import deque

from MARSim.grid_config import GridConfig

# Memory encoding
UNKNOWN = -1
FREE    = 0
BLOCK   = 1


class GridMemory:
    """
    Dynamically growing agent-centric memory with states: UNKNOWN/FREE/BLOCK.
    Local obstacle windows are stamped at the agent's world (x,y).
    """
    def __init__(self, start_radius: int = 64):
        side = start_radius * 2 + 1
        self._mem = np.full((side, side), UNKNOWN, dtype=np.int8)

    @staticmethod
    def _blit_centered(dst: np.ndarray, cx: int, cy: int, src: np.ndarray) -> bool:
        h, w = src.shape
        rx, ry = h // 2, w // 2
        x0, x1 = cx - rx, cx + rx + 1
        y0, y1 = cy - ry, cy + ry + 1
        if x0 < 0 or y0 < 0 or x1 > dst.shape[0] or y1 > dst.shape[1]:
            return False
        dst[x0:x1, y0:y1] = src
        return True

    def _grow(self):
        old = self._mem
        H, W = old.shape
        new = np.full((H * 2 + 1, W * 2 + 1), UNKNOWN, dtype=np.int8)
        cx, cy = new.shape[0] // 2, new.shape[1] // 2
        x0, y0 = cx - H // 2, cy - W // 2
        new[x0:x0 + H, y0:y0 + W] = old
        self._mem = new

    def update(self, x: int, y: int, obstacles_patch: np.ndarray):
        """
        obstacles_patch: 1=obstacle, 0=free. Any other values are ignored as FREE.
        """
        patch = (obstacles_patch == 1).astype(np.int8)  # 1=BLOCK, 0=FREE
        while True:
            cx, cy = self._mem.shape[0] // 2, self._mem.shape[1] // 2
            if self._blit_centered(self._mem, cx + x, cy + y, patch):
                break
            self._grow()

    def _inside(self, x: int, y: int) -> bool:
        rX, rY = self._mem.shape[0] // 2, self._mem.shape[1] // 2
        return (-rX <= x <= rX) and (-rY <= y <= rY)

    def get(self, x: int, y: int) -> int:
        if not self._inside(x, y):
            return UNKNOWN
        cx, cy = self._mem.shape[0] // 2, self._mem.shape[1] // 2
        return int(self._mem[cx + x, cy + y])

    def is_block(self, x: int, y: int) -> bool:
        return self.get(x, y) == BLOCK

    def is_free(self, x: int, y: int) -> bool:
        return self.get(x, y) == FREE

    def is_unknown(self, x: int, y: int) -> bool:
        return self.get(x, y) == UNKNOWN

    def neighbors(self, x: int, y: int, moves):
        """Yield ((nx, ny), action_index) for all directions except stay."""
        for a, (dx, dy) in enumerate(moves):
            if a == 0:  # skip 'stay'
                continue
            yield (x + dx, y + dy), a


def manhattan(a, b):
    (ax, ay), (bx, by) = a, b
    return abs(ax - bx) + abs(ay - by)


def a_star_on_memory(start, goal, grid: GridMemory, moves, unknown_penalty=0.25,
                     max_expand=20_000):
    """
    A* over memory: BLOCK impassable, FREE cost=1, UNKNOWN cost=1+penalty.
    Returns [start,...,goal] or [] if no path found within expand budget.
    """
    if start == goal:
        return [start]

    g = {start: 0.0}
    parent = {}
    eps = 1e-4

    def h(n): return manhattan(n, goal)

    def step_cost(nxt):
        if grid.is_block(*nxt):
            return np.inf
        return 1.0 + (unknown_penalty if grid.is_unknown(*nxt) else 0.0)

    open_heap = []
    heappush(open_heap, (h(start), 0.0, start))
    visited = set()
    expands = 0

    while open_heap and expands < max_expand:
        _, gcur, cur = heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        expands += 1

        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return list(reversed(path))

        cx, cy = cur
        for nb, _ in grid.neighbors(cx, cy, moves):
            sc = step_cost(nb)
            if np.isinf(sc):
                continue
            ng = gcur + sc
            if ng + 1e-12 < g.get(nb, np.inf):
                g[nb] = ng
                parent[nb] = cur
                f = ng + (1.0 + eps) * h(nb)
                heappush(open_heap, (f, ng, nb))

    return []


def nearest_frontier_step(agent_xy, goal_xy, grid: GridMemory, moves, bfs_cap=2000):
    """
    ONE step toward the best frontier (known-FREE adjacent to UNKNOWN).
    Returns next coordinate or None if no frontier reachable.
    """
    q = deque([agent_xy])
    seen = {agent_xy}
    frontiers = []
    steps = 0

    def is_frontier(x, y):
        if not grid.is_free(x, y):
            return False
        for (nx, ny), _ in grid.neighbors(x, y, moves):
            if grid.is_unknown(nx, ny):
                return True
        return False

    while q and steps < bfs_cap:
        steps += 1
        x, y = q.popleft()
        if is_frontier(x, y):
            frontiers.append((x, y))
        for (nx, ny), _ in grid.neighbors(x, y, moves):
            if (nx, ny) in seen:
                continue
            if grid.is_free(nx, ny):
                seen.add((nx, ny))
                q.append((nx, ny))

    if not frontiers:
        return None

    frontiers.sort(key=lambda c: (manhattan(c, goal_xy), 0.1 * manhattan(c, agent_xy)))
    best = frontiers[0]
    path = a_star_on_memory(agent_xy, best, grid, moves, unknown_penalty=0.0)
    if len(path) >= 2:
        return path[1]
    return None


class AStarAgent:
    """
    UGV navigator that:
      - learns a 3-state map,
      - plans with A* (unknown traversable with penalty),
      - falls back to frontier-seeking,
      - NEVER returns STOP for valid targets; will force a move (left-hand rule) to keep searching.
    """
    def __init__(self, seed=0, unknown_penalty=0.25, tabu_len=6, reverse_margin=0.75,
                 allow_stop_when_truly_boxed=False):
        """
        reverse_margin: allow immediate reverse only if it's clearly better (smaller => more permissive).
        allow_stop_when_truly_boxed: if False, we still output a non-zero action even if physically boxed.
        """
        # MOVES: [[0,0], [-1,0], [1,0], [0,-1], [0,1]]
        self._moves = [tuple(v) for v in GridConfig().MOVES]
        self._rev = {self._moves[i]: i for i in range(len(self._moves))}
        self._gm = GridMemory()
        self._rng = np.random.default_rng(seed)

        self._last_xy = None
        self._last_action = None
        self._tabu = deque(maxlen=int(tabu_len))

        self.unknown_penalty = float(unknown_penalty)
        self.reverse_margin = float(reverse_margin)
        self.allow_stop_when_truly_boxed = bool(allow_stop_when_truly_boxed)

        # For left-hand rule we need an orientation; derive from last action.
        # Order of direction indices for our MOVES (1:N, 2:S, 3:W, 4:E)
        self._dir_order = [1, 4, 2, 3]  # N, E, S, W (clockwise)

    def clear_state(self):
        self._gm = GridMemory()
        self._last_xy = None
        self._last_action = None
        self._tabu.clear()

    def update_from_drones(self, drone_observations):
        """Feed obstacle patches from friendly drones into grid memory.

        Args:
            drone_observations: list of ((x, y), obstacles_patch) tuples.
        """
        for (x, y), patch in drone_observations:
            self._gm.update(x, y, patch.astype(np.int8))

    def _strict_action_from_step(self, cur, nxt) -> int:
        dx, dy = (nxt[0] - cur[0], nxt[1] - cur[1])
        key = (dx, dy)
        if key in self._rev:
            return self._rev[key]
        # robust fallback: nearest by L1 among non-zero moves
        best_a, best_d = None, None
        for a, (mx, my) in enumerate(self._moves):
            if a == 0:
                continue
            d = abs(mx - dx) + abs(my - dy)
            if best_d is None or d < best_d:
                best_a, best_d = a, d
        if best_a is None:
            # Should never happen with your MOVES
            return 1  # arbitrary non-zero
        return best_a

    def _neighbor_actions(self, cur_xy):
        acts = []
        x, y = cur_xy
        for a, (dx, dy) in enumerate(self._moves):
            if a == 0:
                continue
            nx, ny = x + dx, y + dy
            if not self._gm.is_block(nx, ny):
                acts.append(a)
        return acts

    def _avoid_instant_reverse(self, cur_xy, candidates, goal_xy):
        if not candidates or self._last_action is None or self._last_action == 0:
            return candidates

        dx, dy = self._moves[self._last_action]
        rev_vec = (-dx, -dy)
        rev_a = self._rev.get(rev_vec, None)
        if rev_a is None:
            return candidates

        def f_of(a):
            mx, my = self._moves[a]
            nxt = (cur_xy[0] + mx, cur_xy[1] + my)
            return manhattan(nxt, goal_xy)

        scored = sorted(candidates, key=f_of)
        best = scored[0]
        if best != rev_a:
            return scored

        if len(scored) == 1:
            return scored

        second = scored[1]
        if f_of(best) <= f_of(second) * self.reverse_margin:
            return scored

        reordered = [a for a in scored if a != rev_a] + [rev_a]
        return reordered

    def _forced_move(self, cur_xy):
        """
        Left-hand wall-following preference to ensure a non-zero action even when
        candidates appear empty (e.g., all neighbors BLOCK in memory).
        """
        # establish an orientation from last action; default to 'north' (1)
        if self._last_action in (1, 2, 3, 4):
            facing = self._last_action
        else:
            facing = 1  # N

        # Define left, forward, right, back relative to current facing on _dir_order cycle
        order = self._dir_order
        idx = order.index(facing)
        prefer = [order[(idx - 1) % 4], order[idx], order[(idx + 1) % 4], order[(idx + 2) % 4]]

        # Try each in order; if memory says BLOCK everywhere, still return the first (non-zero)
        x, y = cur_xy
        for a in prefer:
            dx, dy = self._moves[a]
            nx, ny = x + dx, y + dy
            if not self._gm.is_block(nx, ny):
                return a

        # Truly boxed (all BLOCK in memory): still output a non-zero to "keep searching"
        return prefer[0]

    def _exploration_step(self, cur_xy, goal_xy):
        nxt = nearest_frontier_step(cur_xy, goal_xy, self._gm, moves=self._moves)
        if nxt is not None:
            return self._strict_action_from_step(cur_xy, nxt)

        candidates = self._neighbor_actions(cur_xy)  # FREE or UNKNOWN; excludes BLOCK
        if candidates:
            candidates = self._avoid_instant_reverse(cur_xy, candidates, goal_xy)

            def score(a):
                mx, my = self._moves[a]
                nx, ny = cur_xy[0] + mx, cur_xy[1] + my
                s = 0.0
                s += 0.0 if self._gm.is_free(nx, ny) else 0.5  # UNKNOWN slightly worse
                s += 0.05 * manhattan((nx, ny), goal_xy)
                s += 0.2 if (nx, ny) in self._tabu else 0.0
                return s

            return min(candidates, key=score)

        # No candidates (memory thinks all neighbors are BLOCK) -> force a move anyway
        return self._forced_move(cur_xy)

    def act(self, obs):
        """
        Returns a non-zero action whenever target is valid (!= (-1, -1)).
        """
        xy = (int(obs['xy'][0]), int(obs['xy'][1]))
        tgt = (int(obs['target_xy'][0]), int(obs['target_xy'][1]))
        obstacles = obs['obstacles'].astype(np.int8)

        # If target is dummy (-1,-1) (e.g., for UAVs), remain idle (or change to patrol if desired)
        if tgt == (-1, -1):
            self._last_action = 0
            self._last_xy = xy
            self._tabu.append(xy)
            return 0

        # Teleport/new episode detection
        if self._last_xy is not None and manhattan(self._last_xy, xy) > 1:
            self.clear_state()

        # Update memory
        self._gm.update(*xy, obstacles)

        # Plan
        path = a_star_on_memory(xy, tgt, self._gm, self._moves, unknown_penalty=self.unknown_penalty)

        if len(path) >= 2:
            nxt = path[1]
            act = self._strict_action_from_step(xy, nxt)
        else:
            act = self._exploration_step(xy, tgt)

        # Enforce: never STOP unless explicitly allowed and truly boxed (we still avoid 0 by default)
        if act == 0 and not self.allow_stop_when_truly_boxed:
            act = self._forced_move(xy)

        # Bookkeeping
        self._tabu.append(xy)
        self._last_action = act
        self._last_xy = xy
        return int(act)


class BatchAStarAgent:
    """Keeps one AStarAgent per index for batched calls."""
    def __init__(self, seed=0, **agent_kwargs):
        self._agents = {}
        self._seed = int(seed)
        self._kwargs = dict(agent_kwargs)

    def act(self, observations):
        seq = observations[0] if (len(observations) == 2) else observations
        actions = []
        for idx, obs in enumerate(seq):
            if idx not in self._agents:
                self._agents[idx] = AStarAgent(seed=self._seed + idx, **self._kwargs)
            actions.append(self._agents[idx].act(obs))
        return actions

    def reset_states(self):
        self._agents.clear()
