"""
Position and target generation for MARSim.

Provides BFS-based connected-component analysis and random placement of
agent start/finish positions within connected free-space regions.
"""

import numpy as np

from MARSim.grid_config import GridConfig


def bfs(grid: np.ndarray, moves: tuple, start_id: int, free_cell: int) -> list[int]:
    """
    Label connected components of free cells using BFS and return their sizes.

    Modifies *grid* in-place: each free cell is replaced by its component ID.

    Args:
        grid: 2D array where *free_cell* marks traversable cells.
        moves: Tuple of (dx, dy) movement deltas.
        start_id: First component ID to assign (must be > max cell value).
        free_cell: Value that identifies unvisited free cells.

    Returns:
        List where ``components[id]`` is the number of cells in that component.
    """
    current_id = start_id
    components = [0] * start_id

    size_x, size_y = grid.shape

    for x in range(size_x):
        for y in range(size_y):
            if grid[x, y] != free_cell:
                continue

            grid[x, y] = current_id
            components.append(1)
            queue = [(x, y)]

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in moves:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size_x and 0 <= ny < size_y and grid[nx, ny] == free_cell:
                        grid[nx, ny] = current_id
                        components[current_id] += 1
                        queue.append((nx, ny))

            current_id += 1

    return components


def placing(order, components, grid, start_id, num_agents):
    """
    Assign start and finish positions for *num_agents* from the labelled grid.

    Agents are placed in cells belonging to components with at least 2 free
    cells (one for start, one for finish).

    Returns:
        (positions_xy, finishes_xy) — lists of (x, y) tuples.
    """
    requests = [[] for _ in range(len(components))]
    done_requests = 0
    positions_xy = []
    finishes_xy = [(-1, -1)] * num_agents

    for x, y in order:
        if grid[x, y] < start_id:
            continue

        comp_id = grid[x, y]
        grid[x, y] = 0

        if requests[comp_id]:
            agent_idx = requests[comp_id].pop()
            finishes_xy[agent_idx] = (x, y)
            done_requests += 1
            continue

        if len(positions_xy) >= num_agents:
            if done_requests >= num_agents:
                break
            continue

        if components[comp_id] >= 2:
            components[comp_id] -= 2
            requests[comp_id].append(len(positions_xy))
            positions_xy.append((x, y))

    return positions_xy, finishes_xy


def generate_from_possible_positions(grid_config: GridConfig):
    """
    Randomly select start/finish positions from pre-defined candidate lists.

    Raises:
        OverflowError: If there aren't enough candidates for *num_agents*.
    """
    if (
        len(grid_config.possible_agents_xy) < grid_config.num_agents
        or len(grid_config.possible_targets_xy) < grid_config.num_agents
    ):
        raise OverflowError(
            f"Not enough possible positions for {grid_config.num_agents} agents."
        )

    rng = np.random.default_rng(grid_config.seed)
    rng.shuffle(grid_config.possible_agents_xy)
    rng.shuffle(grid_config.possible_targets_xy)

    return (
        grid_config.possible_agents_xy[:grid_config.num_agents],
        grid_config.possible_targets_xy[:grid_config.num_agents],
    )


def generate_positions_and_targets_fast(obstacles: np.ndarray, grid_config: GridConfig):
    """
    Generate random start/finish positions using BFS component analysis.

    Uses connected-component labelling to ensure each agent's start and
    finish are in the same connected region.
    """
    grid = obstacles.copy()
    start_id = max(grid_config.FREE, grid_config.OBSTACLE) + 1

    components = bfs(grid, tuple(grid_config.MOVES), start_id, free_cell=grid_config.FREE)

    height, width = obstacles.shape
    order = [(x, y) for x in range(height) for y in range(width) if grid[x, y] >= start_id]
    np.random.default_rng(grid_config.seed).shuffle(order)

    return placing(order, components, grid, start_id, grid_config.num_agents)
