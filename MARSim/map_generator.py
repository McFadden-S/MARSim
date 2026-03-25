"""
Procedural battlefield map generator for MARSim.

Generates corridor-style maps using a "treeline" architecture: vertical
columns of rectangular fields separated by walls, with entrances carved
between adjacent fields and columns to ensure connectivity.

After generation, a BFS connectivity check guarantees at least one
traversable path from the left third to the right third of the map
(the UGV spawn zone to its target zone).

Usage::

    bf = Battlefield()          # 50x50 default
    bf = Battlefield(width=80, height=80, field_width=12)
    env = make_MARSim(GridConfig(map=bf.map))
"""

import random
from collections import deque


class Battlefield:
    """
    Generates a 2D grid map where 0 = free space and 1 = wall.

    The map is built by sweeping left-to-right in fixed-width columns.
    Within each column, rectangular "fields" of random height are carved
    out.  Entrances link fields vertically (within a column) and
    horizontally (between adjacent columns).  After all fields are placed,
    remaining walls are randomly removed until *target_open_ratio* of
    cells are free, ensuring the map is largely traversable.

    A final connectivity pass guarantees a path from the left third to
    the right third of the map.

    Attributes:
        map: list[list[int]] — the generated grid (0 = free, 1 = wall).
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        field_width: int = 10,
        min_field_height: int = 5,
        max_field_height: int = 15,
        target_open_ratio: float = 0.8,
    ):
        self.width = width
        self.height = height
        self.field_width = field_width
        self.min_field_height = min_field_height
        self.max_field_height = max_field_height
        self.target_open_ratio = target_open_ratio

        # Initialise as all walls, then carve out fields
        self.map: list[list[int]] = [[1] * width for _ in range(height)]
        self._generate_fields()
        self._ensure_left_right_path()

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------

    def _generate_fields(self):
        """Sweep left-to-right, carving field columns and linking them."""
        prev_column: list[tuple[int, int]] = []
        x = 0

        while x < self.width:
            column_height = 0
            y_positions: list[tuple[int, int]] = []

            # Stack fields vertically within this column
            while column_height < self.height:
                remaining = self.height - column_height

                if remaining < self.min_field_height:
                    # Edge case: leftover strip too small for a proper field
                    y_positions.append((column_height, self.height))
                    column_height = self.height
                    break

                fh = random.randint(
                    self.min_field_height,
                    min(self.max_field_height, remaining),
                )
                y_positions.append((column_height, column_height + fh))
                column_height += fh + 1  # +1 leaves a wall row between fields

            # Carve each field and add vertical connections
            for i, (y1, y2) in enumerate(y_positions):
                enclosed = (y2 - y1) >= self.min_field_height
                self._carve_rectangle(
                    x, y1, min(x + self.field_width, self.width), y2,
                    enclosed=enclosed,
                )

                # Vertical entrance between consecutive fields in this column
                if i > 0:
                    prev_y1, prev_y2 = y_positions[i - 1]
                    overlap_start = max(y1, prev_y1)
                    overlap_end = min(y2, prev_y2)
                    if overlap_end - overlap_start > 2:
                        ey = random.randint(overlap_start + 1, overlap_end - 2)
                        self.map[ey][x] = 0

            # Horizontal connections to the previous column
            if prev_column:
                for y1, y2 in y_positions:
                    for prev_y1, prev_y2 in prev_column:
                        overlap_start = max(y1, prev_y1)
                        overlap_end = min(y2, prev_y2)
                        if overlap_end - overlap_start > 2:
                            ey = random.randint(overlap_start + 1, overlap_end - 2)
                            self.map[ey][x] = 0
                            break

            prev_column = y_positions
            x += self.field_width + 1  # +1 for the wall column

        self._fill_open_space()

    def _carve_rectangle(self, x1: int, y1: int, x2: int, y2: int, *, enclosed: bool = True):
        """Clear the interior of [x1, x2) x [y1, y2); optionally keep boundary walls."""
        # Clear interior
        for y in range(y1 + 1, min(y2 - 1, self.height - 1)):
            for x in range(x1 + 1, min(x2 - 1, self.width - 1)):
                self.map[y][x] = 0

        if enclosed:
            # Reinforce boundary walls
            for y in range(y1, y2):
                if 0 <= y < self.height:
                    if 0 <= x1 < self.width:
                        self.map[y][x1] = 1
                    if 0 <= x2 - 1 < self.width:
                        self.map[y][x2 - 1] = 1
            for x in range(x1, x2):
                if 0 <= y1 < self.height:
                    self.map[y1][x] = 1
                if 0 <= y2 - 1 < self.height:
                    self.map[y2 - 1][x] = 1

    def _fill_open_space(self):
        """Randomly remove walls until *target_open_ratio* of cells are free."""
        total = self.width * self.height
        open_cells = sum(c == 0 for row in self.map for c in row)
        target_open = int(total * self.target_open_ratio)

        wall_positions = [
            (y, x)
            for y in range(self.height)
            for x in range(self.width)
            if self.map[y][x] == 1
        ]
        random.shuffle(wall_positions)

        for y, x in wall_positions:
            if open_cells >= target_open:
                break
            self.map[y][x] = 0
            open_cells += 1

    # ------------------------------------------------------------------
    # Path guarantee
    # ------------------------------------------------------------------

    def _ensure_left_right_path(self):
        """
        Guarantee at least one path from the left third to the right third.

        Uses BFS from every free cell in the left third.  If no cell in the
        right third is reachable, carve a corridor along the BFS parent
        chain toward the right, breaking through walls as needed.
        """
        left_bound = self.width // 3
        right_bound = 2 * self.width // 3

        # Collect free cells in the left third as BFS seeds
        seeds = []
        for r in range(self.height):
            for c in range(left_bound):
                if self.map[r][c] == 0:
                    seeds.append((r, c))

        if not seeds:
            # No free cells on the left at all — carve a horizontal line
            mid_r = self.height // 2
            for c in range(self.width):
                self.map[mid_r][c] = 0
            return

        # BFS from all left-third free cells
        visited = set(seeds)
        parent = {}
        queue = deque(seeds)
        reached_right = False

        while queue:
            r, c = queue.popleft()
            if c >= right_bound:
                reached_right = True
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    if self.map[nr][nc] == 0:
                        queue.append((nr, nc))
                    else:
                        # Wall — add to queue with high priority (BFS still
                        # explores it, but we prefer free cells first so we
                        # append walls to the right end)
                        queue.append((nr, nc))

        if reached_right:
            # A path exists through mixed free/wall cells — carve it
            cur = (r, c)
            while cur in parent:
                cr, cc = cur
                self.map[cr][cc] = 0
                cur = parent[cur]
            return

        # BFS couldn't reach the right third at all (shouldn't happen with
        # wall traversal above, but as a fallback carve a horizontal line)
        mid_r = self.height // 2
        for c in range(self.width):
            self.map[mid_r][c] = 0

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def as_string(self) -> str:
        """Return the map as a flat string ('.' = free, '#' = wall)."""
        return "".join("." if c == 0 else "#" for row in self.map for c in row)

    def display(self):
        """Print the map to stdout."""
        for row in self.map:
            print("".join("." if c == 0 else "#" for c in row))


if __name__ == "__main__":
    bf = Battlefield()
    bf.display()
