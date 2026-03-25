"""
Pygame-based real-time visualisation for MARSim.

Renders the grid world with supersampled anti-aliasing (SSAA).  Static
elements (obstacles, grid lines) are drawn once; dynamic elements (agents,
targets) are redrawn each frame and composited on top.
"""

import math

import numpy as np
import pygame
from MARSim.grid import Grid
from MARSim.grid_config import AgentType


class PygameRenderer:
    """
    Renders the MARSim grid world using Pygame with configurable SSAA.

    Usage::

        renderer = PygameRenderer(width=50, height=50, obs_radius=3)
        renderer.draw_static(obstacles_array)
        renderer.draw_state(agents_xy, targets_xy, agent_types, is_active,
                            effective_directions=[...])
        # ... each frame ...
        renderer.close()
    """

    # Style / palette
    scale_size: int = 100
    draw_start: int = 100
    stroke_width: int = 10

    obstacle_color: str   = "darkolivegreen"
    friendly_color: str   = "dodgerblue2"
    enemy_color: str      = "firebrick4"
    unknown_color: str    = "#5f008a"
    overlap_color: str    = "#8a2be2"   # blueviolet for overlap
    dead_color: str       = "#888888"   # grey for destroyed drones
    background_color: str = "lemonchiffon4"
    line_color: str       = "lemonchiffon3"

    # Geometry for vehicle glyphs
    r: int  = 35
    rx: int = 15

    def __init__(
        self,
        height: int,
        width: int,
        obs_radius: int,
        cell_px: int = 17,
        fps: int = 20,
        show_grid: bool = True,
        show_obs_range: bool = True,
        window_title: str = "MARSim – Live",
        ssaa: int = 5,
    ):
        self.height = int(height + 2)
        self.width = int(width + 2)
        self.wr = obs_radius - 1
        self.obs_radius = obs_radius
        self.show_obs_range = show_obs_range

        # Precompute cone masks for rendering
        self._cone_masks = Grid.build_cone_masks(obs_radius)

        # Logical (window) resolution
        self.cell_px_lo = int(cell_px)
        self.pad_lo = int(self.draw_start * (self.cell_px_lo / max(1, self.scale_size)))

        # High-res internal resolution for SSAA
        self.ssaa = max(1, int(ssaa))
        self.cell_px_hi = self.cell_px_lo * self.ssaa
        self.pad_hi = int(self.draw_start * (self.cell_px_hi / max(1, self.scale_size)))

        self.fps = int(fps)
        self.show_grid = show_grid
        self.window_title = window_title

        # Pygame state (initialised lazily in draw_static)
        self.clock = None
        self.screen = None
        self.static_hi = None
        self.static_lo = None
        self.frame_hi = None
        self._w = self._h = None
        self._hi_w = self._hi_h = None
        self._inited = False

        # Track last known positions for destroyed drones
        self._last_known_xy = {}

        # Pre-convert colour strings to pygame.Color
        self._COL_BG      = pygame.Color(self.background_color)
        self._COL_GRID    = pygame.Color(self.line_color)
        self._COL_OBS     = pygame.Color(self.obstacle_color)
        self._COL_FRI     = pygame.Color(self.friendly_color)
        self._COL_ENE     = pygame.Color(self.enemy_color)
        self._COL_UNK     = pygame.Color(self.unknown_color)
        self._COL_OVERLAP = pygame.Color(self.overlap_color)
        self._COL_DEAD    = pygame.Color(self.dead_color)

    # ------------------------------------------------------------------
    # Static layer (drawn once per episode)
    # ------------------------------------------------------------------

    def draw_static(self, obstacles):
        """Build the window and render the static background (grid + obstacles)."""
        pygame.init()

        self._w = self.width * self.cell_px_lo + 2 * self.pad_lo
        self._h = self.height * self.cell_px_lo + 2 * self.pad_lo
        self._hi_w = self.width * self.cell_px_hi + 2 * self.pad_hi
        self._hi_h = self.height * self.cell_px_hi + 2 * self.pad_hi

        self.screen = pygame.display.set_mode((self._w, self._h))
        pygame.display.set_caption(self.window_title)
        self.clock = pygame.time.Clock()

        self.static_hi = pygame.Surface((self._hi_w, self._hi_h)).convert()
        self.static_hi.fill(self._COL_BG)
        self.frame_hi = pygame.Surface(
            (self._hi_w, self._hi_h), flags=pygame.SRCALPHA,
        ).convert_alpha()

        self._draw_grid_hi()
        self._draw_obstacles_hi(obstacles)

        self.static_lo = pygame.transform.smoothscale(self.static_hi, (self._w, self._h))
        self._last_known_xy.clear()
        self._inited = True

    def _draw_grid_hi(self):
        if not self.show_grid:
            return
        sw = max(1, (self.stroke_width // 4) * self.ssaa)
        for j in range(self.width + 1):
            x = self.pad_hi + j * self.cell_px_hi
            pygame.draw.line(self.static_hi, self._COL_GRID,
                             (x, self.pad_hi),
                             (x, self.pad_hi + self.height * self.cell_px_hi), width=sw)
        for i in range(self.height + 1):
            y = self.pad_hi + i * self.cell_px_hi
            pygame.draw.line(self.static_hi, self._COL_GRID,
                             (self.pad_hi, y),
                             (self.pad_hi + self.width * self.cell_px_hi, y), width=sw)

    def _draw_obstacles_hi(self, obstacles):
        obs = obstacles[self.wr:-self.wr, self.wr:-self.wr] if self.wr > 0 else obstacles
        for i in range(self.height):
            for j in range(self.width):
                if obs[i][j]:
                    x = int(self.pad_hi + i * self.cell_px_hi)
                    y = int(self.pad_hi + j * self.cell_px_hi)
                    pygame.draw.rect(self.static_hi, self._COL_OBS,
                                     pygame.Rect(x, y, self.cell_px_hi, self.cell_px_hi))

    # ------------------------------------------------------------------
    # Dynamic layer (drawn each frame)
    # ------------------------------------------------------------------

    def draw_state(self, agents_xy, targets_xy, agent_types, is_active,
                   effective_directions=None):
        """Render all agents, observation cones, and target waypoints for one frame."""
        assert self._inited, "Call draw_static() before draw_state()."

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.blit(self.static_lo, (0, 0))
        self.frame_hi.fill((0, 0, 0, 0))

        # Update last-known positions for active agents
        for idx, (ai, aj) in enumerate(agents_xy):
            if is_active[idx]:
                self._last_known_xy[idx] = (ai, aj)

        # --- PASS 1: observation cone overlays ---
        if self.show_obs_range and effective_directions is not None:
            self._draw_cone_overlay(agents_xy, agent_types, is_active,
                                    effective_directions)

        # --- PASS 2: all agent glyphs (on top of cones) ---
        rpx_hi = max(2, int(self.r * (self.cell_px_hi / max(1, self.scale_size))))

        for idx in range(len(agents_xy)):
            atype = agent_types[idx] if agent_types is not None else None

            # Determine position (last-known for destroyed agents)
            if is_active[idx]:
                ai, aj = agents_xy[idx]
            elif idx in self._last_known_xy:
                ai, aj = self._last_known_xy[idx]
            else:
                continue

            cx = int(self.pad_hi + (ai - self.wr + 0.5) * self.cell_px_hi)
            cy = int(self.pad_hi + (aj - self.wr + 0.5) * self.cell_px_hi)

            if atype in (AgentType.FRIENDLY_UAV, AgentType.ENEMY_UAV):
                if is_active[idx]:
                    colour = self._COL_FRI if atype == AgentType.FRIENDLY_UAV else self._COL_ENE
                else:
                    colour = self._COL_DEAD
                self._draw_uav(cx, cy, rpx_hi, colour)
                # Target marker only for active agents
                if is_active[idx]:
                    ti, tj = targets_xy[idx]
                    ctx = int(self.pad_hi + (ti - self.wr + 0.5) * self.cell_px_hi)
                    cty = int(self.pad_hi + (tj - self.wr + 0.5) * self.cell_px_hi)
                    pygame.draw.circle(self.frame_hi, colour, (ctx, cty),
                                       rpx_hi, width=rpx_hi // 4)

            elif atype == AgentType.FRIENDLY_UGV:
                self._draw_ugv(cx, cy, rpx_hi)
                ti, tj = targets_xy[idx]
                ctx = int(self.pad_hi + (ti - self.wr + 0.5) * self.cell_px_hi)
                cty = int(self.pad_hi + (tj - self.wr + 0.5) * self.cell_px_hi)
                pygame.draw.circle(self.frame_hi, self._COL_FRI, (ctx, cty),
                                   rpx_hi, width=rpx_hi // 4)

            else:
                colour = self._COL_UNK if is_active[idx] else self._COL_DEAD
                pygame.draw.circle(self.frame_hi, colour, (cx, cy), rpx_hi)

        # --- Composite and display ---
        frame_lo = pygame.transform.smoothscale(self.frame_hi, (self._w, self._h))
        self.screen.blit(frame_lo, (0, 0))
        pygame.display.flip()
        if self.fps:
            self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # Cone overlay with overlap coloring
    # ------------------------------------------------------------------

    def _draw_cone_overlay(self, agents_xy, agent_types, is_active,
                           effective_directions):
        """Draw cell-level cone overlays with friendly/enemy/overlap coloring."""
        r = self.obs_radius
        friendly_cells = set()
        enemy_cells = set()

        for idx, (ai, aj) in enumerate(agents_xy):
            if not is_active[idx]:
                continue
            atype = agent_types[idx] if agent_types is not None else None
            if atype not in (AgentType.FRIENDLY_UAV, AgentType.ENEMY_UAV):
                continue

            direction = effective_directions[idx]
            cone = self._cone_masks[direction]

            for di in range(2 * r + 1):
                for dj in range(2 * r + 1):
                    if cone[di, dj]:
                        ci = ai + di - r
                        cj = aj + dj - r
                        if atype == AgentType.FRIENDLY_UAV:
                            friendly_cells.add((ci, cj))
                        else:
                            enemy_cells.add((ci, cj))

        overlap = friendly_cells & enemy_cells
        fri_only = friendly_cells - overlap
        ene_only = enemy_cells - overlap

        for ci, cj in fri_only:
            self._draw_cell_tint(ci, cj, self._COL_FRI, 30)
        for ci, cj in ene_only:
            self._draw_cell_tint(ci, cj, self._COL_ENE, 30)
        for ci, cj in overlap:
            self._draw_cell_tint(ci, cj, self._COL_OVERLAP, 45)

    def _draw_cell_tint(self, ci, cj, color, alpha):
        """Draw a semi-transparent tint over a single grid cell."""
        x = int(self.pad_hi + (ci - self.wr) * self.cell_px_hi)
        y = int(self.pad_hi + (cj - self.wr) * self.cell_px_hi)
        fill = pygame.Color(color.r, color.g, color.b, alpha)
        pygame.draw.rect(self.frame_hi, fill,
                         pygame.Rect(x, y, self.cell_px_hi, self.cell_px_hi))

    # ------------------------------------------------------------------
    # Glyph drawing helpers
    # ------------------------------------------------------------------

    def _draw_uav(self, cx: int, cy: int, rpx: int, colour):
        """Draw a quad-rotor UAV glyph (X-shaped arms + rotors)."""
        hub_r = int(rpx * 0.3)
        arm_w = max(2, rpx // 4)
        arm_outer = int(rpx * 1.0)
        rotor_r = int(rpx * 0.45)
        d_out = int(arm_outer / math.sqrt(2))

        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            pygame.draw.line(self.frame_hi, colour, (cx, cy),
                             (cx + sx * d_out, cy + sy * d_out), arm_w)
        pygame.draw.circle(self.frame_hi, colour, (cx, cy), hub_r)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            pygame.draw.circle(self.frame_hi, colour,
                               (cx + sx * d_out, cy + sy * d_out), rotor_r)

    def _draw_ugv(self, cx: int, cy: int, rpx: int):
        """Draw a ground vehicle glyph (rectangle body + wheels)."""
        body_w = int(1.8 * rpx)
        body_h = int(1.5 * rpx)
        pygame.draw.rect(
            self.frame_hi, self._COL_FRI,
            pygame.Rect(cx - body_w // 2, cy - body_h // 2, body_w, body_h),
            border_radius=rpx // 3,
        )
        wheel_r = rpx // 2
        wheel_offset_x = body_w // 3
        wheel_y = cy + body_h // 2
        for k in range(-1, 2):
            pygame.draw.circle(self.frame_hi, (30, 30, 30),
                               (cx + k * wheel_offset_x, wheel_y), wheel_r)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Shut down the Pygame window."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
