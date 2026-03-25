"""
Episode history recording wrapper.

Wraps a MARSim environment to record a sparse log of agent state changes
over the course of an episode.  The compressed history can later be
decompressed into per-step trajectories for replay and analysis.
"""

from gymnasium import Wrapper


class AgentState:
    """Snapshot of a single agent's state at a given timestep."""

    def __init__(self, x: int, y: int, tx: int, ty: int, step: int, active: bool):
        self.x = x
        self.y = y
        self.tx = tx
        self.ty = ty
        self.step = step
        self.active = active

    def get_xy(self) -> tuple[int, int]:
        return self.x, self.y

    def get_target_xy(self) -> tuple[int, int]:
        return self.tx, self.ty

    def is_active(self) -> bool:
        return self.active

    def get_step(self) -> int:
        return self.step

    def __eq__(self, other):
        return (
            self.x == other.x
            and self.y == other.y
            and self.tx == other.tx
            and self.ty == other.ty
            and self.active == other.active
        )

    def __str__(self):
        return str([self.x, self.y, self.tx, self.ty, self.step, self.active])


class PersistentWrapper(Wrapper):
    """
    Records per-agent state changes during an episode.

    Only state *transitions* are stored (not every step), making the
    history compact.  Use ``decompress_history`` to expand into
    per-step trajectories.
    """

    def __init__(self, env, xy_offset=None):
        super().__init__(env)
        self._step = None
        self._agent_states = None
        self._xy_offset = xy_offset

    def step(self, action):
        result = self.env.step(action)
        self._step += 1
        for idx in range(self.get_num_agents()):
            state = self._get_agent_state(self.grid, idx)
            if state != self._agent_states[idx][-1]:
                self._agent_states[idx].append(state)
        return result

    def _get_agent_state(self, grid, agent_idx: int) -> AgentState:
        x, y = grid.positions_xy[agent_idx]
        tx, ty = grid.finishes_xy[agent_idx]
        active = grid.is_active[agent_idx]
        if self._xy_offset:
            x += self._xy_offset
            y += self._xy_offset
            tx += self._xy_offset
            ty += self._xy_offset
        return AgentState(x, y, tx, ty, self._step, active)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._step = 0
        self._agent_states = [
            [self._get_agent_state(self.grid, i)]
            for i in range(self.get_num_agents())
        ]
        return result

    @staticmethod
    def agent_state_to_full_list(agent_states: list[AgentState], num_steps: int) -> list[AgentState]:
        """Expand a sparse state list into a per-step trajectory."""
        result = []
        state_idx = 0
        for step in range(num_steps + 1):
            if state_idx < len(agent_states) - 1 and agent_states[state_idx + 1].step == step:
                state_idx += 1
            result.append(agent_states[state_idx])
        return result

    @classmethod
    def decompress_history(cls, history: list[list[AgentState]]) -> list[list[AgentState]]:
        """Decompress sparse histories for all agents into full trajectories."""
        max_steps = max(states[-1].step for states in history)
        return [cls.agent_state_to_full_list(states, max_steps) for states in history]

    def get_full_history(self) -> list[list[AgentState]]:
        return [self.agent_state_to_full_list(states, self._step) for states in self._agent_states]

    def get_history(self) -> list[list[AgentState]]:
        return self._agent_states
