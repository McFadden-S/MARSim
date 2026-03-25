"""
Multi-agent time-limit wrapper.

Extends Gymnasium's ``TimeLimit`` to set truncated flags for *all* agents
simultaneously when the step limit is reached (standard TimeLimit only
handles a single truncated bool).
"""

from gymnasium.wrappers import TimeLimit


class MultiTimeLimit(TimeLimit):
    """TimeLimit that broadcasts truncation to all agents."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = [True] * self.unwrapped.get_num_agents()
        return observation, reward, terminated, truncated, info

    def set_elapsed_steps(self, elapsed_steps: int):
        """Override elapsed steps (only valid for persistent environments)."""
        if not self.unwrapped.grid_config.persistent:
            raise ValueError("Cannot set elapsed steps for non-persistent environment.")
        assert elapsed_steps >= 0
        self._elapsed_steps = elapsed_steps
