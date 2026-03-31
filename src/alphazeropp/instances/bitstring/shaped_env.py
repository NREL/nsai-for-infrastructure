"""
Gymnasium wrapper for potential-based reward shaping of BitStringGym.

Supports two reward modes:
  - "dense_potential": r_t = (f(s_{t+1}) - f(s_t)) / N
  - "sparse_pm1":     passes through the original reward from BitStringGym

Also supports cycling through a list of frozen initial states for
deterministic evaluation.
"""

import gymnasium as gym
import numpy as np
from typing import Callable, Optional, Sequence

from alphazeropp.instances.bitstring.game import BitStringGym


def make_initial_states(
    seed: int,
    n_sites: int,
    n_states: int,
    n_ones: int = 2,
) -> list[np.ndarray]:
    """Generate a list of deterministic initial bitstring states.

    Uses its own numpy Generator (not the global RNG).

    Args:
        seed: Random seed for reproducibility.
        n_sites: Length of bitstring.
        n_states: Number of initial states to generate.
        n_ones: Number of bits initially set to 1 (matching BitStringGym default).

    Returns:
        List of np.float32 arrays, each of shape (n_sites,).
    """
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        s = np.zeros(n_sites, dtype=np.float32)
        ones_idx = rng.choice(n_sites, size=n_ones, replace=False)
        s[ones_idx] = 1.0
        states.append(s)
    return states


class ShapedBitStringGym(gym.Wrapper):
    """Wraps BitStringGym to provide potential-based shaped rewards.

    Args:
        env: A BitStringGym instance.
        potential_fn: A function mapping state array -> int.
        reward_mode: One of "dense_potential" or "sparse_pm1".
        frozen_states: Optional list of initial states. When provided,
            reset() cycles through them instead of randomizing.
    """

    def __init__(
        self,
        env: BitStringGym,
        potential_fn: Callable[[np.ndarray], int],
        reward_mode: str = "dense_potential",
        frozen_states: Optional[Sequence[np.ndarray]] = None,
    ):
        super().__init__(env)
        assert reward_mode in ("dense_potential", "sparse_pm1"), \
            f"Unknown reward_mode: {reward_mode}"

        self.potential_fn = potential_fn
        self.reward_mode = reward_mode
        self.frozen_states = frozen_states
        self._frozen_idx = 0
        self._prev_potential: int = 0

    def __getattr__(self, name):
        """Proxy attribute access to the wrapped BitStringGym env.

        Gymnasium 1.0 Wrapper does not proxy custom attributes automatically.
        BitStringGame needs n_sites, step_count, state from the underlying env.

        The 'env' guard prevents infinite recursion during copy.deepcopy:
        deepcopy creates an empty object (no __dict__ yet), then checks
        hasattr(y, '__setstate__'), which triggers __getattr__ → self.env
        → __getattr__('env') → infinite loop.
        """
        if "env" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, **kwargs):
        if self.frozen_states is not None:
            # Call env.reset to initialize internal state, then overwrite
            obs, info = self.env.reset(**kwargs)
            state = self.frozen_states[self._frozen_idx % len(self.frozen_states)]
            self.env.state = state.copy()
            self.env.step_count = 0
            obs = state.copy()
            self._frozen_idx += 1
        else:
            obs, info = self.env.reset(**kwargs)
            info = {}

        self._prev_potential = self.potential_fn(obs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        if self.reward_mode == "dense_potential":
            new_potential = self.potential_fn(obs)
            reward = (new_potential - self._prev_potential) / self.env.n_sites
            info["potential_prev"] = self._prev_potential
            info["potential_curr"] = new_potential
            self._prev_potential = new_potential
        else:
            # sparse_pm1: pass through original reward
            reward = original_reward

        return obs, reward, terminated, truncated, info

    def reset_frozen_index(self):
        """Reset the frozen state cycling index to 0."""
        self._frozen_idx = 0
