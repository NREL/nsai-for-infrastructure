"""
Tests for potential-based shaped rewards on BitStringGym.

Two main categories:
1. Telescoping identity: cumulative shaped reward == (f(x_T) - f(x_0)) / N exactly.
2. Greedy oracle: greedy policy for OneMax achieves solve_rate=1.0 on frozen states.
"""

import numpy as np
import pytest

from alphazeropp.instances.bitstring.potentials import (
    onemax, leading_ones, binval, POTENTIAL_REGISTRY,
)
from alphazeropp.instances.bitstring.game import BitStringGym
from alphazeropp.instances.bitstring.shaped_env import (
    ShapedBitStringGym, make_initial_states,
)


# ---------------------------------------------------------------------------
# Potential function unit tests
# ---------------------------------------------------------------------------

class TestPotentials:
    def test_onemax_basic(self):
        assert onemax(np.array([0, 0, 0])) == 0
        assert onemax(np.array([1, 1, 1])) == 3
        assert onemax(np.array([1, 0, 1, 0, 1])) == 3

    def test_leading_ones_basic(self):
        assert leading_ones(np.array([1, 1, 1, 0, 1])) == 3
        assert leading_ones(np.array([0, 1, 1, 1])) == 0
        assert leading_ones(np.array([1, 1, 1, 1])) == 4
        assert leading_ones(np.array([0])) == 0

    def test_binval_basic(self):
        assert binval(np.array([1, 0, 1])) == 5
        assert binval(np.array([0, 0, 0])) == 0
        assert binval(np.array([1, 1, 1])) == 7
        assert binval(np.array([1, 0, 0, 0])) == 8

    def test_registry_contains_all(self):
        assert set(POTENTIAL_REGISTRY.keys()) == {"onemax", "leading_ones", "binval"}


# ---------------------------------------------------------------------------
# make_initial_states tests
# ---------------------------------------------------------------------------

class TestMakeInitialStates:
    def test_deterministic(self):
        s1 = make_initial_states(seed=42, n_sites=10, n_states=5)
        s2 = make_initial_states(seed=42, n_sites=10, n_states=5)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_shape_and_dtype(self):
        states = make_initial_states(seed=0, n_sites=10, n_states=3)
        assert len(states) == 3
        for s in states:
            assert s.shape == (10,)
            assert s.dtype == np.float32
            assert int(np.sum(s)) == 2  # n_ones=2 default

    def test_different_seeds_differ(self):
        s1 = make_initial_states(seed=0, n_sites=10, n_states=5)
        s2 = make_initial_states(seed=1, n_sites=10, n_states=5)
        any_diff = any(not np.array_equal(a, b) for a, b in zip(s1, s2))
        assert any_diff


# ---------------------------------------------------------------------------
# Telescoping identity tests
# ---------------------------------------------------------------------------

class TestTelescoping:
    """Verify: sum_t r_t == (f(x_T) - f(x_0)) / N exactly."""

    @pytest.mark.parametrize("potential_name", ["onemax", "leading_ones", "binval"])
    @pytest.mark.parametrize("n_sites", [6, 10, 20])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_telescoping_integer(self, potential_name, n_sites, seed):
        """Play with random actions and check telescoping via integer arithmetic."""
        potential_fn = POTENTIAL_REGISTRY[potential_name]

        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(
            env=base_env,
            potential_fn=potential_fn,
            reward_mode="dense_potential",
        )

        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        f_initial = potential_fn(obs)

        total_int_diff = 0
        done = False
        while not done:
            action = int(rng.integers(0, n_sites))
            obs, reward, terminated, truncated, info = env.step(action)
            total_int_diff += info["potential_curr"] - info["potential_prev"]
            done = terminated or truncated

        f_final = potential_fn(obs)
        assert total_int_diff == f_final - f_initial, \
            f"Telescoping failed: {total_int_diff} != {f_final} - {f_initial}"

    @pytest.mark.parametrize("potential_name", ["onemax", "leading_ones", "binval"])
    def test_telescoping_float(self, potential_name):
        """Verify float reward sum matches (f(x_T) - f(x_0)) / N within epsilon."""
        potential_fn = POTENTIAL_REGISTRY[potential_name]
        n_sites = 10

        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(
            env=base_env,
            potential_fn=potential_fn,
            reward_mode="dense_potential",
        )

        rng = np.random.default_rng(42)
        obs, _ = env.reset(seed=42)
        f_initial = potential_fn(obs)

        total_reward = 0.0
        done = False
        while not done:
            action = int(rng.integers(0, n_sites))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        f_final = potential_fn(obs)
        expected = (f_final - f_initial) / n_sites
        assert abs(total_reward - expected) < 1e-12, \
            f"Float telescoping error: {abs(total_reward - expected)}"


# ---------------------------------------------------------------------------
# Greedy oracle tests
# ---------------------------------------------------------------------------

class TestGreedyOracle:
    """A greedy policy for OneMax should achieve solve_rate=1.0."""

    @staticmethod
    def greedy_onemax_action(obs: np.ndarray) -> int:
        """Greedy oracle: flip the first 0-bit to 1."""
        zero_indices = np.where(obs == 0.0)[0]
        if len(zero_indices) == 0:
            return 0
        return int(zero_indices[0])

    def test_greedy_oracle_solve_rate(self):
        """Greedy oracle must solve 100% of frozen initial states for OneMax."""
        n_sites = 10
        n_states = 50

        frozen = make_initial_states(seed=0, n_sites=n_sites, n_states=n_states)

        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(
            env=base_env,
            potential_fn=onemax,
            reward_mode="dense_potential",
            frozen_states=frozen,
        )

        solves = 0
        for _ in range(n_states):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.greedy_onemax_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if np.all(obs == 1.0):
                solves += 1

        assert solves / n_states == 1.0, \
            f"Greedy oracle solve_rate={solves / n_states}, expected 1.0"

    def test_greedy_oracle_step_efficiency(self):
        """Greedy oracle should solve in exactly (n_sites - n_ones) steps."""
        n_sites = 10
        n_ones = 2

        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(
            env=base_env,
            potential_fn=onemax,
            reward_mode="dense_potential",
        )

        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            steps = 0
            done = False
            while not done:
                action = self.greedy_onemax_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated

            assert steps == n_sites - n_ones, \
                f"Greedy should solve in {n_sites - n_ones} steps, took {steps}"


# ---------------------------------------------------------------------------
# Frozen states cycling tests
# ---------------------------------------------------------------------------

class TestFrozenStates:
    def test_cycling_through_frozen_states(self):
        n_sites = 6
        frozen = make_initial_states(seed=0, n_sites=n_sites, n_states=3)

        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(
            env=base_env,
            potential_fn=onemax,
            reward_mode="dense_potential",
            frozen_states=frozen,
        )

        for cycle in range(2):
            env.reset_frozen_index()
            for i in range(3):
                obs, _ = env.reset()
                np.testing.assert_array_equal(obs, frozen[i])

    def test_reset_frozen_index(self):
        n_sites = 6
        frozen = make_initial_states(seed=0, n_sites=n_sites, n_states=2)

        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(
            env=base_env,
            potential_fn=onemax,
            reward_mode="dense_potential",
            frozen_states=frozen,
        )

        obs1, _ = env.reset()  # frozen[0]
        env.reset_frozen_index()
        obs2, _ = env.reset()  # frozen[0] again
        np.testing.assert_array_equal(obs1, obs2)
