"""Tests for DoorsPDDLLiteEnv, grammar restrictions, and is_solved integration."""

import numpy as np
import pytest

from alphazeropp.instances.doors.doors_pddl_lite import (
    DoorsPDDLLiteEnv,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.synthesis.derivation import (
    DerivationState, _condition_productions,
)
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.interpreter import run_policy_episode


# ---------------------------------------------------------------------------
# DoorsPDDLLiteEnv basics
# ---------------------------------------------------------------------------

class TestDoorsPDDLLiteEnvBasics:
    """Basic DoorsPDDLLiteEnv tests."""

    def test_reset_returns_valid_obs(self):
        """reset() returns obs with correct shape, dtype, and initial state."""
        env = DoorsPDDLLiteEnv.make_d2()
        obs, info = env.reset()
        # D=2, M=4 => obs_size = 4 + 2*2 - 1 = 7
        assert obs.shape == (7,)
        assert obs.dtype == np.float32
        # at start_loc=0
        assert obs[0] == 1.0
        # room 0 unlocked
        assert obs[4] == 1.0  # unlocked[0] at index M=4
        # room 1 locked
        assert obs[5] == 0.0  # unlocked[1] at index M+1=5
        # key 0 available
        assert obs[6] == 1.0  # key_available[0] at index M+D=6

    def test_move_to_unlocked_room(self):
        """MOVE_TO within unlocked room works."""
        env = DoorsPDDLLiteEnv.make_d2()
        env.reset()
        # Move from loc 0 to loc 1 (both in room 0, which is unlocked)
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs[1] == 1.0  # now at loc 1
        assert obs[0] == 0.0  # no longer at loc 0
        assert not terminated

    def test_move_blocked_into_locked_room(self):
        """MOVE_TO to locked room is a noop."""
        env = DoorsPDDLLiteEnv.make_d2()
        obs0, _ = env.reset()
        # Try to move to loc 2 (room 1, locked)
        obs, reward, terminated, truncated, info = env.step(2)
        # Should still be at loc 0
        assert obs[0] == 1.0
        assert obs[2] == 0.0

    def test_pick_requires_conjunction(self):
        """PICK from wrong location is a noop."""
        env = DoorsPDDLLiteEnv.make_d2()
        env.reset()
        # Agent at loc 0, key 0 at loc 1. PICK(0) = action M=4
        obs, reward, terminated, truncated, info = env.step(4)
        # Key should still be available (pick failed)
        assert obs[6] == 1.0  # key_available[0] unchanged
        assert obs[5] == 0.0  # room 1 still locked

    def test_pick_succeeds_at_key_location(self):
        """PICK works when both preconditions met (at key loc AND key available)."""
        env = DoorsPDDLLiteEnv.make_d2()
        env.reset()
        # Move to loc 1 (where key 0 is)
        env.step(1)
        # Now PICK(0) = action index M + 0 = 4
        obs, reward, terminated, truncated, info = env.step(4)
        assert obs[6] == 0.0  # key consumed
        assert obs[5] == 1.0  # room 1 unlocked

    def test_unlock_enables_movement(self):
        """After PICK, can move to newly unlocked room."""
        env = DoorsPDDLLiteEnv.make_d2()
        env.reset()
        # Move to key loc
        env.step(1)
        # Pick key
        env.step(4)
        # Now move to loc 2 (room 1, now unlocked)
        obs, reward, terminated, truncated, info = env.step(2)
        assert obs[2] == 1.0  # at loc 2

    def test_solved_at_goal(self):
        """Terminated when at_loc[goal_loc]==1."""
        env = DoorsPDDLLiteEnv.make_d2()
        env.reset()
        # D2: start=0, goal=3, key at 1
        # Move to 1, pick key, move to 3 (goal)
        env.step(1)
        env.step(4)
        obs, reward, terminated, truncated, info = env.step(3)
        assert terminated
        assert obs[3] == 1.0

    def test_horizon_truncation(self):
        """Truncated at horizon."""
        env = DoorsPDDLLiteEnv.make_d2(horizon=3)
        env.reset()
        # Do NOOPs until horizon
        noop = env.M + env.K  # NOOP action index
        env.step(noop)
        env.step(noop)
        obs, reward, terminated, truncated, info = env.step(noop)
        assert truncated
        assert not terminated

    def test_noop_no_change(self):
        """NOOP doesn't change state."""
        env = DoorsPDDLLiteEnv.make_d2()
        obs0, _ = env.reset()
        noop = env.M + env.K
        obs1, _, _, _, _ = env.step(noop)
        # State bits should be the same
        np.testing.assert_array_equal(obs0, obs1)

    def test_invalid_action_noop(self):
        """Out-of-range action is noop."""
        env = DoorsPDDLLiteEnv.make_d2()
        obs0, _ = env.reset()
        # obs_size=7, valid actions are 0..5 (M+K=5), noop=5, invalid=6
        obs1, _, _, _, _ = env.step(6)
        np.testing.assert_array_equal(obs0, obs1)


# ---------------------------------------------------------------------------
# D=3 full solve sequence
# ---------------------------------------------------------------------------

class TestDoorsD3:
    """Tests for the D=3 (3-room) layout."""

    def test_d3_full_solve(self):
        """Complete D=3 sequence: pick key0 -> unlock room1 -> pick key1 -> unlock room2 -> reach goal."""
        env = DoorsPDDLLiteEnv.make_d3()
        obs, _ = env.reset()
        # D=3, M=6: loc_room=[0,0,1,1,2,2], key_loc=[1,2], key_unlocks=[1,2]
        # start=0, goal=5
        # n_sites = 6 + 2*3 - 1 = 11
        assert obs.shape == (11,)

        # Move to loc 1 (room 0, key 0 location)
        obs, _, _, _, _ = env.step(1)
        assert obs[1] == 1.0

        # PICK(0) = action M + 0 = 6
        obs, _, _, _, info = env.step(6)
        assert obs[9] == 0.0   # key 0 consumed (key_offset = M+D = 6+3 = 9)
        assert obs[7] == 1.0   # room 1 unlocked (unlocked_offset + 1 = 6 + 1)

        # Move to loc 2 (room 1, key 1 location)
        obs, _, _, _, _ = env.step(2)
        assert obs[2] == 1.0

        # PICK(1) = action M + 1 = 7
        obs, _, _, _, info = env.step(7)
        assert obs[10] == 0.0  # key 1 consumed
        assert obs[8] == 1.0   # room 2 unlocked

        # Move to loc 5 (goal)
        obs, reward, terminated, truncated, _ = env.step(5)
        assert obs[5] == 1.0
        assert terminated


# ---------------------------------------------------------------------------
# Frozen states
# ---------------------------------------------------------------------------

class TestFrozenStates:
    """Tests for frozen-state cycling."""

    def test_frozen_states_cycling(self):
        """Frozen states cycle on reset."""
        state_a = np.zeros(7, dtype=np.float32)
        state_a[0] = 1.0
        state_a[4] = 1.0
        state_a[6] = 1.0

        state_b = np.zeros(7, dtype=np.float32)
        state_b[1] = 1.0
        state_b[4] = 1.0
        state_b[6] = 1.0

        env = DoorsPDDLLiteEnv.make_d2(frozen_states=[state_a, state_b])

        obs1, _ = env.reset()
        np.testing.assert_array_equal(obs1, state_a)

        obs2, _ = env.reset()
        np.testing.assert_array_equal(obs2, state_b)

        # Should cycle back
        obs3, _ = env.reset()
        np.testing.assert_array_equal(obs3, state_a)


# ---------------------------------------------------------------------------
# Grammar restrictions
# ---------------------------------------------------------------------------

class TestGrammarRestrictions:
    """Test allow_and and allow_not grammar flags."""

    def test_grammar_allow_and_false(self):
        """No And productions when allow_and=False."""
        # Test condition productions at budget=3 (where And would normally appear)
        prods = _condition_productions(3, n_sites=7, mode="exact",
                                        allow_and=False, allow_not=True)
        for p in prods:
            assert "And" not in p.label, f"Found And production: {p.label}"

    def test_grammar_allow_and_true(self):
        """And productions present when allow_and=True."""
        prods = _condition_productions(3, n_sites=7, mode="exact",
                                        allow_and=True, allow_not=True)
        and_prods = [p for p in prods if "And" in p.label]
        assert len(and_prods) > 0, "Expected And productions"

    def test_grammar_allow_not_false(self):
        """No Not productions when allow_not=False."""
        prods = _condition_productions(2, n_sites=7, mode="exact",
                                        allow_and=True, allow_not=False)
        for p in prods:
            assert "Not" not in p.label, f"Found Not production: {p.label}"

    def test_grammar_allow_not_true(self):
        """Not productions present when allow_not=True."""
        prods = _condition_productions(2, n_sites=7, mode="exact",
                                        allow_and=True, allow_not=True)
        not_prods = [p for p in prods if "Not" in p.label]
        assert len(not_prods) > 0, "Expected Not productions"


# ---------------------------------------------------------------------------
# DoorsGameConfig
# ---------------------------------------------------------------------------

class TestDoorsGameConfig:
    """Tests for DoorsGameConfig and doors_initial_state."""

    def test_initial_state(self):
        """doors_initial_state produces valid initial state."""
        cfg = DoorsGameConfig(num_rooms=2)
        state = doors_initial_state(cfg)
        assert state.shape == (cfg.obs_size(),)
        # at start_loc
        assert state[cfg.start_loc] == 1.0
        # room 0 unlocked
        assert state[cfg.M] == 1.0
        # key available
        assert state[cfg.M + cfg.D] == 1.0

    def test_make_env(self):
        """make_env returns a working environment."""
        cfg = DoorsGameConfig(num_rooms=2)
        state = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[state])
        obs, _ = env.reset()
        assert obs.shape == (cfg.obs_size(),)

    def test_is_solved(self):
        """is_solved checks goal location."""
        cfg = DoorsGameConfig(num_rooms=2)
        obs = np.zeros(cfg.obs_size(), dtype=np.float32)
        assert not cfg.is_solved(obs)
        obs[cfg.goal_loc] = 1.0
        assert cfg.is_solved(obs)


# ---------------------------------------------------------------------------
# is_solved callback integration
# ---------------------------------------------------------------------------

class TestIsSolvedCallback:
    """Test that run_policy_episode uses is_solved callback."""

    def test_is_solved_callback(self):
        """run_policy_episode uses custom is_solved when provided."""
        # Create a simple program: Default(Flip(0))
        program = Default(Flip(0))

        # Create a 2-site bitstring env (simple)
        from alphazeropp.instances.bitstring.dsl.game_config import GameConfig
        cfg = GameConfig(n_ones=1)
        env = cfg.make_env(2, frozen_states=[np.array([0, 1], dtype=np.float32)])
        env.reset()

        # Custom is_solved that always returns True
        result = run_policy_episode(env, program, is_solved=lambda obs: True)
        assert result.solved is True

        # Reset and run with is_solved that always returns False
        env.reset()
        result = run_policy_episode(env, program, is_solved=lambda obs: False)
        assert result.solved is False


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

class TestRewardStructure:
    """Test reward components."""

    def test_step_penalty(self):
        """Each step incurs the step penalty."""
        env = DoorsPDDLLiteEnv.make_d2(step_penalty=0.05)
        env.reset()
        noop = env.M + env.K
        _, reward, _, _, _ = env.step(noop)
        assert abs(reward - (-0.05)) < 1e-6

    def test_unlock_bonus(self):
        """Unlocking a room gives the unlock bonus."""
        env = DoorsPDDLLiteEnv.make_d2(unlock_bonus=0.2, step_penalty=0.0)
        env.reset()
        env.step(1)  # move to key loc
        _, reward, _, _, info = env.step(4)  # pick key
        assert abs(reward - 0.2) < 1e-6
        assert "unlocked_room" in info

    def test_goal_reward(self):
        """Reaching goal gives +1.0."""
        env = DoorsPDDLLiteEnv.make_d2(step_penalty=0.0)
        env.reset()
        env.step(1)  # move to key
        env.step(4)  # pick key
        _, reward, terminated, _, _ = env.step(3)  # move to goal
        assert terminated
        assert abs(reward - 1.0) < 1e-6
