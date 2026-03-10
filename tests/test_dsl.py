"""
Tests for the BitString decision-list DSL.

Categories:
1. AST node_count and pretty tests
2. Totality: all 2^N states produce valid actions
3. interp_ops: hand-calculated cost verification
4. Behavioral: greedy OneMax policy solves instances
5. Episode runner integration
6. Validation: out-of-range indices
"""

import logging

import numpy as np
import pytest

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.interpreter import (
    eval_condition, eval_program, interp_ops,
    run_policy_episode, format_trace,
)
from alphazeropp.instances.bitstring.game import BitStringGym
from alphazeropp.instances.bitstring.shaped_env import ShapedBitStringGym
from alphazeropp.instances.bitstring.potentials import onemax


def _all_ones(obs: np.ndarray) -> bool:
    """BitString solved check: all observation values are 1."""
    return bool(np.all(obs == 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_greedy_onemax(n_sites: int) -> Ite:
    """Build a greedy OneMax decision list: flip the first zero bit."""
    prog = Default(Flip(n_sites - 1))
    for i in range(n_sites - 2, -1, -1):
        prog = Ite(IsZero(i), Flip(i), prog)
    return prog


def _bits(val: int, n: int) -> np.ndarray:
    """Convert integer to a float32 bitstring array of length n (LSB-first)."""
    return np.array([(val >> j) & 1 for j in range(n)], dtype=np.float32)


# ---------------------------------------------------------------------------
# node_count tests
# ---------------------------------------------------------------------------

class TestNodeCount:
    def test_flip(self):
        assert Flip(0).node_count() == 1

    def test_iszero(self):
        assert IsZero(2).node_count() == 1

    def test_not(self):
        # Not(IsZero(0)) = 1 + 1 = 2
        assert Not(IsZero(0)).node_count() == 2

    def test_and(self):
        # And(IsZero(0), IsZero(1)) = 1 + 1 + 1 = 3
        assert And(IsZero(0), IsZero(1)).node_count() == 3

    def test_default(self):
        # Default(Flip(0)) = 1 + 1 = 2
        assert Default(Flip(0)).node_count() == 2

    def test_ite_simple(self):
        # Ite(IsZero(0), Flip(0), Default(Flip(1)))
        # = 1 (Ite) + 1 (IsZero) + 1 (Flip) + 1 (Default) + 1 (Flip) = 5
        p = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        assert p.node_count() == 5

    def test_nested(self):
        # Ite(And(IsZero(0), Not(IsZero(1))), Flip(0), Default(Flip(1)))
        # Ite=1, And=1, IsZero(0)=1, Not=1, IsZero(1)=1, Flip(0)=1, Default=1, Flip(1)=1
        # = 8
        cond = And(IsZero(0), Not(IsZero(1)))
        p = Ite(cond, Flip(0), Default(Flip(1)))
        assert p.node_count() == 8


# ---------------------------------------------------------------------------
# pretty tests
# ---------------------------------------------------------------------------

class TestPretty:
    def test_flip(self):
        assert Flip(3).pretty() == "Flip(3)"

    def test_iszero(self):
        assert IsZero(2).pretty() == "IsZero(2)"

    def test_not(self):
        assert Not(IsZero(1)).pretty() == "Not(IsZero(1))"

    def test_and(self):
        assert And(IsZero(0), IsZero(1)).pretty() == "And(IsZero(0), IsZero(1))"

    def test_decision_list(self):
        """Decision list pretty-prints as if/elif/else."""
        prog = Ite(IsZero(0), Flip(0),
                    Ite(IsZero(1), Flip(1),
                        Default(Flip(2))))
        text = prog.pretty()
        assert "if IsZero(0):" in text
        assert "elif IsZero(1):" in text
        assert "else:" in text
        assert "Flip(2)" in text

    def test_stability(self):
        """pretty() is deterministic."""
        prog = Ite(IsZero(0), Flip(0),
                    Ite(IsZero(1), Flip(1),
                        Default(Flip(2))))
        assert prog.pretty() == prog.pretty()


# ---------------------------------------------------------------------------
# eval_condition tests
# ---------------------------------------------------------------------------

class TestEvalCondition:
    def test_iszero_true(self):
        state = np.array([0.0, 1.0])
        assert eval_condition(IsZero(0), state) is True

    def test_iszero_false(self):
        state = np.array([1.0, 0.0])
        assert eval_condition(IsZero(0), state) is False

    def test_not(self):
        state = np.array([1.0, 0.0])
        assert eval_condition(Not(IsZero(0)), state) is True
        assert eval_condition(Not(IsZero(1)), state) is False

    def test_and_both_true(self):
        state = np.array([0.0, 0.0])
        assert eval_condition(And(IsZero(0), IsZero(1)), state) is True

    def test_and_one_false(self):
        state = np.array([1.0, 0.0])
        assert eval_condition(And(IsZero(0), IsZero(1)), state) is False

    def test_complex(self):
        # And(IsZero(0), Not(IsZero(1))) on [0, 1] -> True and True = True
        state = np.array([0.0, 1.0])
        cond = And(IsZero(0), Not(IsZero(1)))
        assert eval_condition(cond, state) is True


# ---------------------------------------------------------------------------
# eval_program tests
# ---------------------------------------------------------------------------

class TestEvalProgram:
    def test_default(self):
        state = np.array([1.0, 1.0, 1.0])
        assert eval_program(Default(Flip(2)), state) == 2

    def test_ite_match(self):
        prog = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        state = np.array([0.0, 1.0])
        assert eval_program(prog, state) == 0

    def test_ite_fallthrough(self):
        prog = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        state = np.array([1.0, 0.0])
        assert eval_program(prog, state) == 1

    def test_chain(self):
        prog = Ite(IsZero(0), Flip(0),
                    Ite(IsZero(1), Flip(1),
                        Default(Flip(2))))
        assert eval_program(prog, np.array([0.0, 1.0, 1.0])) == 0
        assert eval_program(prog, np.array([1.0, 0.0, 1.0])) == 1
        assert eval_program(prog, np.array([1.0, 1.0, 0.0])) == 2
        assert eval_program(prog, np.array([1.0, 1.0, 1.0])) == 2  # default


# ---------------------------------------------------------------------------
# Totality tests
# ---------------------------------------------------------------------------

class TestTotality:
    @pytest.mark.parametrize("n_sites", [3, 4, 5])
    def test_all_states_produce_valid_action(self, n_sites):
        """Every possible state maps to a valid action index."""
        prog = _make_greedy_onemax(n_sites)
        for val in range(2 ** n_sites):
            state = _bits(val, n_sites)
            action = eval_program(prog, state)
            assert 0 <= action < n_sites, \
                f"Invalid action {action} for state {state}"


# ---------------------------------------------------------------------------
# interp_ops tests (golden, hand-worked)
# ---------------------------------------------------------------------------

class TestInterpOps:
    def test_default_costs_1(self):
        """Default(Flip(x)) costs 1 op (the Flip emit)."""
        state = np.array([0.0, 1.0, 0.0, 1.0])
        assert interp_ops(Default(Flip(0)), state) == 1

    def test_ite_match_first_rule(self):
        """Ite(IsZero(0), Flip(0), Default(Flip(1))) on [0,1,0,1].
        IsZero(0) = 1 op, matches -> Flip(0) = 1 op. Total = 2."""
        prog = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        state = np.array([0.0, 1.0, 0.0, 1.0])
        assert interp_ops(prog, state) == 2

    def test_ite_fallthrough(self):
        """Ite(IsZero(0), Flip(0), Default(Flip(1))) on [1,0,0,1].
        IsZero(0) = 1 op, fails -> Default = 1 op. Total = 2."""
        prog = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        state = np.array([1.0, 0.0, 0.0, 1.0])
        assert interp_ops(prog, state) == 2

    def test_and_no_shortcircuit(self):
        """And always pays for both children.
        Ite(And(IsZero(0), IsZero(1)), Flip(0), Default(Flip(1))) on [1, 0]:
        And cost = 1 + 1 + 1 = 3, cond false, Default = 1. Total = 4."""
        cond = And(IsZero(0), IsZero(1))
        prog = Ite(cond, Flip(0), Default(Flip(1)))
        state = np.array([1.0, 0.0])
        assert interp_ops(prog, state) == 4

    def test_chained_fallthrough(self):
        """Three-rule chain, first two fail.
        Ite(IsZero(0), Flip(0),
            Ite(IsZero(1), Flip(1),
                Default(Flip(2))))
        on [1, 1, 0]:
          IsZero(0) = 1 op, fails
          IsZero(1) = 1 op, fails
          Default(Flip(2)) = 1 op
          Total = 3 ops"""
        prog = Ite(IsZero(0), Flip(0),
                    Ite(IsZero(1), Flip(1),
                        Default(Flip(2))))
        state = np.array([1.0, 1.0, 0.0])
        assert interp_ops(prog, state) == 3

    def test_spec_example_state_a(self):
        """From the spec: Ite(IsZero(1), Flip(1), Default(Flip(3)))
        on [1,0,1,0]: IsZero(1)=true (1 op) + Flip (1 op) = 2 ops."""
        prog = Ite(IsZero(1), Flip(1), Default(Flip(3)))
        state = np.array([1.0, 0.0, 1.0, 0.0])
        assert interp_ops(prog, state) == 2

    def test_spec_example_state_b(self):
        """From the spec: Ite(IsZero(1), Flip(1), Default(Flip(3)))
        on [1,1,1,0]: IsZero(1)=false (1 op) + Default (1 op) = 2 ops."""
        prog = Ite(IsZero(1), Flip(1), Default(Flip(3)))
        state = np.array([1.0, 1.0, 1.0, 0.0])
        assert interp_ops(prog, state) == 2

    def test_not_condition_cost(self):
        """Ite(Not(IsZero(0)), Flip(0), Default(Flip(1))) on [1, 0]:
        Not cost = 1 + 1 = 2, cond true -> Flip = 1. Total = 3."""
        prog = Ite(Not(IsZero(0)), Flip(0), Default(Flip(1)))
        state = np.array([1.0, 0.0])
        assert interp_ops(prog, state) == 3


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------

class TestBehavioral:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_greedy_onemax_solves(self, seed):
        """Greedy OneMax DSL policy achieves solve_rate=1.0."""
        n_sites = 6
        prog = _make_greedy_onemax(n_sites)
        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(base_env, onemax, "dense_potential")
        env.reset(seed=seed)
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        assert result.solved, \
            f"Greedy OneMax failed to solve (seed={seed})"

    def test_greedy_onemax_step_efficiency(self):
        """Greedy policy solves in exactly n_sites - n_ones steps."""
        n_sites = 8
        n_ones = 2  # BitStringGym default
        prog = _make_greedy_onemax(n_sites)
        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(base_env, onemax, "dense_potential")
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        assert result.total_env_steps == n_sites - n_ones, \
            f"Expected {n_sites - n_ones} steps, got {result.total_env_steps}"

    def test_reward_is_positive(self):
        """Greedy OneMax policy accumulates positive reward."""
        n_sites = 6
        prog = _make_greedy_onemax(n_sites)
        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(base_env, onemax, "dense_potential")
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        assert result.cumulative_reward > 0, \
            f"Expected positive reward, got {result.cumulative_reward}"

    def test_interp_ops_accumulated(self):
        """Total interp_ops across an episode equals sum of per-step ops."""
        n_sites = 4
        prog = _make_greedy_onemax(n_sites)
        base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        env = ShapedBitStringGym(base_env, onemax, "dense_potential")
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        manual_total = sum(s.interp_ops for s in result.steps)
        assert result.total_interp_ops == manual_total


# ---------------------------------------------------------------------------
# Episode runner integration
# ---------------------------------------------------------------------------

class TestEpisodeRunner:
    def test_result_structure(self):
        """EpisodeResult has all expected fields."""
        n_sites = 4
        prog = Default(Flip(0))
        env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        assert isinstance(result.total_env_steps, int)
        assert isinstance(result.total_interp_ops, int)
        assert isinstance(result.cumulative_reward, float)
        assert isinstance(result.solved, bool)
        assert isinstance(result.steps, list)
        assert len(result.steps) == result.total_env_steps

    def test_with_x0_override(self):
        """Episode can start from a specified initial state."""
        n_sites = 4
        prog = _make_greedy_onemax(n_sites)
        env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        x0 = np.array([1.0, 1.0, 0.0, 0.0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=_all_ones)
        # Exactly 2 zero bits to fix
        assert result.total_env_steps == 2
        assert result.solved

    def test_verbose_does_not_crash(self):
        """Verbose mode runs without errors."""
        logging.basicConfig(level=logging.DEBUG)
        n_sites = 4
        prog = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        result = run_policy_episode(env, prog, verbose=True, is_solved=_all_ones)
        assert result.total_env_steps > 0

    def test_format_trace_output(self):
        """format_trace produces a non-empty string with key sections."""
        n_sites = 4
        prog = _make_greedy_onemax(n_sites)
        env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        text = format_trace(result, program=prog)
        assert "=== Policy ===" in text
        assert "=== Summary ===" in text
        assert "Env steps:" in text
        assert "interp ops" in text.lower()

    def test_works_with_raw_bitstring_gym(self):
        """Episode runner works without ShapedBitStringGym wrapper."""
        n_sites = 4
        prog = _make_greedy_onemax(n_sites)
        env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
        result = run_policy_episode(env, prog, is_solved=_all_ones)
        assert result.solved


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_flip_out_of_range(self):
        with pytest.raises(ValueError):
            Flip(10).validate(n_sites=4)

    def test_flip_negative(self):
        with pytest.raises(ValueError):
            Flip(-1).validate(n_sites=4)

    def test_iszero_out_of_range(self):
        with pytest.raises(ValueError):
            IsZero(5).validate(n_sites=4)

    def test_valid_program_no_error(self):
        """No error for a valid program."""
        prog = Ite(IsZero(0), Flip(0), Default(Flip(1)))
        prog.validate(n_sites=4)  # should not raise

    def test_deep_validation(self):
        """Validation propagates through nested nodes."""
        prog = Ite(
            And(IsZero(0), Not(IsZero(1))),
            Flip(0),
            Default(Flip(99)),  # out of range
        )
        with pytest.raises(ValueError):
            prog.validate(n_sites=4)
