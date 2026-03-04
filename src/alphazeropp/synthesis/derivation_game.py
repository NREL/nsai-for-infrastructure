"""
DerivationGame: A single-player game for MCTS-guided program synthesis.

Casts grammar derivation as a Game (compatible with core MCTS) where:
  - States are partial ASTs with holes
  - Actions are grammar productions (expanding the leftmost hole)
  - Terminal reward = leaf evaluation of the completed program

Also provides UniformPolicyValueNet, a baseline policy-value network
that returns uniform policy and zero value (MCTS relies entirely on
backed-up leaf values).
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.core.policy_value_net import PolicyValueNet
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.budget_grammar import (
    ProgramHole, ConditionHole,
)
from alphazeropp.synthesis.derivation import (
    DerivationState, Production,
    _program_productions, _condition_productions,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Observation encoding constants
# ---------------------------------------------------------------------------

NODE_TYPE_IDS = {
    "PAD": 0,
    "Flip": 1,
    "IsZero": 2,
    "Not": 3,
    "And": 4,
    "Ite": 5,
    "Default": 6,
    "ProgramHole": 7,
    "ConditionHole": 8,
}


def _preorder_items(node: Any) -> list[tuple[float, float]]:
    """Collect (type_id, parameter) pairs via preorder traversal."""
    if isinstance(node, ProgramHole):
        return [(NODE_TYPE_IDS["ProgramHole"], float(node.budget))]
    if isinstance(node, ConditionHole):
        return [(NODE_TYPE_IDS["ConditionHole"], float(node.budget))]
    if isinstance(node, Flip):
        return [(NODE_TYPE_IDS["Flip"], float(node.index))]
    if isinstance(node, IsZero):
        return [(NODE_TYPE_IDS["IsZero"], float(node.index))]
    if isinstance(node, Not):
        return [(NODE_TYPE_IDS["Not"], 0.0)] + _preorder_items(node.child)
    if isinstance(node, And):
        return ([(NODE_TYPE_IDS["And"], 0.0)]
                + _preorder_items(node.left)
                + _preorder_items(node.right))
    if isinstance(node, Default):
        return [(NODE_TYPE_IDS["Default"], 0.0)] + _preorder_items(node.action)
    if isinstance(node, Ite):
        return ([(NODE_TYPE_IDS["Ite"], 0.0)]
                + _preorder_items(node.cond)
                + _preorder_items(node.action)
                + _preorder_items(node.else_prog))
    return []


def compute_max_productions(
    budget: int, n_sites: int, mode: str = "exact",
    allow_and: bool = True, allow_not: bool = True,
    n_actions: int | None = None,
) -> int:
    """Compute the maximum number of legal productions at any derivation step.

    Iterates over all possible hole budgets that could appear during
    derivation from ProgramHole(budget) and returns the maximum
    production count.
    """
    max_prods = 0
    for k in range(2, budget + 1):
        n = len(_program_productions(k, n_sites, mode, n_actions=n_actions))
        if n > max_prods:
            max_prods = n
    for k in range(1, budget + 1):
        n = len(_condition_productions(k, n_sites, mode=mode,
                allow_and=allow_and, allow_not=allow_not))
        if n > max_prods:
            max_prods = n
    return max_prods


# ---------------------------------------------------------------------------
# DerivationGame
# ---------------------------------------------------------------------------

class DerivationGame(Game):
    """Single-player game where actions are grammar productions.

    The game starts with ProgramHole(budget) and at each step applies
    a production to the leftmost hole. The game terminates when the
    partial AST has no remaining holes (i.e., it's a complete program).

    Reward is 0.0 at non-terminal steps and leaf_evaluator(program) at
    the terminal step.

    Subclasses Game directly (NOT EnvGame) because this is not a
    Gymnasium environment.
    """

    def __init__(
        self,
        budget: int,
        n_sites: int,
        leaf_evaluator: LeafEvaluator,
        program_budget_mode: str = "exact",
        allow_and: bool = True,
        allow_not: bool = True,
        n_actions: int | None = None,
        one_hot_groups: list[list[int]] | None = None,
    ):
        super().__init__()
        self.budget = budget
        self.n_sites = n_sites
        self.leaf_evaluator = leaf_evaluator
        self._mode = program_budget_mode
        self._allow_and = allow_and
        self._allow_not = allow_not
        self._n_actions = n_actions
        self._one_hot_groups = one_hot_groups

        self._max_productions = compute_max_productions(
            budget, n_sites, mode=program_budget_mode,
            allow_and=allow_and, allow_not=allow_not,
            n_actions=n_actions,
        )
        self.action_space = spaces.Discrete(self._max_productions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * budget,), dtype=np.float32,
        )

        # Derivation state (set on reset)
        self._deriv_state: DerivationState | None = None
        self._current_productions: list[Production] = []

    def _legal_productions(self) -> list[Production]:
        """Get legal productions for current state with all grammar params."""
        return self._deriv_state.legal_productions(
            self.n_sites, mode=self._mode,
            allow_and=self._allow_and, allow_not=self._allow_not,
            n_actions=self._n_actions,
            one_hot_groups=self._one_hot_groups,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._deriv_state = DerivationState.initial(self.budget)
        self._current_productions = self._legal_productions()
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prod = self._current_productions[action]
        self._deriv_state = self._deriv_state.apply(prod)
        self._current_productions = self._legal_productions()

        is_complete = self._deriv_state.is_terminal()
        # Dead end: holes remain but no legal productions (e.g., ProgramHole
        # with budget 3 or 4 has no valid grammar rules). Treat as truncated.
        is_dead_end = not is_complete and len(self._current_productions) == 0

        terminated = is_complete
        truncated = is_dead_end
        info: dict[str, Any] = {
            "production": prod,
            "hole_type": prod.hole_kind,
            "hole_budget": prod.hole_budget,
            "branching": len(self._current_productions),
            "is_complete": is_complete,
            "is_dead_end": is_dead_end,
        }

        if is_complete:
            program = self._deriv_state.to_program()
            reward = self.leaf_evaluator(program)
            info["program"] = program
            info["leaf_value"] = reward
        elif is_dead_end:
            reward = 0.0
            info["dead_end"] = True
        else:
            reward = 0.0

        obs = self._encode_obs()
        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._max_productions, dtype=bool)
        mask[:len(self._current_productions)] = True
        return mask

    @property
    def hashable_obs(self) -> str:
        return self._deriv_state.pretty()

    def get_program(self):
        """Return the completed program, or None if derivation is not terminal."""
        if self._deriv_state is not None and self._deriv_state.is_terminal():
            return self._deriv_state.to_program()
        return None

    # -- Stash / Unstash (avoids deepcopy) --

    def stash_state(self) -> tuple:
        """Save game state as a lightweight tuple.

        Safe because AST nodes and Productions are frozen dataclasses.
        """
        return (
            self._deriv_state,
            self._current_productions,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        )

    def unstash_state(self, state: tuple):
        """Restore game state from a stashed tuple."""
        (
            self._deriv_state,
            self._current_productions,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> "DerivationGame":
        """Return a lightweight independent copy.

        Safe because AST nodes and Productions are frozen dataclasses,
        and step() replaces (rather than mutates) all mutable fields.
        """
        new = DerivationGame(self.budget, self.n_sites, self.leaf_evaluator,
                             program_budget_mode=self._mode,
                             allow_and=self._allow_and,
                             allow_not=self._allow_not,
                             n_actions=self._n_actions,
                             one_hot_groups=self._one_hot_groups)
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new

    # -- Observation encoding --

    def _encode_obs(self) -> np.ndarray:
        """Encode the partial AST as a fixed-size float32 array.

        Layout: pairs of (node_type_id, parameter) in preorder traversal,
        padded with zeros to length 2 * budget.
        """
        obs = np.zeros(2 * self.budget, dtype=np.float32)
        items = _preorder_items(self._deriv_state.root)
        for i, (type_id, param) in enumerate(items):
            if i >= self.budget:
                break
            obs[2 * i] = type_id
            obs[2 * i + 1] = param
        return obs


# ---------------------------------------------------------------------------
# UniformPolicyValueNet
# ---------------------------------------------------------------------------

class UniformPolicyValueNet(PolicyValueNet):
    """Uniform-random policy, constant value=0. MCTS baseline.

    With this network, MCTS relies entirely on PUCT exploration and
    backed-up leaf values. Any search structure comes from the value
    signal (leaf evaluation), not the policy prior.
    """

    def __init__(self, action_size: int):
        self.action_size = action_size

    def predict(self, state) -> tuple[np.ndarray, np.ndarray]:
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        value = np.array(0.0, dtype=np.float32)
        return policy, value

    def train(self, examples):
        pass

    def save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, save_dir):
        pass

    def push_multiprocessing(self):
        pass

    def pop_multiprocessing(self, *args):
        pass
