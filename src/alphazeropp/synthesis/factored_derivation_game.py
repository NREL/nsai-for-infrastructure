"""
FactoredDerivationGame: Two-phase game splitting (structure, parameter).

Each grammar production is factored into:
  1. Structure phase — pick a template (e.g., "Default" or "Ite(C(3),?,P(5))")
  2. Parameter phase — pick the Flip/IsZero index (only if parameterized)

Productions without parameters (Not, And) apply immediately in structure
phase — no parameter phase entered.

This reduces per-step branching from O(structures × parameters) to
O(max(structures, parameters)).  The set of reachable programs is
identical to DerivationGame (lossless factorization).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
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
from alphazeropp.synthesis.derivation_game import _preorder_items
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Structure template helpers
# ---------------------------------------------------------------------------

@dataclass
class StructureTemplate:
    """A group of productions sharing the same AST shape.

    For program productions, all variants in the group differ only in
    Flip index.  For condition productions, IsZero variants differ in
    observation index, while Not/And have exactly one production.
    """
    key: tuple
    productions: list[Production]
    needs_parameter: bool  # True iff len(productions) > 1


def _structure_key(prod: Production) -> tuple:
    """Extract template identity from a production's result node."""
    if prod.macro_key is not None:
        return prod.macro_key
    r = prod.result
    if isinstance(r, Default):
        return ("Default",)
    if isinstance(r, Ite):
        return ("Ite", r.cond.budget, r.else_prog.budget)
    if isinstance(r, IsZero):
        return ("IsZero",)
    if isinstance(r, Not):
        return ("Not", r.child.budget)
    if isinstance(r, And):
        return ("And", r.left.budget, r.right.budget)
    raise ValueError(f"Unknown production result type: {type(r)}")


def _group_by_structure(prods: list[Production]) -> list[StructureTemplate]:
    """Group productions by structure template, preserving order."""
    groups: OrderedDict[tuple, list[Production]] = OrderedDict()
    for p in prods:
        key = _structure_key(p)
        groups.setdefault(key, []).append(p)
    return [
        StructureTemplate(key=k, productions=ps, needs_parameter=len(ps) > 1)
        for k, ps in groups.items()
    ]


# ---------------------------------------------------------------------------
# Action space computation
# ---------------------------------------------------------------------------

def compute_max_factored_actions(
    budget: int, n_sites: int, mode: str = "exact",
    allow_and: bool = True, allow_not: bool = True,
    n_actions: int | None = None,
    macro_productions_fn: Callable[[int], list] | None = None,
    max_condition_budget: int | None = None,
) -> int:
    """Compute the factored action space size.

    Returns max(max_structures_across_all_holes, max_params_across_all_holes).

    Args:
        macro_productions_fn: Optional callable(budget) returning extra
            Productions (with ``macro_key`` set) to include at each
            ProgramHole budget level.
        max_condition_budget: Cap on condition budget in Ite templates.
    """
    max_structures = 0
    max_params = 0

    for k in range(2, budget + 1):
        prods = _program_productions(k, n_sites, mode, n_actions=n_actions,
                                     max_condition_budget=max_condition_budget)
        if macro_productions_fn is not None:
            prods = prods + macro_productions_fn(k)
        if prods:
            templates = _group_by_structure(prods)
            max_structures = max(max_structures, len(templates))
            for t in templates:
                if t.needs_parameter:
                    max_params = max(max_params, len(t.productions))

    max_cond_k = budget
    if max_condition_budget is not None:
        max_cond_k = min(max_cond_k, max_condition_budget)
    for k in range(1, max_cond_k + 1):
        prods = _condition_productions(k, n_sites, mode=mode,
                    allow_and=allow_and, allow_not=allow_not)
        if prods:
            templates = _group_by_structure(prods)
            max_structures = max(max_structures, len(templates))
            for t in templates:
                if t.needs_parameter:
                    max_params = max(max_params, len(t.productions))

        # Also check parent_is_not=True variant (fewer templates)
        prods_not = _condition_productions(k, n_sites, parent_is_not=True,
                        mode=mode, allow_and=allow_and, allow_not=allow_not)
        if prods_not:
            templates_not = _group_by_structure(prods_not)
            max_structures = max(max_structures, len(templates_not))
            for t in templates_not:
                if t.needs_parameter:
                    max_params = max(max_params, len(t.productions))

    return max(max_structures, max_params)


# ---------------------------------------------------------------------------
# FactoredDerivationGame
# ---------------------------------------------------------------------------

class FactoredDerivationGame(Game):
    """Two-phase derivation game: structure choice then parameter choice.

    Same semantics as DerivationGame but with factored action space.
    Each derivation step becomes 1-2 game steps:
      - Structure phase: pick a template from grouped productions
      - Parameter phase: pick the Flip/IsZero index (if template is
        parameterized)

    Templates without parameters (Not, And) apply immediately during
    the structure phase — no parameter phase entered.
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
        macro_productions_fn: Callable[[int], list] | None = None,
        max_condition_budget: int | None = None,
        _cached_max_factored: int | None = None,
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
        self._macro_productions_fn = macro_productions_fn
        self._max_condition_budget = max_condition_budget

        if _cached_max_factored is not None:
            self._max_factored = _cached_max_factored
        else:
            self._max_factored = compute_max_factored_actions(
                budget, n_sites, mode=program_budget_mode,
                allow_and=allow_and, allow_not=allow_not,
                n_actions=n_actions,
                macro_productions_fn=macro_productions_fn,
                max_condition_budget=max_condition_budget,
            )
        self.action_space = spaces.Discrete(self._max_factored)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * budget + 2,), dtype=np.float32,
        )

        # Derivation state (set on reset)
        self._deriv_state: DerivationState | None = None
        self._phase: str = "structure"
        self._pending_template: StructureTemplate | None = None
        self._structure_templates: list[StructureTemplate] = []
        self._parameter_productions: list[Production] = []

        # Template cache: avoids regenerating productions for the same
        # derivation state across MCTS simulations.
        self._template_cache: dict[str, list[StructureTemplate]] = {}

    def _compute_templates(self) -> list[StructureTemplate]:
        """Get structure templates for current state (cached)."""
        cache_key = self._deriv_state.pretty()
        cached = self._template_cache.get(cache_key)
        if cached is not None:
            return cached

        prods = self._deriv_state.legal_productions(
            self.n_sites, mode=self._mode,
            allow_and=self._allow_and, allow_not=self._allow_not,
            n_actions=self._n_actions,
            one_hot_groups=self._one_hot_groups,
            max_condition_budget=self._max_condition_budget,
        )
        # Append macro productions if at a ProgramHole
        if self._macro_productions_fn is not None:
            hole = self._deriv_state.leftmost_hole()
            if isinstance(hole, ProgramHole):
                prods = prods + self._macro_productions_fn(hole.budget)
        templates = _group_by_structure(prods)
        self._template_cache[cache_key] = templates
        return templates

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._deriv_state = DerivationState.initial(self.budget)
        self._phase = "structure"
        self._pending_template = None
        self._template_cache = {}
        self._structure_templates = self._compute_templates()
        self._parameter_productions = []
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._phase == "structure":
            return self._step_structure(action)
        else:
            return self._step_parameter(action)

    def _step_structure(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Handle a structure phase action."""
        template = self._structure_templates[action]

        if template.needs_parameter:
            # Enter parameter phase without applying
            self._phase = "parameter"
            self._pending_template = template
            self._parameter_productions = template.productions
            obs = self._encode_obs()
            info: dict[str, Any] = {
                "phase": "structure->parameter",
                "structure_key": template.key,
                "branching": len(self._parameter_productions),
            }
            return obs, 0.0, False, False, info
        else:
            # No parameter needed — apply immediately
            prod = template.productions[0]
            return self._apply_production(prod)

    def _step_parameter(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Handle a parameter phase action."""
        prod = self._parameter_productions[action]
        return self._apply_production(prod)

    def _apply_production(self, prod: Production) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply a production and transition to structure phase for next hole."""
        self._deriv_state = self._deriv_state.apply(prod)
        self._phase = "structure"
        self._pending_template = None
        self._parameter_productions = []
        self._structure_templates = self._compute_templates()

        is_complete = self._deriv_state.is_terminal()
        is_dead_end = not is_complete and len(self._structure_templates) == 0

        terminated = is_complete
        truncated = is_dead_end
        info: dict[str, Any] = {
            "production": prod,
            "hole_type": prod.hole_kind,
            "hole_budget": prod.hole_budget,
            "branching": len(self._structure_templates),
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
        mask = np.zeros(self._max_factored, dtype=bool)
        if self._phase == "structure":
            mask[:len(self._structure_templates)] = True
        else:
            mask[:len(self._parameter_productions)] = True
        return mask

    @property
    def hashable_obs(self) -> str:
        pending_key = str(self._pending_template.key) if self._pending_template else ""
        return f"{self._deriv_state.pretty()}|{self._phase}|{pending_key}"

    def get_program(self):
        """Return the completed program, or None if derivation is not terminal."""
        if self._deriv_state is not None and self._deriv_state.is_terminal():
            return self._deriv_state.to_program()
        return None

    # -- Stash / Unstash (avoids deepcopy) --

    def stash_state(self) -> tuple:
        """Save game state as a lightweight tuple."""
        return (
            self._deriv_state,
            self._phase,
            self._pending_template,
            self._structure_templates,
            self._parameter_productions,
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
            self._phase,
            self._pending_template,
            self._structure_templates,
            self._parameter_productions,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> "FactoredDerivationGame":
        """Return a lightweight independent copy."""
        new = FactoredDerivationGame(
            self.budget, self.n_sites, self.leaf_evaluator,
            program_budget_mode=self._mode,
            allow_and=self._allow_and,
            allow_not=self._allow_not,
            n_actions=self._n_actions,
            one_hot_groups=self._one_hot_groups,
            macro_productions_fn=self._macro_productions_fn,
            max_condition_budget=self._max_condition_budget,
            _cached_max_factored=self._max_factored,
        )
        new._template_cache = self._template_cache  # share cache
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new

    # -- Observation encoding --

    def _encode_obs(self) -> np.ndarray:
        """Encode partial AST + phase info as fixed-size float32 array.

        Layout: preorder (type_id, param) pairs padded to 2*budget,
        then [phase_id, pending_structure_id].
        """
        obs = np.zeros(2 * self.budget + 2, dtype=np.float32)
        items = _preorder_items(self._deriv_state.root)
        for i, (type_id, param) in enumerate(items):
            if i >= self.budget:
                break
            obs[2 * i] = type_id
            obs[2 * i + 1] = param

        # Phase info
        obs[2 * self.budget] = 1.0 if self._phase == "parameter" else 0.0
        if self._pending_template is not None:
            # 1-indexed template ordinal (find index in structure_templates)
            for idx, t in enumerate(self._structure_templates):
                if t.key == self._pending_template.key:
                    obs[2 * self.budget + 1] = float(idx + 1)
                    break
        return obs
