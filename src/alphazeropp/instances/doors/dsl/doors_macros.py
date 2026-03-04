"""Domain-aware macro productions for the Doors environment.

Generates PickRule and MoveRule macro productions that expand a
ProgramHole into a multi-node Ite subtree in a single derivation step.
The resulting AST uses only existing node types — no interpreter or
evaluator changes are needed.

PickRule(key_id) — 7 AST nodes:
    Ite(And(Not(IsZero(key_loc[k])), Not(IsZero(M+D+k))),
        Flip(M+k),
        ProgramHole(budget - 7))

MoveRule(key_id) — 3 AST nodes:
    Ite(IsZero(M + key_unlocks[k]),
        Flip(key_loc[k]),
        ProgramHole(budget - 3))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite,
)
from alphazeropp.synthesis.budget_grammar import ProgramHole
from alphazeropp.synthesis.derivation import Production

if TYPE_CHECKING:
    from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig

PICK_RULE_COST = 7  # Ite + And + Not + IsZero + Not + IsZero + Flip
MOVE_RULE_COST = 3  # Ite + IsZero + Flip


def doors_macro_productions(
    budget: int,
    doors_cfg: DoorsGameConfig,
) -> list[Production]:
    """Generate PickRule and MoveRule macro productions for a ProgramHole.

    Args:
        budget: Remaining budget for the ProgramHole being expanded.
        doors_cfg: Doors layout providing key_loc, key_unlocks, M, D, K.

    Returns:
        List of Productions with ``macro_key`` set for factored-game
        template grouping.
    """
    prods: list[Production] = []
    M = doors_cfg.M
    D = doors_cfg.D
    K = doors_cfg.K

    # PickRule: budget >= PICK_RULE_COST + 2 (need ≥2 remaining for Default)
    if budget >= PICK_RULE_COST + 2:
        for k in range(K):
            else_budget = budget - PICK_RULE_COST
            loc_idx = doors_cfg.key_loc[k]
            avail_idx = M + D + k
            pick_action = M + k
            result = Ite(
                And(Not(IsZero(loc_idx)), Not(IsZero(avail_idx))),
                Flip(pick_action),
                ProgramHole(else_budget),
            )
            prods.append(Production(
                hole_kind="P",
                hole_budget=budget,
                result=result,
                label=f"P({budget}) -> PickRule(key={k}, P({else_budget}))",
                macro_key=("PickRule",),
            ))

    # MoveRule: budget >= MOVE_RULE_COST + 2
    if budget >= MOVE_RULE_COST + 2:
        for k in range(K):
            else_budget = budget - MOVE_RULE_COST
            lock_idx = M + doors_cfg.key_unlocks[k]
            move_action = doors_cfg.key_loc[k]
            result = Ite(
                IsZero(lock_idx),
                Flip(move_action),
                ProgramHole(else_budget),
            )
            prods.append(Production(
                hole_kind="P",
                hole_budget=budget,
                result=result,
                label=f"P({budget}) -> MoveRule(key={k}, P({else_budget}))",
                macro_key=("MoveRule",),
            ))

    return prods


def make_macro_fn(doors_cfg: DoorsGameConfig):
    """Create a closure suitable for FactoredDerivationGame's macro_productions_fn."""
    def macro_fn(budget: int) -> list[Production]:
        return doors_macro_productions(budget, doors_cfg)
    return macro_fn
