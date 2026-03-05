"""
Domain-agnostic DSL synthesis engine.

Provides AST nodes, grammar, derivation engine, interpreter, and
leaf evaluator for MCTS-guided program synthesis. Domain-specific
game configs (bitstring, doors) are in their respective instance packages.
"""

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
    Condition, Program,
)
from alphazeropp.synthesis.interpreter import (
    eval_condition, eval_program, interp_ops, run_policy_episode,
)
from alphazeropp.synthesis.budget_grammar import (
    ProgramHole, ConditionHole,
    enumerate_programs, enumerate_conditions,
    count_programs, count_conditions,
)
from alphazeropp.synthesis.derivation import DerivationState, Production
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.synthesis.derivation_game import DerivationGame, UniformPolicyValueNet
