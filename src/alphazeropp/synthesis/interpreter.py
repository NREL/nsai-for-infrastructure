"""
Interpreter for the decision-list DSL.

Provides:
  eval_condition(cond, state) -> bool
  eval_program(program, state) -> int       (action index)
  interp_ops(program, state)  -> int        (interpretation cost)
  run_policy_episode(env, program, ...)      -> EpisodeResult
  format_trace(result)                       -> str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
    Condition, Program,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition / program evaluation
# ---------------------------------------------------------------------------

def eval_condition(cond: Condition, state: np.ndarray) -> bool:
    """Evaluate a condition on a bitstring state."""
    if isinstance(cond, IsZero):
        return bool(state[cond.index] == 0)
    elif isinstance(cond, Not):
        return not eval_condition(cond.child, state)
    elif isinstance(cond, And):
        return eval_condition(cond.left, state) and eval_condition(cond.right, state)
    else:
        raise TypeError(f"Unknown condition type: {type(cond)}")


def eval_program(program: Program, state: np.ndarray) -> int:
    """Walk the decision list and return the first matching action index.

    Totality is guaranteed structurally: every Program terminates with
    a Default node.
    """
    if isinstance(program, Default):
        return program.action.index
    elif isinstance(program, Ite):
        if eval_condition(program.cond, state):
            return program.action.index
        return eval_program(program.else_prog, state)
    else:
        raise TypeError(f"Unknown program type: {type(program)}")


# ---------------------------------------------------------------------------
# Interpretation cost model
# ---------------------------------------------------------------------------

def _condition_ops(cond: Condition) -> int:
    """Cost of evaluating a condition (purely structural, no short-circuit).

    IsZero(i):  1   (bit read)
    Not(c):     1 + cost(c)
    And(l, r):  1 + cost(l) + cost(r)   [both sides always counted]
    """
    if isinstance(cond, IsZero):
        return 1
    elif isinstance(cond, Not):
        return 1 + _condition_ops(cond.child)
    elif isinstance(cond, And):
        return 1 + _condition_ops(cond.left) + _condition_ops(cond.right)
    else:
        raise TypeError(f"Unknown condition type: {type(cond)}")


def interp_ops(program: Program, state: np.ndarray) -> int:
    """Total interpretation ops for evaluating *program* on *state*.

    Accounts for first-match semantics: conditions that fail still incur
    their full cost, but once a condition matches the remaining rules are
    not evaluated.

    Cost at the program level:
      Default(action):            cost(action) = 1
      Ite(cond, action, else):
        if cond true:   cost(cond) + 1          (cond + Flip)
        if cond false:  cost(cond) + interp_ops(else, state)
    """
    if isinstance(program, Default):
        return 1  # Flip emit
    elif isinstance(program, Ite):
        cond_cost = _condition_ops(program.cond)
        if eval_condition(program.cond, state):
            return cond_cost + 1  # cond + Flip
        return cond_cost + interp_ops(program.else_prog, state)
    else:
        raise TypeError(f"Unknown program type: {type(program)}")


# ---------------------------------------------------------------------------
# Execution trace data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Record of a single step in a policy episode."""
    step_num: int
    state: np.ndarray
    action: int
    reward: float
    interp_ops: int
    rule_trace: list[str]   # per-rule verdict for this step


@dataclass
class EpisodeResult:
    """Complete result of running a policy episode."""
    steps: list[StepRecord]
    total_env_steps: int
    total_interp_ops: int
    final_state: np.ndarray
    cumulative_reward: float
    solved: bool


# ---------------------------------------------------------------------------
# Rule-trace helper
# ---------------------------------------------------------------------------

def _trace_rules(program: Program, state: np.ndarray) -> list[str]:
    """Walk the decision list, building a human-readable trace of each rule.

    Returns a list of strings like:
      ["Rule 1: IsZero(0) -> TRUE -> Flip(0)"]
    or
      ["Rule 1: IsZero(0) -> FALSE (bit 0 = 1)",
       "Default: Flip(3)"]
    """
    traces: list[str] = []
    rule_num = 1
    node = program

    while isinstance(node, Ite):
        cond_str = node.cond.pretty()
        matched = eval_condition(node.cond, state)
        if matched:
            traces.append(
                f"Rule {rule_num}: {cond_str} -> TRUE -> {node.action.pretty()}"
            )
            return traces
        else:
            detail = _condition_fail_detail(node.cond, state)
            traces.append(
                f"Rule {rule_num}: {cond_str} -> FALSE ({detail})"
            )
        rule_num += 1
        node = node.else_prog

    # Must be Default
    assert isinstance(node, Default)
    traces.append(f"Default: {node.action.pretty()}")
    return traces


def _condition_fail_detail(cond: Condition, state: np.ndarray) -> str:
    """Short explanation of why a condition evaluated to its result."""
    if isinstance(cond, IsZero):
        return f"obs[{cond.index}] = {int(state[cond.index])}"
    elif isinstance(cond, Not):
        inner = eval_condition(cond.child, state)
        return f"child={inner}"
    elif isinstance(cond, And):
        l = eval_condition(cond.left, state)
        r = eval_condition(cond.right, state)
        return f"left={l}, right={r}"
    return ""


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_policy_episode(
    env,
    program: Program,
    x0: Optional[np.ndarray] = None,
    verbose: bool = False,
    *,
    is_solved: Callable[[np.ndarray], bool],
) -> EpisodeResult:
    """Run the DSL program as a policy on the environment.

    Args:
        env: A Gymnasium-compatible environment.
        program: A Program (Ite or Default) AST node.
        x0: Optional initial state override (copied into env after reset).
        verbose: If True, log detailed execution trace at DEBUG level.
        is_solved: Callable that checks if an observation is a solved state.

    Returns:
        EpisodeResult with per-step records and summary statistics.
    """
    obs, _ = env.reset()
    if x0 is not None:
        env.state = x0.copy()
        obs = x0.copy()

    steps: list[StepRecord] = []
    total_interp = 0
    cumulative_reward = 0.0

    if verbose:
        state_str = _fmt_state(obs)
        logger.debug("=== Episode Start ===")
        logger.debug("Initial state: %s", state_str)

    done = False
    step_num = 0
    while not done:
        action = eval_program(program, obs)
        ops = interp_ops(program, obs)
        rule_trace = _trace_rules(program, obs)

        prev_obs = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        step_num += 1
        total_interp += ops
        cumulative_reward += reward

        record = StepRecord(
            step_num=step_num,
            state=prev_obs,
            action=action,
            reward=reward,
            interp_ops=ops,
            rule_trace=rule_trace,
        )
        steps.append(record)

        if verbose:
            logger.debug(
                "Step %d: %s -> Flip(%d) -> reward=%+.4f, ops=%d",
                step_num, _fmt_state(prev_obs), action, reward, ops,
            )
            for line in rule_trace:
                logger.debug("  %s", line)

        done = terminated or truncated

    result = EpisodeResult(
        steps=steps,
        total_env_steps=step_num,
        total_interp_ops=total_interp,
        final_state=obs.copy(),
        cumulative_reward=cumulative_reward,
        solved=is_solved(obs),
    )

    if verbose:
        logger.debug("=== Episode End ===")
        logger.debug("Final state: %s  %s",
                      _fmt_state(result.final_state),
                      "SOLVED" if result.solved else "NOT SOLVED")
        logger.debug("Env steps: %d | Interp ops: %d | Reward: %.4f",
                      result.total_env_steps, result.total_interp_ops,
                      result.cumulative_reward)

    return result


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------

def _fmt_state(state: np.ndarray) -> str:
    return "[" + ", ".join(str(int(b)) for b in state) + "]"


def format_trace(result: EpisodeResult, program: Optional[Program] = None) -> str:
    """Format a complete episode trace as a human-readable string.

    Args:
        result: EpisodeResult from run_policy_episode.
        program: Optional program to print at the top of the trace.

    Example output::

        === Policy ===
        if IsZero(0):
          Flip(0)
        elif IsZero(1):
          Flip(1)
        else:
          Flip(3)

        Step 1: [0, 1, 0, 1]
          Rule 1: IsZero(0) -> TRUE -> Flip(0)
          reward=+0.2500, interp_ops=2

        Step 2: [1, 1, 0, 1]
          Rule 1: IsZero(0) -> FALSE (bit 0 = 1)
          Default: Flip(3)
          reward=-0.2500, interp_ops=2

        === Summary ===
        Final state: [1, 1, 1, 1] -- SOLVED
        Env steps: 2 | Total interp ops: 4 | Cumulative reward: 0.5000
    """
    lines: list[str] = []

    if program is not None:
        lines.append("=== Policy ===")
        lines.append(program.pretty())
        lines.append("")

    for step in result.steps:
        state_str = _fmt_state(step.state)
        lines.append(f"Step {step.step_num}: {state_str}")
        for rule_line in step.rule_trace:
            lines.append(f"  {rule_line}")
        lines.append(
            f"  reward={step.reward:+.4f}, interp_ops={step.interp_ops}"
        )
        lines.append("")

    final_str = _fmt_state(result.final_state)
    status = "SOLVED" if result.solved else "NOT SOLVED"
    lines.append("=== Summary ===")
    lines.append(f"Final state: {final_str} -- {status}")
    lines.append(
        f"Env steps: {result.total_env_steps} | "
        f"Total interp ops: {result.total_interp_ops} | "
        f"Cumulative reward: {result.cumulative_reward:.4f}"
    )
    return "\n".join(lines)
