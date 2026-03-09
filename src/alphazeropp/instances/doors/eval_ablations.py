"""Evaluation-time ablation dispatch for AlphaZero experiments.

Maps --eval-ablation mode names to evaluation functions that produce
(solve_rate, avg_reward) given an agent, without modifying training state.

Modes:
    full            Standard evaluation (MCTS + learned network)
    policy-only     Raw network policy, no MCTS (n_simulations=-1, temperature=1.0)
    value-only-1step  One-step lookahead using value head only
    uniform-prior   MCTS with uniform policy prior, learned value
    zero-value      MCTS with learned policy, value=0 at leaves
    unguided-search MCTS with uniform prior AND zero value
    random-net      MCTS with fresh random-weight network (training unaffected)
"""

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.instances.doors.network_ablations import (
    RandomNetWrapper,
    UniformPolicyWrapper,
    ZeroValueWrapper,
    value_only_1step_policy,
)

ABLATION_MODES = [
    "full",
    "policy-only",
    "value-only-1step",
    "uniform-prior",
    "zero-value",
    "unguided-search",
    "random-net",
]


def _eval_loop(agent, n_episodes, policy_fn=None):
    """Run greedy evaluation episodes.

    If policy_fn is provided, use it instead of agent.policy().
    policy_fn signature: (game_state) -> action_index

    Returns (solve_rate, avg_reward).
    """
    solves = 0
    total_reward = 0.0

    for i in range(n_episodes):
        game = agent.game.clone()
        game.reset_wrapper(seed=1000 + i)
        cumulative = 0.0

        for _ in range(game.env.horizon):
            if policy_fn is not None:
                action = policy_fn(game)
            else:
                probs = agent.policy(game, "", add_noise=False,
                                     temperature_override=0.01)
                action = int(np.argmax(probs))
            _, reward, terminated, truncated, _ = game.step_wrapper(action)
            cumulative += reward
            if terminated or truncated:
                break

        if game.env.is_solved(game.obs):
            solves += 1
        total_reward += cumulative

    return solves / n_episodes, total_reward / n_episodes


def _make_temp_agent(agent, net_override=None, mcts_override=None):
    """Create a lightweight Agent copy for eval-only use.

    Safe because Agent.__init__ doesn't mutate game/net (agent.py:40-56)
    and agent.policy() clones game internally (agent.py:93).
    """
    mcts_params = mcts_override if mcts_override is not None else agent.mcts_params
    return Agent(
        game=agent.game,
        net=net_override if net_override is not None else agent.net,
        mcts_params=mcts_params,
        reward_discount=agent.reward_discount,
        random_seeds=agent.random_seeds,
    )


def compute_solve_rate_ablated(agent, mode, n_episodes=20, gamma=1.0):
    """Run evaluation under the specified ablation mode.

    Training state (agent.net weights, trainer) is never modified.

    Parameters
    ----------
    agent : Agent
        The trained agent (not modified).
    mode : str
        One of ABLATION_MODES.
    n_episodes : int
        Number of evaluation episodes.
    gamma : float
        Discount factor for value-only-1step mode.

    Returns
    -------
    (solve_rate, avg_reward) : tuple[float, float]
    """
    if mode == "full":
        return _eval_loop(agent, n_episodes)

    if mode == "policy-only":
        mcts_params = agent.mcts_params.copy()
        mcts_params["n_simulations"] = -1
        temp = _make_temp_agent(agent, mcts_override=mcts_params)
        # Use temperature=1.0 so raw network policy is returned unsharpened
        return _eval_loop(temp, n_episodes,
                          policy_fn=lambda g: int(np.argmax(
                              temp.policy(g, "", add_noise=False,
                                          temperature_override=1.0))))

    if mode == "value-only-1step":
        net = agent.net

        def _v1_policy(game):
            probs = value_only_1step_policy(game, net, gamma=gamma)
            return int(np.argmax(probs))

        return _eval_loop(agent, n_episodes, policy_fn=_v1_policy)

    # Wrapper-based modes: create temp agent with wrapped net
    if mode == "uniform-prior":
        wrapped = UniformPolicyWrapper(agent.net)
    elif mode == "zero-value":
        wrapped = ZeroValueWrapper(agent.net)
    elif mode == "unguided-search":
        wrapped = ZeroValueWrapper(UniformPolicyWrapper(agent.net))
    elif mode == "random-net":
        wrapped = RandomNetWrapper(agent.net)
    else:
        raise ValueError(f"Unknown ablation mode: {mode!r}")

    temp = _make_temp_agent(agent, net_override=wrapped)
    return _eval_loop(temp, n_episodes)
