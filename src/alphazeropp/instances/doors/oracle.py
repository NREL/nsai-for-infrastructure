"""Oracle policy and environment auditing utilities for the Doors environment.

Provides:
- optimal_steps(D): minimum steps to solve D-room layout
- optimal_return(D): cumulative reward under optimal play
- reachable_state_count(D, locs_per_room): number of reachable (location, unlock_prefix) states
- oracle_action(obs, env): greedy oracle action for any state
- run_oracle_episode(env): run oracle to completion, return stats
"""

from __future__ import annotations

import numpy as np

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


def optimal_steps(D: int) -> int:
    """Minimum steps to solve a D-room Doors layout.

    Sequence: for each key k=0..D-2, MOVE to key location then PICK,
    then final MOVE to goal.  Total = 2*(D-1) + 1.
    """
    return 2 * (D - 1) + 1


def optimal_return(
    D: int, step_penalty: float = 0.01, unlock_bonus: float = 0.1
) -> float:
    """Cumulative reward under optimal play for D rooms."""
    K = D - 1
    steps = optimal_steps(D)
    return 1.0 + K * unlock_bonus - steps * step_penalty


def reachable_state_count(D: int, locs_per_room: int = 2) -> int:
    """Number of reachable states under monotone unlocking.

    The unlock state is a prefix p in {1, ..., D} (rooms 0..p-1 open).
    Key availability is fully determined by the prefix.
    For prefix p, the agent can occupy any of p * locs_per_room locations.

    Total = locs_per_room * sum(1..D) = locs_per_room * D * (D + 1) / 2.
    """
    return locs_per_room * D * (D + 1) // 2


def oracle_action(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> int:
    """Return the greedy-optimal action for the current observation.

    Strategy:
    1. Find the frontier key (lowest-index available key in an unlocked room).
    2. If at frontier key location and key available -> PICK it.
    3. If no frontier key (all picked) -> MOVE to goal.
    4. Else -> MOVE to frontier key location.
    """
    M = env.M
    key_offset = env._key_offset

    # Find agent location
    agent_loc = int(np.argmax(obs[:M]))

    # Find frontier key: first available key whose room is unlocked
    frontier_k = None
    for k in range(env.K):
        if obs[key_offset + k] == 1.0:
            # Key k is available; check if its room is accessible
            key_room = env.loc_room[env.key_loc[k]]
            if obs[env._unlocked_offset + key_room] == 1.0:
                frontier_k = k
                break

    if frontier_k is not None:
        # Can we pick it right now?
        if agent_loc == env.key_loc[frontier_k]:
            return env.encode_action("pick", frontier_k)
        else:
            return env.encode_action("move", env.key_loc[frontier_k])
    else:
        # All accessible keys picked -> go to goal
        return env.encode_action("move", env.goal_loc)


def run_oracle_episode(env: DoorsPDDLLiteEnv) -> dict:
    """Reset env and run oracle to completion.

    Returns dict with keys: reward, steps, solved, actions.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    actions = []

    for _ in range(env.horizon):
        action = oracle_action(obs, env)
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "steps": len(actions),
        "solved": env.is_solved(obs),
        "actions": actions,
    }


def enumerate_reachable_states(env: DoorsPDDLLiteEnv) -> set[tuple]:
    """BFS-enumerate all reachable (agent_loc, unlock_tuple) states.

    Returns a set of (agent_loc, (unlocked_0, ..., unlocked_{D-1})) tuples.
    Used for testing the closed-form formula.
    """
    obs, _ = env.reset()
    M = env.M
    unlocked_offset = env._unlocked_offset
    D = env.D

    def state_id(o: np.ndarray) -> tuple:
        loc = int(np.argmax(o[:M]))
        unlocked = tuple(int(o[unlocked_offset + r]) for r in range(D))
        return (loc, unlocked)

    visited = set()
    queue = [obs.copy()]
    visited.add(state_id(obs))

    while queue:
        current_obs = queue.pop(0)

        # Set env to this state so action_masks reads correctly
        env._state = current_obs.copy()
        env.step_count = 0
        env._terminated = False
        env._truncated = False
        mask = env.action_masks("precondition")

        for a in range(env.action_space.n):
            if not mask[a]:
                continue
            # Restore to current_obs before each action
            env._state = current_obs.copy()
            env.step_count = 0
            env._terminated = False
            env._truncated = False

            new_obs, _, terminated, _, _ = env.step(a)
            sid = state_id(new_obs)
            if sid not in visited:
                visited.add(sid)
                if not terminated:
                    queue.append(new_obs.copy())

    return visited
