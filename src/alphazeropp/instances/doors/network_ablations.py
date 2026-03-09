"""Network wrappers and ablation utilities for AlphaZero experiments.

Wrappers intercept predict() and/or train() calls to isolate
the contributions of different network components:

- FrozenWrapper: No training (random initialization preserved)
- UniformPolicyWrapper: Replaces policy with uniform distribution
- ZeroValueWrapper: Replaces value prediction with 0.0
- RandomNetWrapper: Eval-only random network (training delegates normally)

Utilities:
- value_only_1step_policy: One-step lookahead using only the value head
"""
import numpy as np


def _safe_getattr(self, name):
    """Pickle-safe __getattr__: avoids recursion when _net isn't set yet."""
    try:
        _net = object.__getattribute__(self, '_net')
    except AttributeError:
        raise AttributeError(name)
    return getattr(_net, name)


class UniformPolicyWrapper:
    """Replaces policy output with uniform distribution, keeps value.

    Tests: What does the value head alone contribute?
    MCTS gets no directional guidance from the policy prior.
    """

    def __init__(self, net):
        self._net = net

    def predict(self, state):
        policy, value = self._net.predict(state)
        uniform = np.ones_like(policy) / len(policy)
        return uniform, value

    def train(self, *args, **kwargs):
        return self._net.train(*args, **kwargs)

    __getattr__ = _safe_getattr


class ZeroValueWrapper:
    """Replaces value output with 0.0, keeps policy.

    Tests: What does the policy prior alone contribute?
    MCTS leaf nodes get no value signal.
    """

    def __init__(self, net):
        self._net = net

    def predict(self, state):
        policy, _ = self._net.predict(state)
        return policy, np.float32(0.0)

    def train(self, *args, **kwargs):
        return self._net.train(*args, **kwargs)

    __getattr__ = _safe_getattr


class FrozenWrapper:
    """Makes train() a no-op. Network stays at random initialization.

    Tests: Can MCTS solve the problem with a random (untrained) network?
    """

    def __init__(self, net):
        self._net = net

    def predict(self, state):
        return self._net.predict(state)

    def train(self, *args, **kwargs):
        # No-op: return dummy values matching expected 5-tuple signature
        # (model, train_batch_losses, train_losses, policy_losses, value_losses)
        return self._net.model, [0.0], [0.0], [0.0], [0.0]

    __getattr__ = _safe_getattr


class RandomNetWrapper:
    """Eval-only: predict() uses a fresh random-weight network.

    Training delegates to the real (learned) network, so the training loop
    is completely unaffected. Only evaluation sees random predictions.
    """

    def __init__(self, net):
        self._net = net
        # Lazy import to avoid circular dependency
        from alphazeropp.instances.doors.network import DoorsDirectNet
        self._random_net = DoorsDirectNet(
            input_size=net.model.input_size,
            output_size=net.model.output_size,
            device=str(net.DEVICE),
        )

    def predict(self, state):
        return self._random_net.predict(state)

    def train(self, *args, **kwargs):
        return self._net.train(*args, **kwargs)

    __getattr__ = _safe_getattr


def value_only_1step_policy(game, net, gamma=1.0):
    """One-step lookahead using only the value head.

    For each legal action: clone state, step, compute r + gamma * V(s').
    Returns a one-hot probability array on the argmax action.

    Parameters
    ----------
    game : DoorsDirectGame
        Current game state (not modified).
    net : network with predict(obs) -> (policy, value)
        Only the value output is used.
    gamma : float
        Discount factor for the next-state value.

    Returns
    -------
    np.ndarray
        One-hot probability vector of shape (n_actions,).
    """
    mask = game.get_action_mask()
    n_actions = len(mask)
    q_values = np.full(n_actions, -np.inf, dtype=np.float32)

    for a in range(n_actions):
        if not mask[a]:
            continue
        g = game.clone()
        _, reward, terminated, truncated, _ = g.step_wrapper(a)
        if terminated or truncated:
            q_values[a] = reward
        else:
            _, value = net.predict(g.obs)
            q_values[a] = reward + gamma * float(value)

    # One-hot on the best action
    best = int(np.argmax(q_values))
    probs = np.zeros(n_actions, dtype=np.float32)
    probs[best] = 1.0
    return probs
