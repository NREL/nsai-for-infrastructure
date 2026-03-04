import warnings
from typing import Any

import numpy as np

from .game import Game
from .policy_value_net import PolicyValueNet

EPS = 1e-8  # Add to UCB numerator to avoid zeroing out the policy when total_N is 0

def entab(s, addl):
    return "  " + s + addl if s else ""

class MCTSTreeNode():
    direct_reward: float  # Any direct reward from the Game for being in this state
    is_terminal_state: bool  # Whether this is an end state (terminated or truncated)
    nn_policy: Any  # The result of the policy network for this state (may be noise-mixed)
    nn_policy_original: Any  # The original NN policy before noise injection, for re-injection on tree reuse
    nn_value: Any  # The result of the value network for this state
    action_mask: Any  # A bit vector of action validity for this state
    total_N: int  # Total number of visits to this node, should equal sum(action_N.values())
    action_Q: dict[tuple, float]  # Q value for each action from this state
    action_N: dict[tuple, int]  # Number of visits for each action from this state

    def __init__(self, direct_reward: float, is_terminal_state: bool):
        self.direct_reward = direct_reward
        self.is_terminal_state = is_terminal_state

        self.nn_policy = None
        self.nn_policy_original = None
        self.nn_value = None
        self.action_mask = None

        self.total_N = 0
        self.action_Q = {}
        self.action_N = {}

class MCTS():
    game: Game
    net: PolicyValueNet
    nodes: dict[Any, MCTSTreeNode]  # Maps game state to MCTSTreeNode
    n_simulations: int  # Number of simulations to perform in each search
    temperature: float  # Temperature for exponentiated visit counts in the final answer
    c_exploration: float  # Exploration constant for UCB

    def __init__(self, game: Game, net: PolicyValueNet,
                 n_simulations: int = 25,
                 temperature: float = 1.0,
                 c_exploration: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.game = game
        self.net = net
        self.nodes = {}

        self.n_simulations = n_simulations
        self.temperature = temperature
        self.c_exploration = c_exploration
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # min max Q value
        self.q_min = float('inf')
        self.q_max = float('-inf')

    def perform_simulations(self, msg, add_noise=False):
        """
        Perform `n_simulations` simulations from the current game state, then return move
        probabilities from exponentiated visit counts. Special case: if `n_simulations` is
        negative, directly query the policy network rather than performing any simulations
        (results are still exponentiated).
        """
        mystate = self.game.hashable_obs
        # Reset min-max Q stats at start of search over this move
        self.q_min = float('inf')
        self.q_max = float('-inf')

        if msg: print(msg, "at start of perform_simulations, obs is", self.game.obs)

        if self.n_simulations < 0:
            if msg: print(msg, "n_simulations < 0, directly querying policy net")
            counts, _, _ = self.query_net_masked(msg)
        else:
            # Check for root node existence and expand if needed
            if mystate not in self.nodes:
                if msg: print(msg, "Root node not found, expanding immediately")
                # Temporarily stash state to query net safely
                old_game_state = self.game.stash_state()
                self.search(entab(msg, ", root expand"))
                self.game = self.game.unstash_state(old_game_state)
            
            mynode = self.nodes[mystate]

            # Add Dirichlet Noise (only if requested and node is fresh)
            if add_noise and mynode.total_N == 0:
                if msg: print(msg, f"Adding Dirichlet Noise (alpha={self.dirichlet_alpha}, eps={self.dirichlet_epsilon})")
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(mynode.nn_policy))
                
                # Apply noise only to valid moves to preserve the mask
                # Optimization: if using masked policy, we should mix noise properly.
                # Here we assume nn_policy is already masked and normalized by query_net_masked.
                # However, Dirichlet noise is over ALL entries usually.
                # Standard AlphaZero: add noise to legal moves only, or add to all and re-mask.
                
                # Rigorous approach: Mix noise into policy, then re-mask and re-normalize.
                # But since our policy is already masked, mixing with unmasked noise might introduce invalid moves.
                # So we should mask the noise too.
                
                mask = mynode.action_mask
                masked_noise = noise * mask
                sum_noise = masked_noise.sum()
                if sum_noise > 0:
                    masked_noise /= sum_noise
                    mynode.nn_policy = (1 - self.dirichlet_epsilon) * mynode.nn_policy + self.dirichlet_epsilon * masked_noise
                else: 
                     # Should basically never happen if mask has valid moves
                     warnings.warn("Dirichlet noise sum is zero after masking, skipping noise injection")

            for i in range(self.n_simulations):
                old_game_state = self.game.stash_state()
                self.search(entab(msg,  f", simulation {i+1}/{self.n_simulations}"))
                self.game = self.game.unstash_state(old_game_state)
                assert mystate == self.game.hashable_obs
            
            mynode = self.nodes[mystate]
            
            # Build counts array directly in the policy shape
            counts = np.zeros_like(mynode.nn_policy)
            for action, count in mynode.action_N.items():
                counts[action] = count
        
        if msg: print(msg, "mynode", mynode, "counts", counts)
        # Numerically stable temperature scaling via log-space:
        #   counts^(1/T) = exp(log(counts) / T)
        # Subtracting the max before exp prevents overflow at low temperatures.
        nonzero = counts > 0
        if nonzero.any():
            log_counts = np.full_like(counts, -np.inf)
            log_counts[nonzero] = np.log(counts[nonzero]) / self.temperature
            log_counts -= log_counts.max()
            probs = np.exp(log_counts)
            probs /= probs.sum()
        else:
            probs = counts  # all zeros — caller will handle
        # For debugging:
        # print(probs)
        
        return probs

    def perform_simulations_reuse(self, msg, add_noise=False):
        """Like perform_simulations, but designed for tree reuse across moves.

        Key difference: noise injection uses nn_policy_original instead of checking
        total_N == 0, so that Dirichlet noise is correctly applied to reused root
        nodes that already have visit counts from prior searches.
        """
        mystate = self.game.hashable_obs
        # Reset min-max Q stats at start of search over this move
        self.q_min = float('inf')
        self.q_max = float('-inf')

        if msg: print(msg, "at start of perform_simulations_reuse, obs is", self.game.obs)

        if self.n_simulations < 0:
            if msg: print(msg, "n_simulations < 0, directly querying policy net")
            counts, _, _ = self.query_net_masked(msg)
        else:
            # Check for root node existence and expand if needed
            if mystate not in self.nodes:
                if msg: print(msg, "Root node not found, expanding immediately")
                old_game_state = self.game.stash_state()
                self.search(entab(msg, ", root expand"))
                self.game = self.game.unstash_state(old_game_state)

            mynode = self.nodes[mystate]

            # Noise injection for tree reuse: always restore from original policy
            # then mix fresh noise, regardless of total_N.
            if add_noise and mynode.nn_policy_original is not None:
                if msg: print(msg, f"Adding Dirichlet Noise (tree reuse, alpha={self.dirichlet_alpha}, eps={self.dirichlet_epsilon})")
                # Restore clean policy before mixing noise (prevents noise-on-noise)
                mynode.nn_policy = mynode.nn_policy_original.copy()
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(mynode.nn_policy))

                mask = mynode.action_mask
                masked_noise = noise * mask
                sum_noise = masked_noise.sum()
                if sum_noise > 0:
                    masked_noise /= sum_noise
                    mynode.nn_policy = (1 - self.dirichlet_epsilon) * mynode.nn_policy + self.dirichlet_epsilon * masked_noise
                else:
                    warnings.warn("Dirichlet noise sum is zero after masking, skipping noise injection")

            for i in range(self.n_simulations):
                old_game_state = self.game.stash_state()
                self.search(entab(msg, f", simulation {i+1}/{self.n_simulations}"))
                self.game = self.game.unstash_state(old_game_state)
                assert mystate == self.game.hashable_obs

            mynode = self.nodes[mystate]

            # Build counts array directly in the policy shape
            counts = np.zeros_like(mynode.nn_policy)
            for action, count in mynode.action_N.items():
                counts[action] = count

        if msg: print(msg, "mynode", mynode, "counts", counts)
        # Numerically stable temperature scaling via log-space
        nonzero = counts > 0
        if nonzero.any():
            log_counts = np.full_like(counts, -np.inf)
            log_counts[nonzero] = np.log(counts[nonzero]) / self.temperature
            log_counts -= log_counts.max()
            probs = np.exp(log_counts)
            probs /= probs.sum()
        else:
            probs = counts  # all zeros — caller will handle
        return probs

    def advance_to(self, action):
        """Advance the MCTS root to the state reached by taking `action`.

        Steps self.game forward. The new self.game.hashable_obs becomes the
        implicit root for the next perform_simulations_reuse() call. The search
        tree (self.nodes) is retained so the subtree below the chosen action
        provides a warm start. No pruning of unreachable nodes is performed.
        """
        self.game.step_wrapper(action)

    def search(self, msg) -> float:
        # NOTE does not guarantee that self.game will be in the same state upon exit
        mystate = self.game.hashable_obs
        # TODO HACK to get the step count
        if msg: print(msg, "BEGIN: searching state", self.game.obs, "step_count", self.game.step_count, "hash", hash(self.game.hashable_obs), "len nodes", len(self.nodes))

        # Initialize node if we've not been here before
        if mystate not in self.nodes:
            reward: float = self.game.reward  # type: ignore
            is_terminal: bool = self.game.terminated or self.game.truncated  # type: ignore
            if msg: print(msg, "adding node", self.game.obs, "hash", hash(self.game.hashable_obs), "terminated", self.game.terminated, "truncated", self.game.truncated, "reward", reward)
            self.nodes[mystate] = MCTSTreeNode(reward, is_terminal)
        mynode = self.nodes[mystate]
        
        # Base case: in a terminal state, return the direct reward
        if mynode.is_terminal_state:
            if msg: print(msg, "Reached terminal state", self.game.obs, "reward", mynode.direct_reward)
            # The reward for entering this state is already captured by the parent node.
            # The future value of a terminal state is 0.
            return 0.0
        
        # Base case: at a new node, query the policy-value network
        if mynode.nn_policy is None:
            if msg: print(msg, "Reached unexpanded node")
            assert mynode.nn_value is None
            
            mypolicy, myvalue, myaction_mask = self.query_net_masked(msg)
            mynode.nn_policy = mypolicy
            mynode.nn_policy_original = mypolicy.copy()
            mynode.nn_value = myvalue
            mynode.action_mask = myaction_mask
            return myvalue
        
        # Recursive case: select the best action using UCB with action masking and descend the tree
        ucbs = self.calc_masked_ucbs(mynode, entab(msg, " ucb"))
        best_action = np.unravel_index(np.argmax(ucbs), ucbs.shape)
        if msg: print(msg, "-> taking action", best_action, "based on UCBs", ucbs)

        # Distinguish scalar actions from 1-tuple actions and step
        to_step = best_action
        if len(self.game.action_space.shape) == 0: to_step, = to_step
        assert to_step in self.game.action_space
        self.game.step_wrapper(to_step)
        
        # Capture the immediate reward from the transition
        immediate_reward = self.game.reward

        # Recursively get the value of the next state (V(s'))
        future_value = self.search(entab(msg, " recurse"))
        # reward = self.search("")
        
        # Bellman update: Q(s,a) = R(s,a) + gamma * V(s')
        # Assuming gamma (discount) is 1.0 since it's not strictly passed to MCTS here,
        # but typically AlphaZero treats it as such or the game handles return calculation.
        # However, for dense rewards, we MUST add the immediate reward.
        total_reward = immediate_reward + future_value
        
        self.update_edge(mynode, best_action, total_reward)
        mynode.total_N += 1
        
        return total_reward

    def query_net(self, msg):
        "Query the policy-value network and perform some basic validation"
        mypolicy, myvalue = self.net.predict(self.game.obs)
        if msg: print(msg, "Queried NN: policy", mypolicy, "value", myvalue)
        if len(myvalue.shape) != 0:
            raise ValueError(f"Expected value to be scalar, got {myvalue.shape}")
        return mypolicy, myvalue
    
    def query_net_masked(self, msg):
        "Run `query_net` and use the game's action mask to zero out invalid moves in the policy"
        mypolicy, myvalue = self.query_net(msg)

        myaction_mask = self.game.get_action_mask()
        sum_mask = myaction_mask.sum()
        if sum_mask == 0:
            raise ValueError("Action mask says no valid moves")
        mypolicy *= myaction_mask
        sum_policy = mypolicy.sum()
        if sum_policy > 0:
            mypolicy /= sum_policy
        else:
            warnings.warn("All valid moves have zero probability, ignoring probabilities")
            mypolicy += myaction_mask / sum_mask

        return mypolicy, myvalue, myaction_mask
    
    def calc_masked_ucbs(self, mynode: MCTSTreeNode, msg) -> np.ndarray:
        mask = mynode.action_mask
        policy = mynode.nn_policy

        # Fast vectorized path for 1D action spaces
        if mask.ndim == 1:
            size = mask.shape[0]
            q_arr = np.zeros(size)
            n_arr = np.zeros(size)
            for action, q in mynode.action_Q.items():
                q_arr[action] = q
            for action, n in mynode.action_N.items():
                n_arr[action] = n

            # Normalize Q using global min/max seen during search
            if self.q_min == float('inf') or self.q_max == float('-inf'):
                q_norm = np.zeros(size)
            elif self.q_max > self.q_min:
                q_norm = (q_arr - self.q_min) / (self.q_max - self.q_min)
            else:
                q_norm = np.full(size, 0.5)

            # UCB = normalized_Q + c * P(a) * sqrt(total_N) / (1 + N(a))
            sqrt_total = np.sqrt(mynode.total_N + EPS)
            u = self.c_exploration * policy * sqrt_total / (1 + n_arr)
            ucbs = q_norm + u

            # Mask invalid actions to -inf
            all_ucbs = np.full(size, -np.inf)
            valid = mask.astype(bool)
            all_ucbs[valid] = ucbs[valid]
            return all_ucbs

        # Fallback for multi-dimensional action spaces
        valid_actions = list(zip(*np.nonzero(mask)))
        all_ucbs = np.full(policy.shape, -np.inf)
        for action in valid_actions:
            q = mynode.action_Q.get(action, 0.0)
            n = mynode.action_N.get(action, 0)
            if self.q_min == float('inf') or self.q_max == float('-inf'):
                q_normalized = 0.0
            elif self.q_max > self.q_min:
                q_normalized = (q - self.q_min) / (self.q_max - self.q_min)
            else:
                q_normalized = 0.5
            u_val = self.c_exploration * policy[action] * np.sqrt(mynode.total_N + EPS) / (1 + n)
            all_ucbs[action] = q_normalized + u_val
        return all_ucbs

    def update_edge(self, mynode: MCTSTreeNode, action: int, reward: float):
        if action not in mynode.action_N:
            assert action not in mynode.action_Q
            mynode.action_N[action] = 0  # could use a collections.Counter for this
            mynode.action_Q[action] = 0.0
        
        mynode.action_Q[action] = (mynode.action_N[action] * mynode.action_Q[action] + reward) / (1 + mynode.action_N[action])
        mynode.action_N[action] += 1

        # Update global min/max Q
        new_q = mynode.action_Q[action]
        if new_q < self.q_min:
            self.q_min = new_q
        if new_q > self.q_max:
            self.q_max = new_q