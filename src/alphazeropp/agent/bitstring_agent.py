import warnings
from copy import deepcopy

import numpy as np

from alphazeropp.agent.agent import Agent
from alphazeropp.agent.mcts import entab


class BitStringAgent(Agent):
    """
    This class is a simple agent that plays the BitStringGame perfectly to generate training data.
    """

    def get_exact_move_probs(self, msg: str = "") -> np.ndarray:
        """
        Returns the exact move probabilities for the current game state.
        """
        if msg: print(msg, "Calculating exact move probabilities for", self.game.obs)
        # return the exact move probabilities
        # based on the current game state. 
        probs = np.zeros(self.game.env.nsites, dtype=np.float32)
        for i in range(self.game.env.nsites):
            if self.game.obs[i] == 0:
                probs[i] = 1.0
            else:
                probs[i] = 0.0
        probs /= np.sum(probs)  # Normalize to make it a probability distribution
        if msg: print(msg, "Exact move probabilities:", probs)
        return probs

    def play_single_game(self, max_moves: int = 10_000, random_seed: int | None = None, msg = ""):
        train_examples = []
        rewards = []
        # mcts = MCTS(self.game, self.net, **self.mcts_params)
        rng = np.random.default_rng(random_seed)
        for i in range(max_moves):
            if msg: print(msg, f"starting move {i}")
            # move_probs = mcts.perform_simulations(entab(msg, f", m{i+1}"))
            # self.game = mcts.game  # TODO HACK because MCTS modifies the game state in place
            move_probs = self.get_exact_move_probs(entab(msg, f", m{i+1}"))
            train_examples.append((deepcopy(self.game.obs), (move_probs, None)))
            selected_move = rng.choice(len(move_probs), p=move_probs)
            if msg: print(msg, "obs", self.game.obs, "hobs", self.game.hashable_obs, "move_probs", move_probs, "selmove", selected_move)
            # print(f"Taking move {selected_move} with probability {move_probs[selected_move]:.2f}")  # TODO logging
            self.game.step_wrapper(selected_move)
            rewards.append(self.game.reward)
            if self.game.terminated or self.game.truncated:
                break
        else:
            # In this case, we might not have any reward to work with
            warnings.warn(f"`play_single_game` timed out after {max_moves} moves without termination/truncation, returning no training examples")
            return []

        # Propagate rewards backwards through steps
        for i in range(len(rewards) - 1, 0, -1):
            rewards[i-1] += self.reward_discount * rewards[i]
        
        # Attach rewards to training examples
        for i in range(len(train_examples)):
            state, (policy, _) = train_examples[i]
            train_examples[i] = (state, (policy, rewards[i]))
        
        return train_examples