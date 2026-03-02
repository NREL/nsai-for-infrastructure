from typing import Any, Callable

import logging

import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.core.policy_value_net import PolicyValueNet
from alphazeropp.core.mcts import MCTS, entab

"""
This is the basic implementation of the Agent class.
It is responsible for playing the game using the policy and value network, and collecting experience for training.
Other than that, the agent is not doing anything else, meaning that we have other classes to handle training, checkpointing, multithreading, etc. 
This is to keep the agent class focused on its core responsibility of playing the game and collecting experience.
"""

logger = logging.getLogger(__name__)

class Agent:
    # Constants
    RNG_NAMES = ["mcts", "train", "eval", "external_policy"]
    POLICY_RESERVED_NAMES = set(["old_net", "new_net", "old_net_no_mcts", "new_net_no_mcts"])
    
    # State
    game: Game
    net: PolicyValueNet
    rngs: dict[str, np.random.Generator]
    
    # Config
    mcts_params: dict
    reward_discount: float
    external_policy: Callable | None # If not None, a policy function to use rather than NN+MCTS for move selection
    external_policy_creators_to_pit: dict[str, Callable]
    
    # Random seeds
    random_seeds: dict[str, Any]
    
    
    def __init__(self,
                 game: Game,
                 net: PolicyValueNet,
                 mcts_params: dict,
                 reward_discount: float=1.0,
                 external_policy: Callable | None=None,
                 external_policy_creators_to_pit: dict[str, Callable]={}, # We ignore this one for now as I don't understand what it is used for.
                 random_seeds: dict[str, Any]={}
                 ):
        self.game = game # We may consider to discard the game object as the member variable. It feels more like something outside of the agent class.
        self.net = net
        
        self.mcts_params = mcts_params
        self.reward_discount = reward_discount
        self.external_policy = external_policy
        self.rng = None
        self._construct_rngs(random_seeds)
        
    def _construct_rngs(self, random_seeds: dict[str, Any]):
        self.rngs = {}
        for rng_name in self.RNG_NAMES:
            seed = random_seeds.get(rng_name, None)
            self.rngs[rng_name] = np.random.default_rng(seed)
        self.random_seeds = random_seeds
        if all(rng_name in random_seeds for rng_name in self.RNG_NAMES):
            logger.info("RNG seeds are fully specified")
        else:
            logger.info("RNG seeds are not fully specified, using nondeterministic seeds for: %s",
                        ", ".join(rng_name for rng_name in self.RNG_NAMES if rng_name not in random_seeds))
            
            
    def _randseed(self, rng_name: str) -> int:
        """Generate a random seed integer from the specified RNG."""
        return int(self.rngs[rng_name].integers(0, 2**31 - 1))
    
    def policy(self, state: Game, msg=None,
               add_noise: bool = True,
               temperature_override: float | None = None) -> np.ndarray:
        """
        The function returns the move probabilities for a game state.

        Args:
            state (Game): The current game state.
            msg (str, optional): Debug message prefix for logging moves. Defaults to None.
            add_noise (bool): Whether to add Dirichlet noise at the MCTS root.
                True for self-play (exploration), False for evaluation.
            temperature_override (float | None): If set, override the MCTS temperature
                for converting visit counts to probabilities. Low values (e.g. 0.05)
                give near-greedy action selection for evaluation.

        Returns:
            np.ndarray: Move probabilities, shape equal to the action space.
        """
        current_game_state = state.clone()

        if self.external_policy is not None:
            move_probs = self.external_policy(current_game_state)
        else:
            mcts = MCTS(current_game_state, self.net, **self.mcts_params)
            if temperature_override is not None:
                mcts.temperature = temperature_override
            move_probs = mcts.perform_simulations("", add_noise=add_noise)
        
        assert len(move_probs.shape) == 1, "move_probs should be a flat array"
        return move_probs
        
    def play_one_round(self, game: Game, max_moves: int = 10_000,
                       random_seed: int | None = None, msg="",
                       add_noise: bool = True,
                       temperature_override: float | None = None):
        """
        The function plays for one round from the given game state using the agent's policy.
        """
        current_game_state = game.clone()
        rng = np.random.default_rng(random_seed)

        collected_experience = []
        collected_rewards = [] # we seperately store rewards, because we want to calculate discounted rewards at the end of the episode.
        cumulative_reward = 0.0
        for i in range(max_moves):
            if msg: print(msg, f"at start of move {i+1}, obs is", current_game_state.obs)
            # We assume that move_probs has already been flattened inside the policy function.
            move_probs = self.policy(current_game_state, "",
                                     add_noise=add_noise,
                                     temperature_override=temperature_override)
            action_idx = rng.choice(len(move_probs), p=move_probs)
            """
            The implementation here is different from the original implementation in that
            we assume move_probs is already a flat array, so we can directly use rng.choice on it.
            """
            collected_experience.append((current_game_state.obs.copy(), move_probs)) # So the collected experience is a list of ((obs, move_probs), action_idx) tuples.
            
            _, reward, terminated, truncated, _ = current_game_state.step_wrapper(action_idx) # We only care about the reward here.
            
            collected_rewards.append(reward) # So the collected experience is a list of ((obs, action_idx), reward) tuples.
            cumulative_reward += reward
            if terminated or truncated:
                break
        
        # Now we calculate discounted rewards and combine them with the observations and move_probs to form the final experience tuples.
        discounted_rewards = []
        cumulative_reward = 0.0
        for reward in reversed(collected_rewards):
            cumulative_reward = reward + self.reward_discount * cumulative_reward
            discounted_rewards.append(cumulative_reward)
        discounted_rewards.reverse() # Now discounted_rewards is in the same order as collected_experience
        
        collected_experience = [(obs, move_probs, discounted_reward) for ((obs, move_probs), discounted_reward) in zip(collected_experience, discounted_rewards)]
        
            
        return collected_experience, cumulative_reward
    
    def play_one_round_reuse_tree(self, game: Game, max_moves: int = 10_000,
                                   random_seed: int | None = None, msg="",
                                   add_noise: bool = True,
                                   temperature_override: float | None = None):
        """Play one round using a single MCTS tree reused across all moves.

        Unlike play_one_round (which creates a fresh MCTS per move via policy()),
        this method creates one MCTS at the start and reuses its tree across
        all moves via perform_simulations_reuse() + advance_to().
        """
        mcts = MCTS(game.clone(), self.net, **self.mcts_params)
        if temperature_override is not None:
            mcts.temperature = temperature_override
        rng = np.random.default_rng(random_seed)

        collected_experience = []
        collected_rewards = []
        cumulative_reward = 0.0

        for i in range(max_moves):
            if msg: print(msg, f"at start of move {i+1}, obs is", mcts.game.obs)

            move_probs = mcts.perform_simulations_reuse("", add_noise=add_noise)
            assert len(move_probs.shape) == 1, "move_probs should be a flat array"

            action_idx = rng.choice(len(move_probs), p=move_probs)
            # Read obs from mcts.game (not a local var) because unstash_state
            # may have replaced the game object reference.
            collected_experience.append((mcts.game.obs.copy(), move_probs))

            mcts.advance_to(action_idx)

            reward = mcts.game.reward
            terminated = mcts.game.terminated
            truncated = mcts.game.truncated

            collected_rewards.append(reward)
            cumulative_reward += reward
            if terminated or truncated:
                break

        # Calculate discounted rewards (same logic as play_one_round)
        discounted_rewards = []
        cumulative_reward = 0.0
        for reward in reversed(collected_rewards):
            cumulative_reward = reward + self.reward_discount * cumulative_reward
            discounted_rewards.append(cumulative_reward)
        discounted_rewards.reverse()

        collected_experience = [(obs, move_probs, discounted_reward) for ((obs, move_probs), discounted_reward) in zip(collected_experience, discounted_rewards)]

        return collected_experience, sum(collected_rewards)

    def play_for_experience(self, game: Game, id: int, reset_seed: int, interaction_seed,
                            add_noise: bool = True,
                            temperature_override: float | None = None):
        import torch
        if torch.cuda.is_available():
            torch.cuda.init() # We need to initialize CUDA in the child process before we can use the network, otherwise we may encounter issues with CUDA context initialization in multiprocessing.
        current_game_state = game.clone() # we make sure that it doesn't interfere with the original game state.
        current_game_state.reset_wrapper(seed=reset_seed)

        return self.play_one_round(current_game_state, random_seed=interaction_seed, msg="",
                                   add_noise=add_noise,
                                   temperature_override=temperature_override)

    def play_for_experience_reuse_tree(self, game: Game, id: int, reset_seed: int, interaction_seed,
                                        add_noise: bool = True,
                                        temperature_override: float | None = None):
        """Like play_for_experience but using tree reuse across moves."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
        current_game_state = game.clone()
        current_game_state.reset_wrapper(seed=reset_seed)

        return self.play_one_round_reuse_tree(current_game_state, random_seed=interaction_seed, msg="",
                                               add_noise=add_noise,
                                               temperature_override=temperature_override)