from typing import Any, Callable

import numpy as np

from alphazeropp.envs.game import Game
from alphazeropp.networks.networks import PolicyValueNet
from alphazeropp.core.mcts import MCTS, entab


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
            print(f"RNG seeds are fully specified")
        else:
            print(f"RNG seeds are not fully specified, using nondeterministic seeds for: {', '.join(rng_name for rng_name in self.RNG_NAMES if rng_name not in random_seeds)}")
            
            
    def policy(self, state: Game, msg=None) -> tuple[np.ndarray, Game]:
        """
        The function returns the move probabilities for a game state.
        Notice that it returns a tuple of (move_probs, state), where state is the original game state passed in.
        
        Args:
            state (Game): The current game state.
            msg (str, optional): Debug message prefix for logging moves. Defaults to None.

        Returns:
            tuple[np.ndarray, Game]: A tuple containing the move probabilities and the original game state.
            We expect move_probs to be a numpy array of shape equal to the action space of the game.
        """
        # We copy the original Game state to avoid modifying it during MCTS simulations
        state = state.stash_state()
        
        if self.external_policy is not None:
            move_probs = self.external_policy(state)
        else:
            mcts = MCTS(self.net, **self.mcts_params)
            move_probs = mcts.perform_simulations(state, "") # The second argument is a message prefix for debugging prints. We don't know what to put there for now, so we just put an empty string.
        
        assert len(move_probs.shape) == 1, "move_probs should be a flat array"
        return move_probs, state
        
    def play_single_game(self, game: Game, max_moves: int = 10_000, random_seed: int | None = None, msg = ""):
        original_game_state = game.stash_state()
        current_game_state = game.stash_state()
        rng = np.random.default_rng(random_seed)
        
        collected_experience = []
        for i in range(max_moves):
            if msg: print(msg, f"at start of move {i+1}, obs is", game.obs)
            # We assume that move_probs has already been flattened inside the policy function.
            move_probs, current_game_state = self.policy(current_game_state, "")
            action_idx = rng.choice(len(move_probs), p=move_probs)
            """
            The implementation here is different from the original implementation in that
            we assume move_probs is already a flat array, so we can directly use rng.choice on it.
            """
            
            _, reward, _, _, _ = current_game_state.step_wrapper(action_idx) # We only care about the reward here.
            
            
            
            