from typing import Any, Callable

import numpy as np

from alphazeropp.envs.game import Game
from alphazeropp.networks.networks import PolicyValueNet
from alphazeropp.agent.mcts import MCTS, entab


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
        self.game = game
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
            
            
    def policy(self, state: Any, use_mcts: bool=True, verbose=False) -> np.ndarray:
        if self.external_policy is not None:
            return self.external_policy(state)