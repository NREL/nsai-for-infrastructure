import gymnasium as gym
import gymnasium.spaces as spaces

import numpy as np
import torch

from alphazeropp.core.game import EnvGame

from typing import Hashable
    
class BitStringGym(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_sites=10):
        super().__init__()
        self.bit_flip = True
        self.sparse_reward = True
        self.n_ones = 2 # Number of 1s that are initialized as 1
        
        self.n_sites = n_sites
        self.max_steps = 2 * self.n_sites if not self.sparse_reward else self.n_sites - self.n_ones
        self.observation_space = spaces.MultiBinary([self.n_sites]) #, seed=42)
        self.action_space = spaces.Discrete(self.n_sites)
        
        # Usually we don't reset the env in the __init__ function.
        self.state = None
        self.step_count = 0
        
    def step(self, action):
        assert self.state is not None, "Environment must be reset before stepping."
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        r = -1.0 / self.n_sites 
        
        if action == -1:
            return self.state.copy(), r, done, {}
        
        if self.state[action] == 0:
            r = 1.0 / self.n_sites
            
        if self.bit_flip:
            self.state[action] = 1 - self.state[action]  # Flip the bit
        else:
            self.state[action] = 1
        done = done or sum(self.state) == self.n_sites
        
        normalizer = self.n_sites
        if self.sparse_reward:
            if done:
                r = sum(self.state) / normalizer
            else:
                r = 0.0
        truncated = done # we now set truncated to be the same as done, since we don't have a separate truncation condition.
        
        return self.state.copy(), r, done, truncated, {}
        
        
    def reset(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            
        ones = np.random.choice(range(self.n_sites), self.n_ones, replace=False)
        self.state = np.zeros(self.n_sites, dtype=np.float32)
        self.state[ones] = 1
        self.step_count = 0
        
        return self.state.copy(), {}

class BitStringGame(EnvGame):
    def __init__(self, **kwargs):
        env = BitStringGym(**kwargs)
        super().__init__(env)
        self.action_mask = np.ones(env.n_sites, dtype=bool)  # All actions are always available
    
    def get_action_mask(self):
        return self.action_mask
    
    @property
    def hashable_obs(self) -> Hashable:
        "Returns a hashable representation of the current observation `obs`."
        return "".join([str(int(x)) for x in self.obs])  + " " + str(self.env.step_count)
    
