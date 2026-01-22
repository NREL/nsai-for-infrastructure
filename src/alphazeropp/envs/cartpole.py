import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazeropp.envs.game import EnvGame

class CumulativeRewardWrapper(gym.Wrapper):
    """Wrapper that changes reward behavior: 0 at every step, total steps at termination."""
    
    def __init__(self, env, max_steps = 100):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = max_steps
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Give 0 reward during the episode, final reward at termination
        if terminated or truncated:
            # print(f"AM TERMINATED {terminated} OR TRUNCATED {truncated}")
            reward = self.step_count / self.max_steps
        else:
            reward = 0

        assert reward == 0.0 or (terminated or truncated)
        assert reward <= 1.0
            
        return observation, reward, terminated, truncated, info
    

class CartPoleGame(EnvGame):
    _ACTION_MASK = np.array([True, True])  # Both actions are always available

    def __init__(self, use_cumulative_reward_rescale=True, max_steps=100, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        if use_cumulative_reward_rescale:
            env = CumulativeRewardWrapper(env, max_steps=max_steps)
        super().__init__(env)
    
    def get_action_mask(self):
        return self._ACTION_MASK