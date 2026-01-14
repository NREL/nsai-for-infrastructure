"Utils that rely on PyTorch and such"

import torch
import gymnasium as gym

def get_device():
    # The following would work on recent PyTorch:
    # (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    # but we want to support PyTorch 2.2, so:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"

# multiprocessing doesn't like anonymous functions so we can't use TransformReward(... lambda...)
class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale