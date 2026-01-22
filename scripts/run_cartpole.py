from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import numpy as np

from alphazeropp.agent import Agent
from alphazeropp.envs.cartpole import CartPoleGame
from alphazeropp.networks.cartpole import CartPolePolicyValueNet

if __name__ == "__main__":
    game = CartPoleGame()
    net = CartPolePolicyValueNet(random_seed=42)
    agent = Agent(game, net, random_seeds={"mcts": 48, "train": 49, "eval": 50})


    agent.play_train_multiple(100)