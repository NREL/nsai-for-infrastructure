from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import numpy as np

from alphazeropp.envs.bitstring import BitStringGame
from alphazeropp.networks.bitstring import BitStringPolicyValueNet
from alphazeropp.agent.bitstring_agent import BitStringAgent

if __name__ == "__main__":
    nsites = 10
    game = BitStringGame(nsites=nsites)
    net = BitStringPolicyValueNet(nsites=nsites, random_seed=42)
    agent = BitStringAgent(game, net, n_games_per_train=100, n_games_per_eval=10, threshold_to_keep=0.4,  n_past_iterations_to_train=5,
                    random_seeds={"mcts": 48, "train": 49, "eval": 50}, mcts_params={"n_simulations": 30, "c_exploration": 0.4})
    agent.play_train_multiple(100)