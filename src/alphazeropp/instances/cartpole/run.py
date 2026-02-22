import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator

from .game import CartPoleGame
from .network import CartPolePolicyValueNet

import copy as copy

def main():
    game = CartPoleGame()
    net = CartPolePolicyValueNet()
    agent = Agent(
        game=game,
        net=net,
        mcts_params={"n_simulations": 25, "temperature": 1.0, "c_exploration": 1.0},
        external_policy=None,
    )

    trainer = Trainer(
        agent=agent,
        net=net,
        game=game,
        n_games_per_train=100,
        n_past_iterations_to_train=20,
        n_procs=-1,
    )
    evaluator = Evaluator(n_games=20, n_procs=-1)
    
    for i in range(20):
        old_agent = copy.deepcopy(trainer.agent)
        trainer.train_multiple(n_iterations=1)
        new_agent = copy.deepcopy(trainer.agent)
        score = evaluator.pit(new_agent=new_agent, old_agent=old_agent, eval_seed=0, mcts_seed=1)
        if score >= 0.55:
            print("Keeping the new network")
            trainer.net = new_agent.net
            agent.net = new_agent.net
        else:
            print("Reverting to the old network")
            trainer.net = old_agent.net
            agent.net = old_agent.net


if __name__ == "__main__":
    main()
