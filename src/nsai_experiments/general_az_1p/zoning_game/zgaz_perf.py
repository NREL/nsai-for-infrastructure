"A quick script for Zoning Game AlphaZero performance profiling"

import logging

from nsai_experiments.general_az_1p.game import Game
from nsai_experiments.general_az_1p.policy_value_net import PolicyValueNet
from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGameGame
from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGamePolicyValueNet

def main():
    mygame = ZoningGameGame()
    mynet = ZoningGamePolicyValueNet(training_params={"epochs": 10})
    myagent = Agent(mygame, mynet)

    logging.getLogger().setLevel(logging.WARN)
    myagent.play_train_multiple(1)

if __name__ == "__main__":
    main()
