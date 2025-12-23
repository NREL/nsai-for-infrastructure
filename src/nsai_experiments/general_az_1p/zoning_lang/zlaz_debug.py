"A quick script for Zoning Language AlphaZero debugging"

from nsai_experiments.general_az_1p.setup_utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import logging

from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.zoning_lang.zoning_lang_az_impl import ZoningLangGame
from nsai_experiments.general_az_1p.zoning_lang.zoning_lang_az_impl import ZoningLangPolicyValueNet

def main():
    mygame = ZoningLangGame()
    mynet = ZoningLangPolicyValueNet(random_seed=47, training_params={"epochs": 10})
    myagent = Agent(mygame, mynet, n_procs=-1, random_seeds={"mcts": 48, "train": 49, "eval": 50})

    logging.getLogger().setLevel(logging.WARN)
    myagent.play_train_multiple(1)

if __name__ == "__main__":
    main()
