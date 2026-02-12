import time
import logging
import copy
from typing import Optional

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.utils.multiprocessing import MultiprocessingManager


logger = logging.getLogger(__name__)

# The following evaluator is not 100% complete.
# I need to come back later to check what is missing and how to implement it.

class Evaluator:
    """Evaluates and compares agents or networks."""

    def __init__(self, n_games: int = 20, n_procs: Optional[int] = None):
        self.n_games = n_games
        self.n_procs = n_procs
        
    def push_multiprocessing(self):
        ### It looks like we don't need to do anything here yet
        pass
    
    def pop_multiprocessing(self, *args):
        ### It looks like we don't need to do anything here yet
        pass

    def _play_for_eval(
        self,
        reset_seed,
        mcts_seed,
        new_agent: Agent,
        old_agent: Agent,
        try_without_mcts: bool = False,
    ):
        """
        Play one eval game for each agent and return rewards.
        """
        new_agent.game.reset_wrapper(seed=reset_seed)
        old_agent.game = copy.deepcopy(new_agent.game)
        results = {}
        
        old_result = old_agent.play_one_round(game=old_agent.game, random_seed=mcts_seed)
        new_result = new_agent.play_one_round(game=new_agent.game, random_seed=mcts_seed)
        
        old_sum_reward = sum(x[2] for x in old_result)
        new_sum_reward = sum(x[2] for x in new_result)
        results["old_net"] = old_sum_reward
        results["new_net"] = new_sum_reward
        if try_without_mcts:
            pass
        return results

    def pit(
        self,
        new_agent: Agent,
        old_agent: Agent,
        eval_seed: int,
        mcts_seed: int,
        try_without_mcts: bool = False,
    ) -> float:
        """
        Compare agent_new vs agent_old and return win rate of new agent.
        """
        start_time = time.time()

        mp_manager = MultiprocessingManager(new_agent.net, old_agent.net, self)
        mp_manager.push()
        try:
            arg_tuples = [
                (
                    eval_seed + i,
                    mcts_seed + i,
                    new_agent,
                    old_agent,
                    try_without_mcts,
                )
                for i in range(self.n_games)
            ]
            eval_results = MultiprocessingManager.starmap(
                self._play_for_eval,
                arg_tuples,
                self.n_procs,
            )
        finally:
            mp_manager.pop()

        old_rewards = np.array([r["old_net"] for r in eval_results])
        new_rewards = np.array([r["new_net"] for r in eval_results])

        print(new_rewards)
        
        wins = np.sum(new_rewards > old_rewards)
        ties = np.sum(np.isclose(new_rewards, old_rewards))
        losses = np.sum(new_rewards < old_rewards)
        score = (wins + ties / 2) / self.n_games

        logger.info(
            "Eval done in %.2fs: new wins=%d, ties=%d, losses=%d, score=%.2f%%",
            time.time() - start_time,
            wins,
            ties,
            losses,
            score * 100,
        )
        return score
