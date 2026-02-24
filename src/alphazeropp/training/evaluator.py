import time
import logging
import copy
from typing import Optional

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.utils.multiprocessing import MultiprocessingManager
from alphazeropp.utils.statistics import StatisticsManager


logger = logging.getLogger(__name__)

# The following evaluator is not 100% complete.
# I need to come back later to check what is missing and how to implement it.

class Evaluator:
    """Evaluates and compares agents or networks."""

    def __init__(self, n_games: int = 20, n_procs: Optional[int] = None):
        self.n_games = n_games
        self.n_procs = n_procs
        self.statistics_manager = StatisticsManager()
        
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
        base_game = new_agent.game.stash_state()
        base_game.reset_wrapper(seed=reset_seed)
        old_game = copy.deepcopy(base_game)
        new_game = copy.deepcopy(base_game)
        results = {}
        
        old_trajectory, old_cumulative_reward = old_agent.play_one_round(game=old_game, random_seed=mcts_seed)
        new_trajectory, new_cumulative_reward = new_agent.play_one_round(game=new_game, random_seed=mcts_seed)
        # old_sum_reward = sum(x[2] for x in old_result)
        # new_sum_reward = sum(x[2] for x in new_result)
        results["old_net"] = old_cumulative_reward
        results["new_net"] = new_cumulative_reward
        if try_without_mcts:
            pass
        return results

    def pit(
        self,
        new_agent: Agent,
        old_agent: Agent,
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
                    old_agent._randseed("eval"),
                    old_agent._randseed("mcts"),
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

        statistics = {
            "new_rewards_mean": np.mean(new_rewards),
            "old_rewards_mean": np.mean(old_rewards),
            "new_rewards_std": np.std(new_rewards),
            "old_rewards_std": np.std(old_rewards),
        }
        self.statistics_manager.record(statistics)

        print("new rewards:", new_rewards)
        print("old rewards:", old_rewards)
        
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
