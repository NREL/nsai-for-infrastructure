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

    def _play_for_eval(
        self,
        reset_seed,
        mcts_seed,
        agent_new: Agent,
        agent_old: Agent,
        try_without_mcts: bool = False,
    ):
        """
        Play one eval game for each agent and return rewards.
        """
        agent_new.game.reset_wrapper(seed=reset_seed)
        agent_old.game = copy.deepcopy(agent_new.game)
        results = {}
        breakpoint() # Double check that the following code collects experience correctly.
        results["old_net"] = agent_old.play_one_round(game=agent_old.game, random_seed=mcts_seed)[-1][1][1]
        results["new_net"] = agent_new.play_one_round(game=agent_new.game, random_seed=mcts_seed)[-1][1][1]
        if try_without_mcts:
            pass

        return results

    def pit(
        self,
        agent_new: Agent,
        agent_old: Agent,
        eval_seed: int,
        mcts_seed: int,
        external_policy_seed: int,
        try_without_mcts: bool = False,
    ) -> float:
        """
        Compare agent_new vs agent_old and return win rate of new agent.
        """
        start_time = time.time()

        mp_manager = MultiprocessingManager(agent_new, agent_old, self)
        mp_manager.push()
        try:
            arg_tuples = [
                (
                    i,
                    eval_seed + i,
                    mcts_seed + i,
                    external_policy_seed + i,
                    agent_new,
                    agent_old,
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
