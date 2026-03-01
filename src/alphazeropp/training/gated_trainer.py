import copy
import logging

from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator


logger = logging.getLogger(__name__)


class GatedTrainer:
    """Wraps Trainer + Evaluator with accept/reject gating.

    After each training iteration the candidate network is pitted against the
    previous best.  If the candidate wins at least ``acceptance_threshold`` of
    the evaluation games it is kept; otherwise the old weights are restored.
    """

    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        acceptance_threshold: float = 0.55,
    ):
        self.trainer = trainer
        self.evaluator = evaluator
        self.acceptance_threshold = acceptance_threshold

    def train_iteration(self) -> tuple[float, bool]:
        """Run one gated training iteration.

        Returns:
            (score, accepted) where *score* is the pit win-rate of the new
            network and *accepted* indicates whether the new weights were kept.
        """
        # 1. Snapshot old agent (full deepcopy – proven pattern from run_cartpole.py)
        old_agent = copy.deepcopy(self.trainer.agent)

        # 2. Train (modifies net weights in-place)
        self.trainer.train_iteration()

        # 3. Snapshot new agent for pit
        new_agent = copy.deepcopy(self.trainer.agent)

        # 4. Pit new vs old
        score = self.evaluator.pit(new_agent=new_agent, old_agent=old_agent)

        # 5. Gate decision
        accepted = score >= self.acceptance_threshold
        if not accepted:
            # Restore old weights IN-PLACE via load_state_dict.
            # trainer.net and trainer.agent.net are the same object,
            # so one load_state_dict updates both references.
            old_state_dict = old_agent.net.model.state_dict()
            self.trainer.net.model.load_state_dict(old_state_dict)

        # 6. One-line summary: score, accepted?, old_mean, new_mean
        eval_record = self.evaluator.statistics_manager._records[-1]
        logger.info(
            "Gate: score=%.1f%% %s | new_mean=%.3f old_mean=%.3f",
            score * 100,
            "ACCEPTED" if accepted else "REJECTED",
            eval_record.get("new_rewards_mean", 0),
            eval_record.get("old_rewards_mean", 0),
        )

        # 7. Record gate stats on trainer's statistics manager
        self.trainer.statistics_manager.record({
            "gate_score": score,
            "gate_accepted": int(accepted),
        })

        return score, accepted
