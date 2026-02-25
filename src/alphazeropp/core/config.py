from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
import json


TGame = TypeVar("TGame")
TNet = TypeVar("TNet")


@dataclass
class AgentConfig:
	mcts_params: dict[str, Any] = field(
		default_factory=lambda: {
			"n_simulations": 100,
			"temperature": 1.0,
			"c_exploration": 1.0,
		}
	)
	reward_discount: float = 1.0
	external_policy: Optional[Callable] = None
	random_seeds: dict[str, Any] = field(
		default_factory=lambda: {}
	)


@dataclass
class TrainerConfig:
	n_games_per_train: int = 100
	n_past_iterations_to_train: int = 20
	n_procs: int = 8
	checkpoint_dir: str = "checkpoints"


@dataclass
class EvaluatorConfig:
	n_games: int = 20
	n_procs: int = 8


@dataclass
class RunConfig:
	n_iterations: int = 100
	accept_threshold: float = 0.55
	plot_every: int = 5
	plot_path: str = "training_metrics.png"


@dataclass
class GameConfig:
	game_cls: Optional[Callable[..., TGame]] = None
	kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetConfig:
	net_cls: Optional[Callable[..., TNet]] = None
	kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaConfig(ABC):
	game: GameConfig = field(default_factory=GameConfig)
	net: NetConfig = field(default_factory=NetConfig)
	agent: AgentConfig = field(default_factory=AgentConfig)
	trainer: TrainerConfig = field(default_factory=TrainerConfig)
	evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
	run: RunConfig = field(default_factory=RunConfig)

	@abstractmethod
	def build(self):
		"""Return (game, net, agent, trainer, evaluator) for this config."""
		raise NotImplementedError
	def _to_serializable_dict(self) -> dict:
		"""Convert config to dict suitable for JSON serialization."""
		def convert(obj):
			if isinstance(obj, (str, int, float, bool, type(None))):
				return obj
			elif isinstance(obj, dict):
				return {k: convert(v) for k, v in obj.items()}
			elif isinstance(obj, (list, tuple)):
				return [convert(item) for item in obj]
			elif callable(obj):
				# Store callable as string representation
				if hasattr(obj, '__name__'):
					return f"<callable: {obj.__module__}.{obj.__name__}>"
				return f"<callable: {obj}>"
			elif hasattr(obj, '__dataclass_fields__'):
				# It's a dataclass
				return convert(asdict(obj))
			else:
				return str(obj)
		return convert(asdict(self))

	def save(self, path: str | Path) -> None:
		"""Save config to JSON file."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		
		config_dict = self._to_serializable_dict()
		with open(path, 'w') as f:
			json.dump(config_dict, f, indent=2)

	def plot(self, save_path: Optional[str | Path] = None) -> None:
		"""Plot config parameters as a figure using matplotlib."""
		try:
			import matplotlib.pyplot as plt
			import matplotlib.patches as mpatches
		except ImportError:
			print("[Warning] matplotlib not installed. Skipping plot generation.")
			return

		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		fig.suptitle("Config Overview", fontsize=16, fontweight='bold')

		# Plot 1: Agent Config
		ax = axes[0, 0]
		ax.axis('off')
		agent_text = f"""Agent Config
━━━━━━━━━━━━━━━━━━━
reward_discount: {self.agent.reward_discount}
external_policy: {self.agent.external_policy is not None}

MCTS Params:
  n_simulations: {self.agent.mcts_params.get('n_simulations', 'N/A')}
  temperature: {self.agent.mcts_params.get('temperature', 'N/A')}
  c_exploration: {self.agent.mcts_params.get('c_exploration', 'N/A')}
"""
		ax.text(0.1, 0.9, agent_text, transform=ax.transAxes, fontsize=10, 
		        verticalalignment='top', fontfamily='monospace',
		        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		# Plot 2: Trainer Config
		ax = axes[0, 1]
		ax.axis('off')
		trainer_text = f"""Trainer Config
━━━━━━━━━━━━━━━━━━━
n_games_per_train: {self.trainer.n_games_per_train}
n_past_iterations_to_train: {self.trainer.n_past_iterations_to_train}
n_procs: {self.trainer.n_procs}
checkpoint_dir: {self.trainer.checkpoint_dir}
"""
		ax.text(0.1, 0.9, trainer_text, transform=ax.transAxes, fontsize=10,
		        verticalalignment='top', fontfamily='monospace',
		        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

		# Plot 3: Evaluator Config
		ax = axes[0, 2]
		ax.axis('off')
		eval_text = f"""Evaluator Config
━━━━━━━━━━━━━━━━━━━
n_games: {self.evaluator.n_games}
n_procs: {self.evaluator.n_procs}
"""
		ax.text(0.1, 0.9, eval_text, transform=ax.transAxes, fontsize=10,
		        verticalalignment='top', fontfamily='monospace',
		        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

		# Plot 4: Game Config
		ax = axes[1, 0]
		ax.axis('off')
		game_cls_name = self.game.game_cls.__name__ if self.game.game_cls else "None"
		game_text = f"""Game Config
━━━━━━━━━━━━━━━━━━━
game_cls: {game_cls_name}
kwargs: {self.game.kwargs}
"""
		ax.text(0.1, 0.9, game_text, transform=ax.transAxes, fontsize=10,
		        verticalalignment='top', fontfamily='monospace',
		        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

		# Plot 5: Net Config
		ax = axes[1, 1]
		ax.axis('off')
		net_cls_name = self.net.net_cls.__name__ if self.net.net_cls else "None"
		net_text = f"""Net Config
━━━━━━━━━━━━━━━━━━━
net_cls: {net_cls_name}
kwargs: {self.net.kwargs}
"""
		ax.text(0.1, 0.9, net_text, transform=ax.transAxes, fontsize=10,
		        verticalalignment='top', fontfamily='monospace',
		        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

		# Plot 6: Run Config
		ax = axes[1, 2]
		ax.axis('off')
		run_text = f"""Run Config
━━━━━━━━━━━━━━━━━━━
n_iterations: {self.run.n_iterations}
accept_threshold: {self.run.accept_threshold}
plot_every: {self.run.plot_every}
plot_path: {self.run.plot_path}
"""
		ax.text(0.1, 0.9, run_text, transform=ax.transAxes, fontsize=10,
		        verticalalignment='top', fontfamily='monospace',
		        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

		plt.tight_layout()

		if save_path:
			save_path = Path(save_path)
			save_path.parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path, dpi=150, bbox_inches='tight')
			print(f"[Plot] Saved to {save_path}")

		plt.close(fig)