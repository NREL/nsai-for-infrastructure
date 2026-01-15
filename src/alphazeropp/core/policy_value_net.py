from typing import Any, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class PolicyValueNet(ABC):
    """Abstract base class for policy-value networks used in reinforcement learning."""

    @abstractmethod
    def __init__(self):
        """Initialize the policy-value network."""
        pass

    @abstractmethod
    def train(self, examples: Iterable[tuple[Any, tuple[Any, Any]]]):
        """
        Train the network on examples.

        Args:
            examples: Training examples as tuples of (state, (policy, value)),
                where state is in the format returned by the game.
        """
        pass

    @abstractmethod
    def predict(self, state) -> tuple[Any, Any]:
        """
        Predict policy and value for a given state.

        Args:
            state: State observation in the format returned by the game.

        Returns:
            Tuple of (policy, value) predictions.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir):
        """
        Save the model checkpoint to the given directory.

        Args:
            save_dir: Directory path where the checkpoint will be saved.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, save_dir):
        """
        Load the model checkpoint from the given directory.

        Args:
            save_dir: Directory path from which to load the checkpoint.
        """
        pass

    @abstractmethod
    def push_multiprocessing(self):
        """
        Prepare the instance for Python multiprocessing.

        Moves tensors from GPU to CPU and returns any state that should be restored
        after multiprocessing via pop_multiprocessing().
        """
        pass

    @abstractmethod
    def pop_multiprocessing(self, *args):
        """
        Restore the instance after multiprocessing.

        Args:
            *args: State returned from push_multiprocessing() to be restored.
        """
        pass

class TorchPolicyValueNet(PolicyValueNet):
    """
    Abstract base class for PyTorch-based policy-value networks.

    Contains a field `model: nn.Module` populated in the constructor; implements
    `save_checkpoint` and `load_checkpoint` methods that save and load this `model`.
    Users still need to implement `train` and `predict` methods to define their own
    training logic and any transformations that may need to happen before or after
    calling `model.forward()`. Functional defaults for `push_multiprocessing` and
    `pop_multiprocessing` are provided, but override these if needed for additional
    state (e.g., an optimizer).
    """

    model: nn.Module
    save_file_name: str = "model_checkpoint.pt"

    def __init__(self, model: nn.Module):
        """
        Initialize the torch policy-value network.

        Args:
            model: The neural network model to use for predictions.
        """
        super().__init__()
        self.model = model

    def save_checkpoint(self, save_dir):
        """
        Save the model checkpoint to the given directory.

        Args:
            save_dir: Directory path where the checkpoint will be saved.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / self.save_file_name)

    def load_checkpoint(self, save_dir):
        """
        Load the model checkpoint from the given directory.

        Args:
            save_dir: Directory path from which to load the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        save_dir = Path(save_dir)
        save_file = save_dir / self.save_file_name
        if not save_file.exists():
            raise FileNotFoundError(f"Checkpoint file {save_file} does not exist.")
        self.model.load_state_dict(torch.load(save_file))

    def push_multiprocessing(self):
        """Prepare the model for multiprocessing by moving it to CPU."""
        self.model.cpu()

    def pop_multiprocessing(self, *args):
        """Restore the model after multiprocessing."""
        pass

class PolicyValueNetModel(nn.Module):
    """
    Neural network with separate policy and value heads for reinforcement learning.

    This network uses a shared body with separate heads for policy and value predictions.
    """

    def __init__(self, input_size: int, output_size: int, n_hidden_layers: int = 2,
                 hidden_size: int = 128):
        """
        Initialize the policy-value network model.

        Args:
            input_size: Dimension of the input state.
            output_size: Dimension of the output policy.
            n_hidden_layers: Number of hidden layers in the body (default: 2).
            hidden_size: Number of neurons in each hidden layer (default: 128).
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.body = nn.Sequential(
            nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
              for _ in range(n_hidden_layers)],
        )
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input state tensor.

        Returns:
            Tuple of (policy_logits, value) predictions.
        """
        x = self.body(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value