"""
MLP-based PolicyValueNet for ScanDerivationGame.

Uses a simple feedforward network since the scan observation is a
fixed-size flat vector (no sequential structure to exploit).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazeropp.core.policy_value_net import TorchPolicyValueNet, PolicyValueNetModel
from alphazeropp.utils import get_device

logger = logging.getLogger(__name__)


class ScanPolicyValueNet(TorchPolicyValueNet):
    """MLP-based policy-value net for ScanDerivationGame."""

    save_file_name = "scan_checkpoint.pt"

    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "policy_weight": 2.0,
    }

    def __init__(
        self,
        n_sites: int,
        d_hidden: int = 128,
        n_hidden_layers: int = 2,
        training_params: dict = {},
        random_seed: int | None = None,
        device=None,
    ):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        input_size = 2 * n_sites
        action_size = n_sites
        model = PolicyValueNetModel(
            input_size=input_size,
            output_size=action_size,
            n_hidden_layers=n_hidden_layers,
            hidden_size=d_hidden,
        )
        self.n_sites = n_sites
        self.action_size = action_size
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        self.DEVICE = get_device() if device is None else device
        logger.info(
            f"ScanPolicyValueNet: d_hidden={d_hidden}, n_layers={n_hidden_layers}, "
            f"device='{self.DEVICE}'"
        )

    # -- predict ---------------------------------------------------------------

    def predict(self, state):
        self.model.cpu()
        nn_input = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            policy_logits, value = self.model(nn_input)
            policy_prob = F.softmax(policy_logits, dim=-1)

        policy_prob = policy_prob.numpy().squeeze(0)
        value = value.numpy().squeeze(0)

        assert policy_prob.shape == (self.action_size,), (
            f"Expected policy shape ({self.action_size},), got {policy_prob.shape}"
        )
        assert value.shape == (), f"Expected scalar value, got shape {value.shape}"
        return policy_prob, value

    # -- train -----------------------------------------------------------------

    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tp["learning_rate"],
            weight_decay=tp["weight_decay"],
        )

        if needs_reshape:
            states = torch.from_numpy(
                np.array([s for s, _, _ in examples], dtype=np.float32)
            )
            policies = torch.from_numpy(
                np.array([p for _, p, _ in examples], dtype=np.float32)
            )
            values = torch.from_numpy(
                np.array([v for _, _, v in examples], dtype=np.float32)
            )
            dataset = torch.utils.data.TensorDataset(states, policies, values)
        else:
            dataset = examples

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=tp["batch_size"], shuffle=True
        )

        train_batch_losses = []
        train_losses = []
        policy_losses = []
        value_losses = []

        for epoch in range(tp["epochs"]):
            model.train()
            epoch_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0

            for inputs, targets_policy, targets_value in train_loader:
                inputs = inputs.to(self.DEVICE)
                targets_policy = targets_policy.to(self.DEVICE)
                targets_value = targets_value.to(self.DEVICE)

                optimizer.zero_grad()
                outputs_policy, outputs_value = model(inputs)

                loss_value = criterion_value(outputs_value, targets_value)
                loss_policy = criterion_policy(outputs_policy, targets_policy)
                loss = loss_value + policy_weight * loss_policy

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                train_batch_losses.append(batch_loss)
                epoch_loss += batch_loss
                epoch_policy_loss += loss_policy.item()
                epoch_value_loss += loss_value.item()

            n_batches = len(train_loader)
            train_losses.append(epoch_loss / n_batches)
            policy_losses.append(epoch_policy_loss / n_batches)
            value_losses.append(epoch_value_loss / n_batches)

            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                logger.info(
                    f"Epoch {epoch+1}/{tp['epochs']}, "
                    f"Loss: {train_losses[-1]:.4f} "
                    f"(value: {value_losses[-1]:.4f}, "
                    f"policy: {policy_losses[-1]:.4f}, "
                    f"weighted policy: {policy_weight * policy_losses[-1]:.4f})"
                )

        return model, train_batch_losses, train_losses, policy_losses, value_losses
