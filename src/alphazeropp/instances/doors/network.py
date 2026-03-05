"""DoorsDirectNet: MLP policy-value network for direct-play Doors."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazeropp.core.policy_value_net import TorchPolicyValueNet, PolicyValueNetModel
from alphazeropp.utils import get_device

logger = logging.getLogger(__name__)


class DoorsDirectNet(TorchPolicyValueNet):
    """MLP policy-value network for direct-play Doors.

    Architecture: input_size → 64 → 64 → (policy head, value head)
    """

    save_file_name = "doors_direct_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "policy_weight": 2.0,
    }

    def __init__(self, input_size=7, output_size=7,
                 n_hidden_layers=1, hidden_size=64,
                 random_seed=None, training_params=None, device=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        model = PolicyValueNetModel(
            input_size=input_size, output_size=output_size,
            n_hidden_layers=n_hidden_layers, hidden_size=hidden_size,
        )
        super().__init__(model)
        self.training_params = self.default_training_params | (training_params or {})
        self.DEVICE = get_device() if device is None else device
        logger.info("DoorsDirectNet training on device '%s'", self.DEVICE)

        # Persistent optimizer -- retains momentum/adaptive state across train() calls
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
        )

    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = self.optimizer

        if needs_reshape:
            states = torch.from_numpy(
                np.array([s for s, _, _ in examples], dtype=np.float32))
            policies = torch.from_numpy(
                np.array([p for _, p, _ in examples], dtype=np.float32))
            values = torch.from_numpy(
                np.array([v for _, _, v in examples], dtype=np.float32))
            dataset = torch.utils.data.TensorDataset(states, policies, values)
        else:
            dataset = examples

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=tp["batch_size"], shuffle=True)

        train_batch_losses = []
        train_losses = []
        policy_losses = []
        value_losses = []

        for epoch in range(tp["epochs"]):
            model.train()
            epoch_loss = 0.0
            epoch_ploss = 0.0
            epoch_vloss = 0.0

            for inputs, targets_policy, targets_value in loader:
                inputs = inputs.to(self.DEVICE)
                targets_policy = targets_policy.to(self.DEVICE)
                targets_value = targets_value.to(self.DEVICE)

                assert inputs.shape[1] == self.model.input_size
                assert targets_policy.shape[1] == self.model.output_size

                optimizer.zero_grad()
                out_policy, out_value = model(inputs)

                loss_v = criterion_value(out_value, targets_value)
                loss_p = criterion_policy(out_policy, targets_policy)
                loss = loss_v + policy_weight * loss_p

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()
                train_batch_losses.append(batch_loss)
                epoch_loss += batch_loss
                epoch_ploss += loss_p.item()
                epoch_vloss += loss_v.item()

            n_batches = len(loader)
            train_losses.append(epoch_loss / n_batches)
            policy_losses.append(epoch_ploss / n_batches)
            value_losses.append(epoch_vloss / n_batches)

            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                logging.info(
                    "Epoch %d/%d, Loss: %.4f (value: %.4f, policy: %.4f)",
                    epoch + 1, tp["epochs"],
                    train_losses[-1], value_losses[-1], policy_losses[-1],
                )

        return model, train_batch_losses, train_losses, policy_losses, value_losses

    def predict(self, state):
        self.model.cpu()
        nn_input = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)

        policy_prob = policy_prob.numpy().squeeze(0)
        assert policy_prob.shape == (self.model.output_size,)

        value = value.numpy().squeeze(0)
        assert value.shape == ()

        return policy_prob, value
