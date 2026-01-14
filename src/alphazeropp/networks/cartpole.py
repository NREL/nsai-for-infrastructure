from alphazeropp.networks.networks import TorchPolicyValueNet, PolicyValueNetModel
from alphazeropp.envs.cartpole import CartPoleGame
from .utils import get_device

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CartPolePolicyValueNet(TorchPolicyValueNet):
    save_file_name = "cartpole_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 1.0,
    }

    def __init__(self, random_seed = None, n_hidden_layers = 2, hidden_size = 128, training_params = {}, device = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        model = PolicyValueNetModel(input_size=4, output_size=2, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size)
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        
        self.DEVICE = get_device() if device is None else device
        print(f"Neural network training will occur on device '{self.DEVICE}'")

        
    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        model = model.to(self.DEVICE)
        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"], weight_decay=tp["weight_decay"])

        # The usual case is that `examples` comes from AlphaZero and is a list of tuples
        # that will need to be reformatted. For convenience, we also support the case where
        # `examples` is already in the proper format, perhaps because the network is being
        # tested outside the AlphaZero context; for this, pass `needs_reshape=False`.
        if needs_reshape:
            # PERF we could use a single Python loop for all three of these
            states = torch.from_numpy(np.array([state for state, (_, _) in examples], dtype=np.float32))
            policies = torch.from_numpy(np.array([policy for _, (policy, _) in examples], dtype=np.float32))
            values = torch.from_numpy(np.array([value for _, (_, value) in examples], dtype=np.float32))
            dataset = torch.utils.data.TensorDataset(states, policies, values)
        else:
            print("Skipping reshape of `examples`.")
            dataset = examples
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=tp["batch_size"], shuffle=True)
        print(f"Training with {len(train_loader)} batches of size {tp['batch_size']}")

        train_mini_losses = []
        train_losses = []

        for epoch in range(tp["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            for inputs, targets_policy, targets_value in train_loader:
                inputs, targets_value, targets_policy = inputs.to(self.DEVICE), targets_value.to(self.DEVICE), targets_policy.to(self.DEVICE)
                optimizer.zero_grad()
                assert len(inputs.shape) == 2, f"Expected input shape to be 2D, got {inputs.shape}"
                assert inputs.shape[1] == self.model.input_size, f"Expected input size {self.model.input_size}, got {inputs.shape[1]}"
                assert len(targets_value.shape) == 1
                assert targets_policy.shape[1] == CartPoleGame._ACTION_MASK.shape[0], f"Expected policy answers size {CartPoleGame._ACTION_MASK.shape[0]}, got {targets_policy.shape[1]}"
                outputs_policy, outputs_value = model(inputs)
                assert outputs_value.shape == targets_value.shape, f"Expected predicted value shape {targets_value.shape}, got {outputs_value.shape}"
                loss_value = criterion_value(outputs_value, targets_value)
                assert outputs_policy.shape == targets_policy.shape, f"Expected predicted policy shape {targets_policy.shape}, got {outputs_policy.shape}"
                loss_policy = criterion_policy(outputs_policy, targets_policy)
                loss = loss_value + policy_weight*loss_policy

                loss.backward()
                optimizer.step()
                loss = loss.item()
                train_mini_losses.append(loss)
                train_loss += loss
                policy_loss += loss_policy
                value_loss += loss_value

            train_losses.append(train_loss / len(train_loader))
            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
            # if True:
                print(f"Epoch {epoch+1}/{tp['epochs']}, Train Loss: {train_losses[-1]:.4f} (value: {value_loss / len(train_loader):.4f}, policy: {policy_loss / len(train_loader):.4f}, weighted policy: {policy_weight * (policy_loss / len(train_loader)):.4f})")

        return model, train_mini_losses, train_losses
    
    def predict(self, state):
        self.model.cpu()
        nn_input = torch.tensor(state).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)
        
        policy_prob = policy_prob.numpy()
        policy_prob = policy_prob.squeeze(0)
        assert policy_prob.shape == (2,)

        value = value.numpy()
        value = value.squeeze(0)
        assert value.shape == ()

        # policy_prob = np.random.random(2)
        return policy_prob, value
