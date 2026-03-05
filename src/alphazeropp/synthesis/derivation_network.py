"""
Transformer-based PolicyValueNet for DerivationGame.

Encodes the preorder-traversal observation (alternating type_id, param pairs)
as a token sequence, processes it with a small Transformer encoder, and
produces policy logits + scalar value from a learned [CLS] token.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazeropp.core.policy_value_net import TorchPolicyValueNet
from alphazeropp.utils import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw nn.Module
# ---------------------------------------------------------------------------

class DerivationTransformerModel(nn.Module):
    """Transformer encoder that reads a DerivationGame observation.

    Input:  flat obs tensor of shape (batch, 2 * budget [+ extra_features]).
    Output: (policy_logits: (batch, action_size), value: (batch,)).

    When ``extra_features > 0`` (e.g. for FactoredDerivationGame), the
    last ``extra_features`` floats of the observation are projected and
    added to the CLS token embedding before the transformer.
    """

    def __init__(
        self,
        budget: int,
        n_sites: int,
        action_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        extra_features: int = 0,
    ):
        super().__init__()
        self.budget = budget
        self.action_size = action_size
        self.extra_features = extra_features
        seq_len = budget  # one token per AST slot

        # --- embeddings ---
        self.type_emb = nn.Embedding(num_embeddings=9, embedding_dim=d_model)
        self.param_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(num_embeddings=seq_len + 1, embedding_dim=d_model)
        self.cls_emb = nn.Parameter(torch.zeros(d_model))

        # --- extra features projection (for factored game phase info) ---
        if extra_features > 0:
            self.extra_proj = nn.Linear(extra_features, d_model)

        # --- transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # --- heads ---
        self.policy_head = nn.Linear(d_model, action_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        L = self.budget

        # Split AST obs from extra features
        ast_obs = x[:, :2 * L]
        extra = x[:, 2 * L:] if self.extra_features > 0 else None

        # Parse obs into (type_id, param) pairs
        type_ids = ast_obs[:, 0::2].long()   # (B, L)
        params = ast_obs[:, 1::2]            # (B, L)

        # Padding mask: True = ignored.  CLS is never masked.
        pad_mask_tokens = (type_ids == 0)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, pad_mask_tokens], dim=1)  # (B, L+1)

        # Token embeddings
        type_e = self.type_emb(type_ids)            # (B, L, d)
        param_e = self.param_proj(params.unsqueeze(-1))  # (B, L, d)
        token_emb = type_e + param_e                # (B, L, d)

        # Prepend CLS (with extra features added if present)
        cls = self.cls_emb.unsqueeze(0).expand(B, -1)  # (B, d)
        if extra is not None:
            cls = cls + self.extra_proj(extra)          # (B, d)
        cls = cls.unsqueeze(1)                          # (B, 1, d)
        seq = torch.cat([cls, token_emb], dim=1)        # (B, L+1, d)

        # Positional embeddings
        positions = torch.arange(L + 1, device=x.device)
        seq = seq + self.pos_emb(positions)

        # Transformer
        hidden = self.transformer(seq, src_key_padding_mask=key_padding_mask)

        # CLS output -> heads
        cls_out = self.final_norm(hidden[:, 0, :])  # (B, d)
        policy_logits = self.policy_head(cls_out)    # (B, action_size)
        value = self.value_head(cls_out).squeeze(-1) # (B,)

        return policy_logits, value


# ---------------------------------------------------------------------------
# AlphaZero interface wrapper
# ---------------------------------------------------------------------------

class DerivationPolicyValueNet(TorchPolicyValueNet):
    """Transformer-based policy-value net for DerivationGame.

    Follows the same interface contract as BitStringPolicyValueNet.
    """

    save_file_name = "derivation_checkpoint.pt"

    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 2.0,
    }

    def __init__(
        self,
        budget: int,
        n_sites: int,
        action_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        extra_features: int = 0,
        training_params: dict = {},
        random_seed: int | None = None,
        device=None,
    ):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        model = DerivationTransformerModel(
            budget=budget,
            n_sites=n_sites,
            action_size=action_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            extra_features=extra_features,
        )
        self.budget = budget
        self.action_size = action_size
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        self.DEVICE = get_device() if device is None else device
        logger.info(
            f"DerivationPolicyValueNet: d_model={d_model}, n_layers={n_layers}, "
            f"device='{self.DEVICE}'"
        )

    # -- predict ---------------------------------------------------------------

    def predict(self, state):
        if next(self.model.parameters()).device.type != 'cpu':
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
