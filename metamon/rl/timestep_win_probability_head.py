from __future__ import annotations

"""
Timestep-only win probability head.

This module attaches the existing WinProbabilityHead MLP directly on top of the
AMAGO timestep encoder outputs (ignores the trajectory encoder). It mirrors the
API of the sequence-based wrapper but computes representations per step using
only the tstep encoder.

Orientation: inputs are always formed as our team vs opponent_team. The head
predicts our win probability from that oriented observation.
"""

from typing import Optional, Tuple, Type

import gin
import torch
from torch import nn

from metamon.rl.win_probability_head import WinProbabilityHead


@gin.configurable
class MetamonTstepWinProbAgent(nn.Module):
    """Attach WinProbabilityHead to the AMAGO timestep encoder output.

    The base agent remains frozen; this wrapper exposes helpers to obtain logits
    per timestep without using the trajectory encoder. The head is the only
    component intended to be trained. Observations are oriented as
    (our team, opponent team); predictions are our win probabilities.
    """

    def __init__(
        self,
        base_agent: "amago.agent.Agent",
        *,
        head_type: Type[WinProbabilityHead] = WinProbabilityHead,
        hidden_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        import amago  # type: ignore

        assert isinstance(base_agent, amago.agent.Agent)
        self.agent = base_agent
        # The timestep encoder exposes its embedding dimension via `emb_dim`.
        state_dim = self.agent.tstep_encoder.emb_dim
        self.head = head_type(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )

    @property
    def parameters_for_optimization(self):
        return self.head.parameters()

    @torch.no_grad()
    def _encode_batch(self, batch: "amago.loading.Batch") -> torch.Tensor:
        """Return timestep embeddings [B, L, D] from the frozen timestep encoder."""
        self.agent.eval()
        tstep_emb = self.agent.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        return tstep_emb

    @staticmethod
    def _last_true_idx(dones_bool_2d: torch.Tensor) -> torch.Tensor:
        """Robust last-True index per row for [B, L-1] dones.

        Falls back to the final transition if there are no True entries.
        """
        rev = torch.flip(dones_bool_2d, dims=[1]).int()
        idx_from_end = rev.argmax(dim=1)
        has_any = dones_bool_2d.any(dim=1)
        last = (dones_bool_2d.size(1) - 1) - idx_from_end
        fallback = torch.full_like(last, dones_bool_2d.size(1) - 1)
        return torch.where(has_any, last, fallback)

    def logits_all_steps_from_batch(
        self, batch: "amago.loading.Batch"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (tstep_emb [B,L,D], logits [B,L,1], valid_mask [B,L,1]).

        valid_mask excludes RL2 padding and the post-terminal timestep (i > last_done).
        """
        tstep_emb = self._encode_batch(batch)
        logits = self.head.forward_all_steps(tstep_emb)  # [B, L, 1]

        # Base validity from RL2 padding
        pad_val = self.agent.pad_val
        base_valid = (~((batch.rl2s == pad_val).all(-1, keepdim=True)))  # [B,L,1]

        # Exclude post-terminal timesteps
        dones = batch.dones.squeeze(-1)  # [B, L-1]
        B = dones.shape[0]
        L = batch.rl2s.shape[1]
        last_done = self._last_true_idx(dones.bool())  # [B]
        time = torch.arange(L, device=logits.device).unsqueeze(0)  # [1, L]
        not_future = (time <= last_done.unsqueeze(1)).unsqueeze(-1)  # [B,L,1]

        valid_mask = base_valid & not_future
        return tstep_emb, logits, valid_mask

    def current_step_logits_from_batch(
        self, batch: "amago.loading.Batch", t_idxs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tstep_emb_selected [B,D], logits_selected [B,1]) for indices t_idxs.

        Args:
            batch: amago Batch with fields obs, rl2s, time_idxs, dones
            t_idxs: Long tensor of shape [B] with selected timestep index per sequence.
        """
        tstep_emb = self._encode_batch(batch)  # [B, L, D]
        B = tstep_emb.shape[0]
        idx = t_idxs.to(tstep_emb.device)
        idx_b = torch.arange(B, device=tstep_emb.device)
        s_rep = tstep_emb[idx_b, idx]  # [B, D]
        # Route through the head's forward for consistency
        s_rep_1 = s_rep.unsqueeze(1)  # [B,1,D]
        ones_mask = torch.ones(B, 1, 1, device=tstep_emb.device, dtype=torch.float32)
        logits = self.head(s_rep_1, mask=ones_mask)  # [B,1]
        return s_rep, logits

    def compute_from_obs(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        *,
        use_model_mode: str = "eval",
    ) -> torch.Tensor:
        """Compute logits [B,1] from a single-timestep batch ([B,1,...] inputs)."""
        if use_model_mode == "train":
            self.head.train()
        else:
            self.head.eval()

        with torch.no_grad():
            self.agent.eval()
            tstep_emb = self.agent.tstep_encoder(obs=obs, rl2s=rl2s)  # [B,1,D]
            pad_val = self.agent.pad_val
            state_mask = (~((rl2s == pad_val).all(-1, keepdim=True))).float()  # [B,1,1]
        logits = self.head(tstep_emb, mask=state_mask)  # [B,1]
        return logits
