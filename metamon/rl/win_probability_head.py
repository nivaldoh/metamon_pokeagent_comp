from __future__ import annotations

"""
Win probability head and a thin wrapper for running it on top of AMAGO's
sequence encoder without changing the base policy logic.

Usage (single process):
    exp = MetamonWinProbExperiment(...)
    exp.start()  # or call init_model() after envs/dsets if you prefer
    # In a loop driven by an external controller that steps envs:
    probs = exp.compute_prob_current_state(envs)  # np.ndarray [B]

This module does not generate actions. It only reads the current env state,
encodes it through tstep_encoder + traj_encoder, and returns a probability [0,1].
"""

from typing import Optional, Tuple, Literal, Type

import gin
import torch
from torch import nn
import torch.nn.functional as F


def _act(name: str):
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.LeakyReLU()


@gin.configurable
class WinProbabilityHead(nn.Module):
    """Small MLP over pooled sequence encodings to predict win probability logits.

    forward(...) returns logits of shape [B, 1]. Use predict_proba(...) to get
    probabilities in [0, 1].
    """

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool_last(self, s_rep: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # s_rep: [B, L, D]; mask: [B, L, 1] with 1/True where valid
        B, L, D = s_rep.shape
        if mask is None:
            return s_rep[:, -1]  # [B, D]
        valid = mask.float().squeeze(-1) > 0.0  # [B, L]
        # Reverse + argmax = index of last True; if all False, argmax=0 -> fallback L-1
        idx_from_end = torch.argmax(torch.flip(valid, dims=[1]).to(torch.long), dim=1)
        last_idxs = (L - 1) - idx_from_end
        return s_rep[torch.arange(B, device=s_rep.device), last_idxs]

    def forward(self, s_rep: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = self._pool_last(s_rep, mask)  # [B, D]
        logits = self.net(pooled)  # [B, 1]
        return logits

    @torch.no_grad()
    def predict_proba(self, s_rep: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience: sigmoid(head(s_rep)). Returns [B, 1] float in [0,1]."""
        return torch.sigmoid(self.forward(s_rep, mask))

    def forward_all_steps(self, s_rep: torch.Tensor) -> torch.Tensor:
        """Apply the head at every timestep. Returns logits [B, L, 1]."""
        return self.net(s_rep)


@gin.configurable
class MetamonWinProbAgent(nn.Module):
    """Attach WinProbabilityHead to an existing AMAGO Agent and expose helpers.

    This wrapper does not change the Agent's policy. It only computes the win
    probability from the shared backbone encodings.
    """

    def __init__(
        self,
        base_agent: "amago.agent.Agent",
        *,
        head_type: Type[WinProbabilityHead] = WinProbabilityHead,
        hidden_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        import amago  # type: ignore

        assert isinstance(base_agent, amago.agent.Agent)
        self.agent = base_agent
        self.head = head_type(
            state_dim=self.agent.traj_encoder.emb_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )

    @property
    def parameters_for_optimization(self):
        return self.head.parameters()

    def compute_from_obs(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        *,
        hidden_state=None,
        use_model_mode: Literal["train", "eval"] = "eval",
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (s_rep, logits).

        Backbone runs under no_grad() in eval mode; gradients flow only through the head.
        """
        import amago  # type: ignore
        # Head mode
        if use_model_mode == "train":
            self.head.train()
        else:
            self.head.eval()

        # Backbone forward under no_grad. If hidden_state is None, the Transformer
        # expects training=True for the training_forward path. Temporarily toggle
        # the relevant modules to train mode to satisfy assertion, but keep no_grad.
        with torch.no_grad():
            # Keep frozen backbone deterministic
            self.agent.eval()
            # If no hidden_state provided, allocate a temporary one for inference path
            if hidden_state is None:
                B = rl2s.shape[0]
                device = rl2s.device
                hidden_state = self.agent.traj_encoder.init_hidden_state(B, device)
            tstep_emb = self.agent.tstep_encoder(obs=obs, rl2s=rl2s)
            s_rep, _ = self.agent.traj_encoder(
                seq=tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
            )
        logits = self.head(s_rep, mask=mask)
        return s_rep, logits

    def compute_from_batch(
        self,
        batch: "amago.loading.Batch",
        *,
        use_model_mode: Literal["train", "eval"] = "eval",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (s_rep, logits). No @torch.no_grad() so the head can learn; backbone is eval+no_grad."""
        import amago  # type: ignore
        if use_model_mode == "train":
            self.head.train()
        else:
            self.head.eval()

        with torch.no_grad():
            # frozen backbone, deterministic encodings using inference path
            self.agent.eval()
            B = batch.rl2s.shape[0]
            device = batch.rl2s.device
            hs = self.agent.traj_encoder.init_hidden_state(B, device)
            o = self.agent.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            s_rep, _ = self.agent.traj_encoder(
                seq=o, time_idxs=batch.time_idxs, hidden_state=hs
            )
        # Build mask like Agent's state mask (True/1 where valid)
        pad_val = self.agent.pad_val
        state_mask = (~((batch.rl2s == pad_val).all(-1, keepdim=True))).float()
        logits = self.head(s_rep, mask=state_mask)
        return s_rep, logits

    @torch.no_grad()
    def _encode_batch(self, batch: "amago.loading.Batch") -> torch.Tensor:
        """Frozen backbone encodings [B, L, D] in eval+no_grad using inference path."""
        self.agent.eval()
        B = batch.rl2s.shape[0]
        device = batch.rl2s.device
        hs = self.agent.traj_encoder.init_hidden_state(B, device)
        o = self.agent.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        s_rep, _ = self.agent.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=hs)
        return s_rep

    def logits_all_steps_from_batch(
        self, batch: "amago.loading.Batch"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (s_rep [B,L,D], logits [B,L,1], valid_mask [B,L,1]).

        valid_mask excludes padding and the post-terminal timestep (i = last_done + 1).
        """
        s_rep = self._encode_batch(batch)
        logits = self.head.forward_all_steps(s_rep)  # [B, L, 1]

        # Base validity from RL2 padding
        pad_val = self.agent.pad_val
        base_valid = (~((batch.rl2s == pad_val).all(-1, keepdim=True)))  # [B,L,1]

        # Exclude post-terminal timestep (i = argmax(dones) + 1)
        dones = batch.dones.squeeze(-1)  # [B, L-1]
        B, Lm1 = dones.shape
        L = batch.rl2s.shape[1]
        last_done = torch.argmax(dones.int(), dim=1)  # [B]
        time = torch.arange(L, device=logits.device).unsqueeze(0)  # [1,L]
        not_future = (time <= last_done.unsqueeze(1)).unsqueeze(-1)  # [B,L,1]

        valid_mask = base_valid & not_future
        return s_rep, logits, valid_mask
