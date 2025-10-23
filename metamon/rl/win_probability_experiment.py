from __future__ import annotations

"""
Experiment subclass that exposes win probability from the current env state
without generating actions. External modules can drive env.step() and call
`compute_prob_current_state(envs)` to get a [B] array of probabilities.
"""

from typing import Optional, Iterable

import gin
import numpy as np
import torch

import amago
from amago import utils as amago_utils

from metamon.rl.win_probability_head import MetamonWinProbAgent


@gin.configurable
class MetamonWinProbExperiment(amago.Experiment):
    """Attach a win probability head and expose a read-only inference API.

    This experiment does not alter training or inference loops of AMAGO. It only
    adds a helper to encode the current env state and return probabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._winprob: Optional[MetamonWinProbAgent] = None
        self._winprob_hidden = None
        self._last_time_idxs: Optional[torch.Tensor] = None

    def init_model(self) -> None:
        super().init_model()
        # Attach head wrapper to the unwrapped policy
        self._winprob = MetamonWinProbAgent(base_agent=self.policy)
        # Persistent hidden state for traj encoder during polling
        self._winprob_hidden = self.policy.traj_encoder.init_hidden_state(
            self.parallel_actors, self.DEVICE
        )
        self._last_time_idxs = torch.zeros(
            (self.parallel_actors,), dtype=torch.long, device=self.DEVICE
        )

    @torch.no_grad()
    def compute_prob_current_state(self, envs) -> np.ndarray:
        """Return per-env win probability for the current timestep.

        Does not call env.step(). It reads env state via current_timestep,
        encodes it with the model (tstep_encoder + traj_encoder), and applies
        the win probability head to the last valid step.

        Args:
            envs: The same vectorized env pool used by the controller driving step().

        Returns:
            np.ndarray of shape [B] with probabilities in [0, 1].
        """
        assert self._winprob is not None and self._last_time_idxs is not None

        # Fetch current batched obs, rl2s, time_idxs from envs
        par_obs_rl2_time = amago_utils.call_async_env(envs, "current_timestep")
        _obs, _rl2s, _time_idxs = [], [], []
        for o, r, t in par_obs_rl2_time:
            _obs.append(o)
            _rl2s.append(r)
            _time_idxs.append(t)
        # stack and to torch on device; add length dim
        obs = {k: torch.from_numpy(v).to(self.DEVICE).unsqueeze(1) for k, v in amago_utils.stack_list_array_dicts(_obs, axis=0, cat=True).items()}
        rl2s = torch.from_numpy(np.concatenate(_rl2s, axis=0)).to(self.DEVICE).unsqueeze(1)
        time_idxs = torch.from_numpy(np.concatenate(_time_idxs, axis=0)).to(self.DEVICE).unsqueeze(1)

        # Reset hidden state where episode restarted (time_idx decreased)
        cur = time_idxs.squeeze(-1).squeeze(1)
        # numpy bool mask expected by reset_hidden_state
        dones = (cur < self._last_time_idxs).detach().cpu().numpy()
        self._winprob_hidden = self.policy.traj_encoder.reset_hidden_state(
            self._winprob_hidden, dones
        )
        self._last_time_idxs = cur

        # Build state mask for pooling: 1 where valid (like Agent state mask, with L=1)
        # rl2s has shape [B, 1, act+1]; MAGIC_PAD_VAL marks padding at invalid timesteps
        pad_val = amago.loading.MAGIC_PAD_VAL
        with torch.no_grad():
            state_mask = (~((rl2s == pad_val).all(-1, keepdim=True))).float()

        # Encode + compute probability using persistent hidden state
        tstep_emb = self.policy.tstep_encoder(obs=obs, rl2s=rl2s)
        s_rep, self._winprob_hidden = self.policy.traj_encoder(
            seq=tstep_emb, time_idxs=time_idxs, hidden_state=self._winprob_hidden
        )
        logits = self._winprob.head(s_rep, mask=state_mask)  # [B, 1]
        prob = torch.sigmoid(logits)
        return prob.squeeze(-1).detach().cpu().numpy()
