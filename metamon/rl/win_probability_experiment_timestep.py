from __future__ import annotations

"""
Experiment exposing timestep-only win probability from the current env state.

This reads the current state from the envs, runs only the frozen timestep
encoder, applies the timestep win-probability head, and returns probabilities.
No trajectory encoder hidden state is used or maintained.
"""

from typing import Optional

import gin
import numpy as np
import torch

import amago
from amago import utils as amago_utils

from metamon.rl.timestep_win_probability_head import MetamonTstepWinProbAgent


@gin.configurable
class MetamonTstepWinProbExperiment(amago.Experiment):
    """Attach a timestep-only win probability head and expose inference API.

    Does not change the base policy. Only computes P(win) from the current
    timestep using the timestep encoder output and a small MLP head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._winprob: Optional[MetamonTstepWinProbAgent] = None
        self._temperature: float = 1.0

    def init_model(self) -> None:
        super().init_model()
        self._winprob = MetamonTstepWinProbAgent(base_agent=self.policy)

    @torch.no_grad()
    def compute_prob_current_state(self, envs) -> np.ndarray:
        """Return per-env win probability for the current timestep.

        Reads env state via `current_timestep`, encodes it with the timestep
        encoder only, and applies the win-probability head.
        """
        assert self._winprob is not None

        # Fetch current batched obs, rl2s, time_idxs from envs
        par_obs_rl2_time = amago_utils.call_async_env(envs, "current_timestep")
        _obs, _rl2s, _time_idxs = [], [], []
        for o, r, t in par_obs_rl2_time:
            _obs.append(o)
            _rl2s.append(r)
            _time_idxs.append(t)

        obs = {
            k: torch.from_numpy(v).to(self.DEVICE).unsqueeze(1)
            for k, v in amago_utils.stack_list_array_dicts(_obs, axis=0, cat=True).items()
        }
        rl2s = torch.from_numpy(np.concatenate(_rl2s, axis=0)).to(self.DEVICE).unsqueeze(1)
        # time_idxs not needed for timestep-only path

        logits = self._winprob.compute_from_obs(obs=obs, rl2s=rl2s)  # [B,1]
        if getattr(self, "_temperature", 1.0) != 1.0:
            logits = logits / float(self._temperature)
        prob = torch.sigmoid(logits)
        return prob.squeeze(-1).detach().cpu().numpy()

    def load_timestep_head(self, checkpoint_path: str) -> None:
        """Load a saved timestep winprob head checkpoint into the wrapper head."""
        assert self._winprob is not None
        state = torch.load(checkpoint_path, map_location=self.DEVICE)
        self._winprob.head.load_state_dict(state["head_state_dict"])
        self._temperature = float(state.get("temperature", 1.0))
