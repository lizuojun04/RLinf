# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Light-weight SAFE (NeurIPS 2025) failure-detector for online scoring inside
RLinf rollout.

This module reconstructs the *architecture only* of the trained SAFE detectors
(``indep`` = MLP and ``lstm``) directly from a ``state_dict`` checkpoint and runs
single-step inference. It deliberately does **not** import the standalone SAFE
repository: the whole codebase is a pure-PyTorch re-implementation kept in sync
with ``SAFE/failure_prob/model/{indep,lstm}.py``.

Feature contract
----------------
The detector expects, at each policy-inference (replan) step, the raw ``suffix_out``
hidden state produced by the openpi PaliGemma backbone, of shape
``(B, num_denoise_steps, n_pred_horizon, hidden_dim)`` (e.g. ``(10, 5, 1024)``).
This is exactly what ``rlinf .../openpi_action_model.py::get_suffix_out`` exports
when ``OpenPi0Config.collect_hidden_states`` is enabled (SAFE ``pre_velocity``).

Preprocessing mirrors ``SAFE/failure_prob/data/pizero.py::load_rollouts_from_root``
with the dataset config ``horizon_idx_rel="mean"`` and ``diff_idx_rel="mean"``:

    1. mean over ``n_pred_horizon`` (axis=-2)  -> ``(B, num_denoise_steps, hidden_dim)``
    2. mean over ``num_denoise_steps`` (axis=-2)-> ``(B, hidden_dim)``

The resulting per-step feature is fed to the SAFE model, which produces a scalar
*score* per env. Higher score = more likely to be a failure (the convention for
``lower_bound=False`` functional-CP bands).
"""

import numpy as np
import torch
import torch.nn as nn


def preprocess_suffix_out(hidden_states) -> torch.Tensor:
    """Reduce a raw openpi suffix_out tensor to per-env 1-D features.

    Args:
        hidden_states: tf/torch tensor or ndarray of shape
            ``(B, num_denoise_steps, n_pred_horizon, hidden_dim)``.

    Returns:
        torch.Tensor of shape ``(B, hidden_dim)`` on CPU.
    """
    if torch.is_tensor(hidden_states):
        x: torch.Tensor = hidden_states.detach().cpu().float()
    else:
        x = torch.as_tensor(np.asarray(hidden_states, dtype=np.float32))
    # horizon mean (SAFE horizon_idx_rel='mean')
    x = x.mean(dim=-2)  # (B, num_denoise_steps, hidden_dim)
    # diff-steps mean (SAFE diff_idx_rel='mean')
    x = x.mean(dim=-2)  # (B, hidden_dim)
    return x


def _time_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum over step axis (SAFE indep cumsum=True)."""
    return torch.cumsum(x, dim=1)


class SafeFailureDetector(nn.Module):
    """Online failure detector scoring one (feature) decision-step at a time.

    Attributes:
        name: ``"indep"`` (MLP) or ``"lstm"``.
        config: dict holding the model hyper-parameters used at training time.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.name = config.get("name", "indep")
        self.config = config

        input_dim = int(config.get("input_dim", 1024))
        hidden_dim = int(config.get("hidden_dim", 256))
        n_layers = int(config.get("n_layers", 2))
        self.n_history_steps = int(config.get("n_history_steps", -1))
        self.cumsum = bool(config.get("cumsum", (self.name == "indep")))
        final_act = config.get("final_act_layer", "sigmoid")
        dropout = float(config.get("dropout", 0.0))

        if self.name == "indep":
            # Replicates SAFE failure_prob/model/indep.py::IndepModel.projector
            # plus a final activation, feeding each timestep independently.
            projector = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(n_layers - 2):
                projector += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            projector.append(nn.Linear(hidden_dim, 1))
            if final_act == "sigmoid":
                projector.append(nn.Sigmoid())
            elif final_act in ("relu", "none"):
                pass
            else:
                raise ValueError(f"Unknown final_act_layer: {final_act}")
            self.projector = nn.Sequential(*projector)
        elif self.name == "lstm":
            # Replicates SAFE failure_prob/model/lstm.py::LstmModel
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.fc = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unsupported detector name: {self.name}")

        self._step_counter: torch.Tensor = None  # (B,) step index
        self._hidden_cache = None  # LSTM per-env hidden state
        self._running_cumsum: torch.Tensor = None  # indep cumsum running total
        self.reset_history(batch=None)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset_history(self, batch: int | tuple[int, int] | None) -> None:
        """Reset per-episode state (LSTM hidden cache, step counter).

        Args:
            batch: None to reset to "no batch" (must reset before scoring), an
                ``int`` for the env count, or a ``(num_layers, num_envs)`` tuple.
        """
        if batch is None:
            self._step_counter = None
            self._hidden_cache = None
            self._running_cumsum = None
            return
        if isinstance(batch, int):
            num_envs = batch
        else:
            num_envs = batch[1]
        self._step_counter = torch.zeros(num_envs, dtype=torch.long)
        if self.name == "indep":
            self._running_cumsum = torch.zeros(num_envs)
        if self.name == "lstm":
            h0 = torch.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size)
            c0 = torch.zeros_like(h0)
            self._hidden_cache = (h0, c0)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def forward_step(self, hidden_states) -> torch.Tensor:
        """Score one step of ``pre_velocity`` hidden states.

        Args:
            hidden_states: see :func:`preprocess_suffix_out`; shape
                ``(B, num_denoise_steps, n_pred_horizon, hidden_dim)``.

        Returns:
            torch.Tensor of shape ``(B,)`` on the model device with the per-env
            failure score for this step. Over successive steps the scores for a
            given env accumulate (indep cumsum or LSTM hidden state).
        """
        x = preprocess_suffix_out(hidden_states).to(self._get_device())
        B = x.shape[0]
        if self._step_counter is None and self.name == "lstm":
            self.reset_history(B)
        elif self._step_counter is None:
            self._step_counter = torch.zeros(B, dtype=torch.long, device=x.device)
            if self.name == "indep" and self._running_cumsum is None:
                self._running_cumsum = torch.zeros(B, device=x.device)
        self._step_counter = self._step_counter.to(x.device)
        self._step_counter += 1

        if self.name == "indep":
            p = self.projector(x).reshape(B)  # (B,) sigmoid output per timestep
            if self.cumsum:
                self._running_cumsum = self._running_cumsum.to(p.device)
                self._running_cumsum += p
                s = self._running_cumsum.clone()
            else:
                s = p
        else:  # lstm, n_history_steps=-1 => maintain hidden state across steps
            h, c = self._hidden_cache
            h = h.to(x.device)
            c = c.to(x.device)
            x_seq = x.unsqueeze(1)  # (B, 1, D)
            out, (h, c) = self.lstm(x_seq, (h, c))
            self._hidden_cache = (h.detach(), c.detach())
            s = self.dropout(out[:, -1, :])  # (B, hidden)
            s = torch.sigmoid(self.fc(s))  # (B, 1)
            s = s.reshape(B)
            if self.cumsum:
                self._running_cumsum = self._running_cumsum.to(s.device)
                self._running_cumsum += s
                s = self._running_cumsum.clone()

        return s

    def _get_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_ckpt(cls, ckpt_path: str, config: dict, map_location="cpu"):
        """Build the detector and load a SAFE ``state_dict`` checkpoint.

        Args:
            ckpt_path: path to a ``model_final.ckpt`` (state_dict).
            config: model hyper-params including ``input_dim``.
        """
        detector = cls(config)
        state = torch.load(ckpt_path, map_location=map_location)
        detector.load_state_dict(state)
        detector.eval()
        return detector
