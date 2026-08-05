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

"""Load SAFE calibration rollouts and compute the functional-CP band at run time.

The SAFE (NeurIPS 2025) conformal-prediction band is *not* a static artifact: it is
recomputed from a calibration set of the *successful* rollouts' score trajectories
each time we evaluate. The score trajectory of a rollout is obtained by running the
trained failure detector over that rollout's per-step hidden-state features (SAFE
``pre_velocity``). This module reproduces, in RLinf and without importing the
standalone SAFE repository, the pipeline behind
``SAFE/failure_prob/utils/metrics.py::eval_functional_conformal``:

    1. Load ``<data_path>/**/{success,fail}/*.pkl`` — the RLinf-native dump layout
       produced by ``rollout.safe_dump``, one self-contained pkl per rollout holding
       the per-decision-step ``hidden_states`` (``pre_velocity``), ``actions``,
       images, states and metadata.
    2. Run the detector over every rollout to obtain a scalar score trajectory
       (indep uses cumsum over steps; lstm maintains a hidden state).
    3. Split time-series into seen / unseen tasks and seen tasks into train / val_seen
       exactly like ``split_rollouts_by_seen_unseen``.
    4. Calibrate on the ``val_seen`` *successful* trajectories and produce the
       upper one-sided functional-CP band at significance ``alpha``
       (``rlinf.../functional_cp.py::get_one_sided_prediction_band``).

The band is kept in memory only; nothing is persisted to disk.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Optional

import numpy as np
import torch

from rlinf.models.embodiment.failure_detector.functional_cp import (
    get_one_sided_prediction_band,
)
from rlinf.models.embodiment.failure_detector.safe_detector import (
    SafeFailureDetector,
)


# --------------------------------------------------------------------------- #
# On-disk rollout loading (RLinf-native dump format)
# --------------------------------------------------------------------------- #
class _CalibRollout:
    """Lightweight rollout metadata + preprocessed per-step feature matrix."""

    __slots__ = (
        "features",
        "task_id",
        "task_description",
        "episode_success",
        "task_min_step",
    )

    def __init__(self, features, task_id, task_description, episode_success):
        # features: (T, num_denoise_steps, n_pred_horizon, hidden_dim) float32
        self.features = features
        self.task_id = task_id
        self.task_description = task_description
        self.episode_success = int(episode_success)
        self.task_min_step: Optional[int] = None


def load_safe_rollouts(data_path: str) -> list[_CalibRollout]:
    """Load every rollout from an RLinf-native ``data_path``.

    Args:
        data_path: root folder of the dump. Rollouts live under any
            ``<task_id>/<run_id>/{success,fail}/*.pkl`` layout written by
            ``rollout.safe_dump`` (recursively discovered).

    Returns:
        A list of :class:`_CalibRollout` in sorted (stable) episode order, each
        with ``features`` of shape ``(T, num_denoise_steps, n_pred_horizon,
        hidden_dim)`` (one raw feature per policy-inference step), plus env
        metadata.
    """
    rollout_paths = natsorted(
        glob.glob(os.path.join(data_path, "**", "success", "*.pkl"), recursive=True)
        + glob.glob(os.path.join(data_path, "**", "fail", "*.pkl"), recursive=True)
    )
    if not rollout_paths:
        raise FileNotFoundError(
            f"No {os.path.join('<task_id>', '<run_id>', 'success|fail', '*.pkl')} "
            f"found under {data_path}"
        )

    rollouts: list[_CalibRollout] = []
    for rpath in rollout_paths:
        with open(rpath, "rb") as f:
            rec = pickle.load(f)
        hidden_states = rec.get("hidden_states")
        if hidden_states is None:
            raise KeyError(
                f"rollout record {rpath} has no 'hidden_states' key. "
                "Regenerate the dump with the current safe_dump writer."
            )
        features = np.asarray(hidden_states, dtype=np.float32)
        rollouts.append(
            _CalibRollout(
                features=features,
                task_id=int(rec.get("task_id", 0)),
                task_description=str(rec.get("task_description", "unknown")),
                episode_success=int(rec.get("episode_success", 0)),
            )
        )

    _set_task_min_step(rollouts)
    return rollouts


def natsorted(paths: list[str]) -> list[str]:
    """Natural sort so ``episode_000002`` sorts before ``episode_000010``."""
    import re

    def _key(p: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]

    return sorted(paths, key=_key)


def _set_task_min_step(rollouts: list[_CalibRollout]) -> None:
    """Set ``task_min_step`` = minimum rollout length per task (SAFE)."""
    by_task: dict[int, list[_CalibRollout]] = {}
    for r in rollouts:
        by_task.setdefault(r.task_id, []).append(r)
    for r in rollouts:
        task_rollouts = by_task[r.task_id]
        r.task_min_step = min(len(x.features) for x in task_rollouts)


# --------------------------------------------------------------------------- #
# Split + scoring + band computation
# --------------------------------------------------------------------------- #
def _split_by_seen_unseen(
    rollouts: list[_CalibRollout],
    unseen_task_ratio: float,
    seen_train_ratio: float,
    split_seed: Optional[int],
) -> dict[str, list[_CalibRollout]]:
    """Split rollouts into train/val_seen/val_unseen by task.

    Replicates SAFE's exact procedure for a merged (single ``data_path``) dataset,
    i.e. ``split_rollouts`` (``openvla.py``) + ``split_rollouts_by_seen_unseen``
    (``utils.py``) when ``data_path_unseen is None``:

        1. ``np.random.shuffle(task_ids)`` over the *set* of task_ids, keep the
           first ``n_seen`` as seen tasks.
        2. Per seen task (in the shuffled order), ``torch.randperm`` to split its
           rollouts into train / val_seen by ``seen_train_ratio``.

    ``split_seed`` seeds the global torch AND numpy RNG exactly like SAFE's
    ``seed_everything(split_seed)`` immediately before ``split_rollouts``; if None,
    the current (unseeded) RNG state is used.
    """
    if split_seed is not None:
        np.random.seed(split_seed)
        torch.manual_seed(split_seed)

    # step 1: shuffle task ids (SAFE openvla.split_rollouts)
    task_ids = list({r.task_id for r in rollouts})
    n_unseen = round(unseen_task_ratio * len(task_ids))
    n_seen = len(task_ids) - n_unseen
    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]

    seen_rollouts = [r for r in rollouts if r.task_id in seen_task_ids]
    unseen_rollouts = [r for r in rollouts if r.task_id in unseen_task_ids]

    # step 2: per seen task, torch.randperm train/val_seen split
    train_rollouts: list[_CalibRollout] = []
    val_seen_rollouts: list[_CalibRollout] = []
    for tid in seen_task_ids:
        task_rollouts = [r for r in seen_rollouts if r.task_id == tid]
        perm_indices = torch.randperm(len(task_rollouts)).tolist()
        n_train = int(seen_train_ratio * len(task_rollouts))
        train_rollouts += [task_rollouts[i] for i in perm_indices[:n_train]]
        val_seen_rollouts += [task_rollouts[i] for i in perm_indices[n_train:]]

    splits = {"train": train_rollouts, "val_seen": val_seen_rollouts}
    if unseen_rollouts:
        splits["val_unseen"] = unseen_rollouts
    return splits


def compute_band_from_calibration(
    score_trajectories: list[np.ndarray],
    alpha: float,
    shuffle_seed: Optional[int],
    pad_length: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Calibrate an upper one-sided functional-CP band from success scores.

    Args:
        score_trajectories: 1-D score trajectories of the calibration (successful)
            rollouts. All are right-edge-padded to ``pad_length`` before use.
        alpha: significance level in ``(0, 1)``.
        shuffle_seed: If given, seed the 30/70 regression/calibration split so the
            resulting band is reproducible (SAFE relies on ``np.random.shuffle``).
        pad_length: target length to edge-pad every trajectory to. MUST match the
            global maximum rollout length (over calibration AND test rollouts), as
            SAFE's ``eval_functional_conformal`` pads to ``max(len(s) for s in
            cal_scores_all + test_scores_all)``. This ensures the returned band
            spans the full(est) decision horizon.

    Returns:
        ``(cp_band, T)`` where ``cp_band`` has shape ``(T,)`` and ``T = pad_length``.
    """
    if not score_trajectories:
        raise ValueError("Need at least one success calibration trajectory.")
    T = (
        int(pad_length)
        if pad_length is not None
        else max(len(t) for t in score_trajectories)
    )
    arr = np.array(
        [np.pad(t, (0, T - len(t)), mode="edge") for t in score_trajectories],
        dtype=np.float64,
    )
    # Reproduce SAFE's exact split: ``np.random.shuffle`` then 30% regression /
    # 70% calibration. We seed the (legacy) global RNG for reproducibility; SAFE
    # relies on the same ``np.random.shuffle`` call (unseeded there).
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
    np.random.shuffle(arr)
    n1 = max(int(len(arr) * 0.3), 1)
    train_part = arr[:n1]
    cal_part = arr[n1:]
    band = get_one_sided_prediction_band(train_part, cal_part, alpha, lower_bound=False)
    return band[0], T


def compute_safe_band(
    data_path: str,
    detector: SafeFailureDetector,
    alpha: float = 0.2,
    unseen_task_ratio: float = 0.3,
    seen_train_ratio: float = 0.6,
    split_seed: Optional[int] = 0,
    shuffle_seed: Optional[int] = 0,
) -> tuple[np.ndarray, dict]:
    """Load SAFE calibration data, score it and produce the runtime CP band.

    Args:
        data_path: dump root with ``**/{success,fail}/*.pkl`` (RLinf-native).
        detector: the loaded SAFE detector (indep or lstm).
        alpha: significance level.
        unseen_task_ratio: fraction of tasks held out as unseen.
        seen_train_ratio: fraction of each seen task's rollouts used for train
            (the remainder becomes val_seen / calibration).
        split_seed: seed for the seen/unseen + train/val_seen split.
        shuffle_seed: seed for the functional-CP 30/70 split.

    Returns:
        ``(cp_band, info)`` where ``cp_band`` is ``(T,)`` and ``info`` carries
        diagnostic counters (n_success_cal, T, task counts, split sizes).
    """
    rollouts = load_safe_rollouts(data_path)
    # Global max trajectory length over ALL rollouts (cal + test, all tasks),
    # matching SAFE's ``eval_functional_conformal`` which pads to
    # ``max(len(s) for s in cal_scores_all + test_scores_all)``. This ensures
    # the band covers the full decision horizon the live episode may reach.
    global_T = max(len(r.features) for r in rollouts)
    splits = _split_by_seen_unseen(
        rollouts,
        unseen_task_ratio=unseen_task_ratio,
        seen_train_ratio=seen_train_ratio,
        split_seed=split_seed,
    )

    val_seen = splits.get("val_seen", [])
    if not val_seen:
        raise RuntimeError(
            f"SAFE calibration: no val_seen rollouts after split of {data_path}. "
            "Increase the number of rollouts or lower unseen_task_ratio/seen_train_ratio."
        )
    success_rollouts = [r for r in val_seen if r.episode_success == 1]
    if not success_rollouts:
        raise RuntimeError(
            f"SAFE calibration: no successful rollouts in val_seen "
            f"({len(val_seen)} val_seen rollouts). Cannot calibrate an upper band."
        )

    score_trajs = _score_rollouts(detector, success_rollouts)
    cp_band, T = compute_band_from_calibration(
        score_trajs, alpha, shuffle_seed, pad_length=global_T
    )

    info = {
        "data_path": data_path,
        "alpha": alpha,
        "T": T,
        "n_rollouts": len(rollouts),
        "n_val_seen": len(val_seen),
        "n_success_cal": len(success_rollouts),
        "seen_tasks": sorted({r.task_id for r in splits["train"] + val_seen}),
        "unseen_tasks": sorted({r.task_id for r in splits.get("val_unseen", [])}),
        "split": {k: len(v) for k, v in splits.items()},
    }
    return cp_band, info


def _score_rollouts(
    detector: SafeFailureDetector, rollouts: list[_CalibRollout]
) -> list[np.ndarray]:
    """Run the detector over whole rollouts to yield one score per step.

    Each rollout is scored independently as a batch of one, feeding its real
    features step-by-step (any cumsum / LSTM state is per-env). This reproduces
    SAFE's offline ``model(batch)`` output ``scores[i, :len(r)]`` exactly: for
    ``indep`` the running cumsum accumulates the sigmoid probabilities over the
    rollout; padded positions in SAFE never affect the pre-padding trajectory, so
    scoring a single rollout without padding is equivalent.
    """
    device = detector._get_device()
    trajs: list[list[float]] = []
    for rollout in rollouts:
        detector.reset_history(1)
        seq = []
        feats = torch.as_tensor(rollout.features, dtype=torch.float32, device=device)
        for step in range(len(rollout.features)):
            out = detector.forward_step(feats[step : step + 1])
            seq.append(float(out.detach().cpu().float().numpy()[0]))
        trajs.append(seq)
    return [np.asarray(t, dtype=np.float64) for t in trajs]
