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

"""Functional-conformal-prediction (CP) band for real-time SAFE failure scoring.

A *band* is a time-varying threshold trajectory ``cp_band[0..T-1]`` calibrated on
the *successful* rollouts' score trajectories. Under the conformal guarantee, at
significance level ``alpha`` a success rollout's score at timestep ``t`` stays
below ``cp_band[t]`` with probability ~``1-alpha``. At inference we compare the
live per-step score against the band: ``score[t] >= cp_band[t]`` raises a failure
flag.

The implementation replicates ``SAFE/src/.../conformal/functional_predictor.py``
(``FunctionalPredictor`` with ``ModulationType.Tfunc`` + ``RegressionType.Mean``),
which is what the offline SAFE evaluation uses. Keeping it in RLinf lets the live
detector stay in agreement with the offline ``SAFE/scripts/eval_ckpt.py``.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Core band computation (functional conformal, upper one-sided bound)
# --------------------------------------------------------------------------- #
def _regress_mean(training_data: np.ndarray) -> np.ndarray:
    """Point-wise mean trajectory over a ``(N, L)`` array -> ``(1, L)``."""
    return np.mean(training_data, axis=0, keepdims=True)


def _modulation_tfunc(
    training_data: np.ndarray,
    prediction_trajectory: np.ndarray,
    alpha: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Modulation trajectory (SAFE ModulationType.Tfunc)."""
    length = training_data.shape[-1]
    train_size = training_data.shape[0]
    centered = np.abs(training_data - prediction_trajectory)  # (N, L)
    if int(np.ceil((train_size + 1) * (1 - alpha))) > train_size:
        modulation = np.max(centered, axis=0, keepdims=True) + eps
    else:
        gamma = np.sort(np.max(centered, axis=1))[
            int(np.ceil((train_size + 1) * (1 - alpha))) - 1
        ]
        modulation = (
            np.max(centered[np.max(centered, axis=1) <= gamma], axis=0, keepdims=True)
            + eps
        )
    assert modulation.shape == (1, length)
    return modulation


def get_one_sided_prediction_band(
    training_data: np.ndarray,
    calibration_data: np.ndarray,
    alpha: float,
    lower_bound: bool = False,
) -> np.ndarray:
    """Upper (or lower) one-sided functional-conformal prediction band.

    Args:
        training_data: ``(N1, L)`` regression trajectories (all aligned-length).
        calibration_data: ``(N2, L)`` calibration trajectories.
        alpha: significance level in ``(0, 1)``.
        lower_bound: False -> upper band; True -> lower band.

    Returns:
        np.ndarray of shape ``(1, L)`` bounding trajectory.
    """
    length = training_data.shape[-1]
    assert length == calibration_data.shape[-1]
    assert 0.0 < alpha < 1.0

    prediction_trajectory = _regress_mean(training_data)
    modulation_trajectory = _modulation_tfunc(
        training_data, prediction_trajectory, alpha
    )

    if lower_bound:
        cal_scores = [
            np.max((prediction_trajectory - c) / modulation_trajectory)
            for c in calibration_data
        ]
    else:
        cal_scores = [
            np.max((c - prediction_trajectory) / modulation_trajectory)
            for c in calibration_data
        ]
    band_width = np.quantile(cal_scores, 1 - alpha)

    if lower_bound:
        bounding = prediction_trajectory - band_width * modulation_trajectory
    else:
        bounding = prediction_trajectory + band_width * modulation_trajectory
    return bounding


# --------------------------------------------------------------------------- #
# Failure decision from a band
# --------------------------------------------------------------------------- #
def flag_from_band(
    scores: np.ndarray,
    cp_band: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decide failure per trajectory from a precomputed band.

    Mirrors SAFE's ``eval_functional_conformal`` detection: for every step
    ``score[t] >= cp_band[t]`` (edge-extended for steps past the band length) a
    trajectory is flagged as a failure as soon as *any* step exceeds the band.

    Args:
        scores: ``(T,)`` or ``(T,)*k`` score trajectory(s).
        cp_band: ``(T,)`` upper band.

    Returns:
        ``(has_detection, first_detection, det_times_norm)``:
            has_detection  ``(B,)`` bool
            first_detection ``(B,)`` int (first step index; ``T`` if none)
            det_times_norm  ``(B,)`` float first-detection / T
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    T = cp_band.shape[0]
    band_full = np.pad(cp_band, (0, max(scores.shape[-1] - T, 0)), mode="edge")
    band_full = band_full[: scores.shape[-1]]
    mask = scores >= band_full[None, :]  # (B, L)
    has_detection = mask.any(axis=1)
    first = np.argmax(mask, axis=1)
    first_detection = np.where(has_detection, first, scores.shape[-1])
    det_times_norm = first_detection / scores.shape[-1]
    return has_detection, first_detection, det_times_norm
