"""Loader for RLinf-native ``rollout.safe_dump`` rollout records.

Each RLinf record is one self-contained pkl per rollout, laid out under
``<root>/<task_id>/<run_id>/{success,fail}/*.pkl``. The ``hidden_states`` field
holds the *raw* openpi ``suffix_out`` tensor of shape ``(T, num_denoise_steps,
n_pred_horizon, hidden_dim)`` — one per policy-inference step. SAFE training
expects ``Rollout.hidden_states`` to be ``(T, d)``, so we aggregate the raw
features per step with ``horizon_idx_rel`` / ``diff_idx_rel`` (default mean /
mean), mirroring ``SAFE_origin/failure_prob/data/pizero.py``.
"""

from __future__ import annotations

import glob
import os
import pickle
import re

import numpy as np
import torch

from .utils import Rollout, process_tensor_idx_rel, set_task_min_step


def natsorted(paths: list[str]) -> list[str]:
    def _key(p: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]

    return sorted(paths, key=_key)


def find_rollout_paths(data_path: str) -> list[str]:
    """Recursively find every ``success``/``fail`` rollout pkl under ``data_path``."""
    paths = glob.glob(os.path.join(data_path, "**", "success", "*.pkl"), recursive=True) + \
        glob.glob(os.path.join(data_path, "**", "fail", "*.pkl"), recursive=True)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError(
            f"No {os.path.join('<task_id>', '<run_id>', 'success|fail', '*.pkl')} "
            f"found under {data_path}"
        )
    return natsorted(paths)


def _aggregate_step(raw: np.ndarray, horizon_idx_rel, diff_idx_rel) -> np.ndarray:
    """Reduce a per-step raw feature ``(k, H, d)`` to ``(d,)``.

    Mirrors SAFE's ``pizero.py``: horizon aggregation then diff-steps
    aggregation, each via ``process_tensor_idx_rel``.
    """
    h = process_tensor_idx_rel(raw, horizon_idx_rel)  # (k, d)
    h = process_tensor_idx_rel(h, diff_idx_rel)       # (d,)
    return np.asarray(h, dtype=np.float32)


def load_rollouts(
    data_path: str,
    horizon_idx_rel: str = "mean",
    diff_idx_rel: str = "mean",
    feat_name: str = "hidden_states",
    task_ids: list[int] | None = None,
    max_per_class_per_task: int | None = None,
    seed: int | None = None,
    load_to_cuda: bool = True,
) -> list[Rollout]:
    """Load RLinf-native rollouts into SAFE ``Rollout`` objects.

    Args:
        data_path: root folder containing ``**/{success,fail}/*.pkl``.
        horizon_idx_rel / diff_idx_rel: aggregation over the horizon / denoise
            dimension (``"mean"`` or a ``concat-*`` command).
        feat_name: the RLinf record key holding the raw features
            (default ``hidden_states``).
        task_ids: if set, only load rollouts whose ``task_id`` is in this list
            (filters out invalid / un-wanted tasks, e.g. those with a single
            class). Default None = all tasks.
        max_per_class_per_task: if set, randomly subsample each (task, class)
            bucket to at most this many rollouts (for balanced datasets).
        seed: RNG seed for the per-class subsampling.
        load_to_cuda: move ``hidden_states`` tensors to cuda after normalizing.

    Returns:
        list of :class:`Rollout` with ``hidden_states`` shaped ``(T, d)``.
    """
    allowed_task_ids = set(task_ids) if task_ids is not None else None

    def _keep_tid(tid: int) -> bool:
        return allowed_task_ids is None or tid in allowed_task_ids

    rng = np.random.default_rng(seed) if (seed is not None or max_per_class_per_task is not None) else None
    paths = find_rollout_paths(data_path)

    rollouts: list[Rollout] = []
    per_key: dict[tuple[int, int], list[str]] = {}
    for p in paths:
        with open(p, "rb") as f:
            rec = pickle.load(f)
        tid = int(rec.get("task_id", 0))
        if not _keep_tid(tid):
            continue
        succ = int(rec.get("episode_success", 0))
        per_key.setdefault((tid, succ), []).append(p)

    if max_per_class_per_task is not None and rng is not None:
        for key, bucket in per_key.items():
            if len(bucket) > max_per_class_per_task:
                idx = rng.choice(len(bucket), size=max_per_class_per_task, replace=False)
                per_key[key] = [bucket[i] for i in sorted(idx)]

    for p in paths:
        with open(p, "rb") as f:
            rec = pickle.load(f)
        tid = int(rec.get("task_id", 0))
        succ = int(rec.get("episode_success", 0))
        if not _keep_tid(tid):
            continue
        if p not in per_key.get((tid, succ), []):
            continue

        raw = rec.get(feat_name)
        if raw is None:
            raise KeyError(f"rollout {p} has no '{feat_name}' key.")
        raw = np.asarray(raw, dtype=np.float32)  # (T, k, H, d)

        hidden = np.stack(
            [_aggregate_step(raw[t], horizon_idx_rel, diff_idx_rel) for t in range(raw.shape[0])],
            axis=0,
        ).astype(np.float32)
        hidden_t = torch.from_numpy(hidden[:, :])

        actions_arr = rec.get("actions")
        action_vectors = None
        if actions_arr is not None:
            a = np.asarray(actions_arr, dtype=np.float32)  # (T, H, action_dim)
            if a.ndim == 3:
                a = a.reshape(a.shape[0], -1)
            action_vectors = torch.from_numpy(a)

        rollouts.append(
            Rollout(
                hidden_states=hidden_t,
                task_suite_name=str(rec.get("task_suite_name", "")),
                task_id=tid,
                task_description=str(rec.get("task_description", "")),
                episode_idx=int(rec.get("episode_idx", 0)),
                episode_success=succ,
                exec_horizon=int(rec.get("replan_steps", 1)) if rec.get("replan_steps") else 1,
                action_vectors=action_vectors,
            )
        )

    rollouts = set_task_min_step(rollouts)
    print(f"Loaded {len(rollouts)} rollouts from {data_path}")

    if load_to_cuda and torch.cuda.is_available():
        rollouts = [r.to("cuda") for r in rollouts]

    return rollouts
