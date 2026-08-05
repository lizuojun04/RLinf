from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Rollout:
    """A single rollout with its per-timestep (T, d) feature sequence."""

    hidden_states: torch.Tensor
    task_suite_name: str | None = None
    task_id: int | None = None
    task_description: str | None = None
    episode_idx: int | None = None
    episode_success: int | None = None
    mp4_path: str | None = None
    logs: pd.DataFrame | None = None
    task_min_step: int | None = None
    exec_horizon: int | None = None
    action_vectors: torch.Tensor | None = None

    def __post_init__(self):
        self.episode_success = int(self.episode_success) if self.episode_success is not None else None

    def to(self, device):
        self.hidden_states = self.hidden_states.to(device)
        if self.action_vectors is not None:
            self.action_vectors = self.action_vectors.to(device)
        return self


class RolloutDataset(Dataset):
    """PyTorch Dataset for the rollout data with inverse-frequency class weights."""

    def __init__(self, rollouts: list[Rollout], lambda_fail=1.0, lambda_success=1.0, device=None):
        self.rollouts = rollouts
        self.length = len(rollouts)
        self.device = device if device is not None else "cpu"

        # Weigh the loss by the inverse frequency of success / failure.
        n_succ = sum(1 for r in rollouts if r.episode_success == 1)
        n_fail = len(rollouts) - n_succ
        freq_0 = (n_fail + 1) / len(rollouts)
        freq_1 = (n_succ + 1) / len(rollouts)
        self.weights = [1.0 / freq_0, 1.0 / freq_1]
        self.weights[0] *= lambda_fail
        self.weights[1] *= lambda_success

        padded, masks, labels, action_vectors = pad_rollout_batch(self.rollouts, self.device)
        self.features, self.valid_masks, self.labels, self.action_vectors = (
            padded, masks, labels, action_vectors,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {
            "features": self.features[idx],
            "valid_masks": self.valid_masks[idx],
            "success_labels": self.labels[idx],
        }
        if self.action_vectors is not None:
            data["action_vectors"] = self.action_vectors[idx]
        return data

    def get_rollouts(self):
        return self.rollouts

    def get_features(self):
        return self.features

    def get_valid_masks(self):
        return self.valid_masks

    def get_labels(self):
        return self.labels

    def get_class_weights(self):
        return self.weights


def pad_rollout_batch(rollouts: list[Rollout], device=None):
    """Pad hidden states to the max length in the batch.

    Returns:
        (features, valid_masks, labels, action_vectors) where valid_masks
        marks *padding* (1 = pad / invalid, 0 = valid) to match the SAFE
        ``RolloutDataset`` convention.
    """
    batch_features = [r.hidden_states for r in rollouts]
    max_length = max(seq.shape[0] for seq in batch_features)
    hidden_dim = batch_features[0].shape[-1]
    batch_size = len(batch_features)

    dtype = batch_features[0].dtype
    if device is None:
        device = batch_features[0].device

    padded_features = torch.zeros(batch_size, max_length, hidden_dim, dtype=dtype, device=device)
    padding_masks = torch.ones(batch_size, max_length, dtype=torch.float32, device=device)

    for i, seq in enumerate(batch_features):
        seq_length = seq.shape[0]
        padded_features[i, :seq_length] = seq.to(device)
        padding_masks[i, seq_length:] = 0.0

    labels = torch.tensor([r.episode_success for r in rollouts], dtype=torch.float32, device=device)

    action_vectors = None
    if rollouts[0].action_vectors is not None:
        action_dim = rollouts[0].action_vectors.shape[-1]
        action_vectors = torch.zeros(batch_size, max_length, action_dim, dtype=dtype, device=device)
        for i, r in enumerate(rollouts):
            seq_length = r.action_vectors.shape[0]
            action_vectors[i, :seq_length] = r.action_vectors.to(device)

    return padded_features, padding_masks, labels, action_vectors


def normalize_rollouts_hidden_states(rollouts: list[Rollout]):
    """Normalize hidden states to zero mean / unit variance, in place."""
    all_hidden_states = torch.cat([r.hidden_states for r in rollouts], dim=0)
    mean = all_hidden_states.mean(dim=0)
    std = all_hidden_states.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    for r in rollouts:
        r.hidden_states = (r.hidden_states - mean) / std
    return rollouts


def split_rollouts_by_seen_unseen(
    all_rollouts: list[Rollout],
    seen_task_ids: list[int],
    unseen_task_ids: list[int],
    seen_train_ratio: float = 0.6,
):
    """Split rollouts into train / val_seen / val_unseen by task.

    Each seen task's rollouts are split into train (seen_train_ratio) and
    val_seen (the remainder) via a seeded torch.randperm.
    """
    print(f"Seen tasks: {seen_task_ids}, Unseen tasks: {unseen_task_ids}")

    seen_rollouts = [r for r in all_rollouts if r.task_id in seen_task_ids]
    unseen_rollouts = [r for r in all_rollouts if r.task_id in unseen_task_ids]

    train_rollouts: list[Rollout] = []
    val_seen_rollouts: list[Rollout] = []
    for task_id in seen_task_ids:
        task_rollouts = [r for r in seen_rollouts if r.task_id == task_id]
        permuted_indices = torch.randperm(len(task_rollouts))
        n_train = int(seen_train_ratio * len(task_rollouts))
        train_rollouts += [task_rollouts[i] for i in permuted_indices[:n_train]]
        val_seen_rollouts += [task_rollouts[i] for i in permuted_indices[n_train:]]

    val_unseen_rollouts = unseen_rollouts

    rollouts_by_split_name = {
        "train": train_rollouts,
        "val_seen": val_seen_rollouts,
    }
    if val_unseen_rollouts:
        rollouts_by_split_name["val_unseen"] = val_unseen_rollouts

    for split, rollouts in rollouts_by_split_name.items():
        n_success = sum(1 for r in rollouts if r.episode_success == 1)
        n_fail = len(rollouts) - n_success
        print(f"{split}: {len(rollouts)} rollouts, {n_success} success, {n_fail} fail")

    return rollouts_by_split_name


def set_task_min_step(rollouts: list[Rollout]):
    """Set each rollout's ``task_min_step`` to the min rollout length of its task."""
    by_task: dict[int, list[Rollout]] = {}
    for r in rollouts:
        by_task.setdefault(r.task_id, []).append(r)
    for r in rollouts:
        task_rollouts = by_task[r.task_id]
        r.task_min_step = min(len(x.hidden_states) for x in task_rollouts)
    return rollouts


def parse_and_index_tensor_last(A: np.ndarray, command: str):
    """Apply a slice / uniform-index command to the last two dims and flatten."""
    if command == "concat":
        new_last_dim = A.shape[-2] * A.shape[-1]
        return A.reshape(*A.shape[:-2], new_last_dim)

    prefix = "concat-"
    sub_cmd = command[len(prefix):]

    if ":" in sub_cmd:
        parts = sub_cmd.split(":")
        if len(parts) == 2:
            start_str, stop_str = parts
            start = int(start_str) if start_str != "" else None
            stop = int(stop_str) if stop_str != "" else None
            indexed = A[..., slice(start, stop), :]
        elif len(parts) == 3:
            start_str, stop_str, step_str = parts
            start = int(start_str) if start_str != "" else None
            stop = int(stop_str) if stop_str != "" else None
            step = int(step_str) if step_str != "" else None
            indexed = A[..., slice(start, stop, step), :]
        else:
            raise ValueError("Invalid slice format in command.")
        new_last_dim = indexed.shape[-2] * indexed.shape[-1]
        return indexed.reshape(*indexed.shape[:-2], new_last_dim)

    try:
        k = int(sub_cmd)
    except ValueError:
        raise ValueError("Invalid command format; expected a colon-based slice or an integer.")

    if k < 2:
        raise ValueError("Uniform indexing requires at least 2 features.")
    c = A.shape[-2]
    indices = np.round(np.linspace(0, c - 1, num=k)).astype(int)
    indexed = A[..., indices, :]
    new_last_dim = indexed.shape[-2] * indexed.shape[-1]
    return indexed.reshape(*indexed.shape[:-2], new_last_dim)


def process_tensor_idx_rel(A: np.ndarray, command: Any):
    """Aggregate / index a feature tensor along its second-to-last axis.

    Supported commands (SAFE compatible):
      - float (0..1): select a single index along axis -2.
      - "mean": mean over axis -2.
      - "concat*": flatten the last two dims (optionally indexed).
    """
    assert len(A.shape) >= 2, "Tensor A must have at least two dimensions."
    if isinstance(command, float):
        assert 0 <= command <= 1, f"Invalid token index ratio: {command}"
        token_idx = round((A.shape[-2] - 1) * command)
        return A[..., token_idx, :]
    elif command == "mean":
        return A.mean(axis=-2)
    elif isinstance(command, str) and "concat" in command:
        return parse_and_index_tensor_last(A, command)
    else:
        raise ValueError(f"Unknown token index: {command}")
