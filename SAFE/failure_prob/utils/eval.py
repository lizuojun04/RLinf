import numpy as np
import torch
from torch.utils.data import DataLoader

from failure_prob.model.base import BaseModel
from failure_prob.utils.torch import move_to_device


def model_forward_dataloader(model: BaseModel, loader: DataLoader):
    """Batch the model forward over a dataloader.

    Returns (scores (B, T), valid_masks (B, T), labels (B,)).
    """
    device = model.get_device()
    scores, valid_masks, labels = [], [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        scores.append(model(batch))
        valid_masks.append(batch["valid_masks"])
        labels.append(batch["success_labels"])
    scores = torch.cat(scores, dim=0).squeeze(-1)
    valid_masks = torch.cat(valid_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    return scores, valid_masks, labels


def score_rollouts(
    model: BaseModel,
    rollouts_by_split_name: dict[str, list],
    batch_size: int,
) -> dict[str, list[np.ndarray]]:
    """Forward a model over each split's rollouts and return per-rollout score arrays.

    Each returned list element is the score trajectory ``(T_i,)`` for one rollout.
    """
    from torch.utils.data import DataLoader

    from failure_prob.data.utils import RolloutDataset

    scores_by_split_name: dict[str, list[np.ndarray]] = {}
    model.eval()
    for split, rollouts in rollouts_by_split_name.items():
        dataset = RolloutDataset(rollouts, device="cpu")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, loader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()
        scores_by_split_name[split] = [
            scores[i, : int(seq_lengths[i])] for i in range(len(seq_lengths))
        ]
    return scores_by_split_name
