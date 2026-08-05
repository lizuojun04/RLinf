import torch


def cumsum_stopgrad(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumulative sum along ``dim`` with stop-gradient.

    ``y[..., i, ...]`` only backprops to ``x[..., i, ...]``.
    """
    if dim < 0:
        dim = x.dim() + dim
    y = x.new_empty(x.shape)
    running = torch.zeros_like(x.select(dim, 0))
    idx = [slice(None)] * x.dim()
    for i in range(x.size(dim)):
        idx[dim] = i
        current = x[tuple(idx)]
        running = running.detach().add(current)
        y[tuple(idx)] = running
    return y


def get_time_weight(use_weighting, valid_masks):
    B, T = valid_masks.shape
    seq_lengths = valid_masks.sum(dim=1).long()
    if use_weighting:
        time_weights = torch.arange(T, device=valid_masks.device)
        time_weights = time_weights.unsqueeze(0).expand(B, -1)
        time_weights = time_weights / seq_lengths.unsqueeze(1)
        time_weights = 5 * torch.exp(-3 * time_weights) + 1
        time_weights = time_weights * valid_masks
        normalizer = time_weights.sum(-1) / seq_lengths
        time_weights = time_weights / normalizer.unsqueeze(1)
    else:
        time_weights = valid_masks.float()
    return time_weights


def aggregate_monitor_loss(
    losses: torch.Tensor,
    valid_masks: torch.Tensor,
    labels: torch.Tensor,
    weights: list[float],
    one_loss_per_seq: bool = False,
):
    B = losses.shape[0]
    fail_mask = labels == 0
    success_mask = labels == 1

    if one_loss_per_seq:
        sampled_indices = torch.multinomial(valid_masks.float(), num_samples=1).squeeze(-1)
        seq_loss = losses[torch.arange(B), sampled_indices]
    else:
        seq_loss = (losses * valid_masks).sum(-1) / valid_masks.sum(-1)

    success_loss = (success_mask * seq_loss).sum()
    fail_loss = (fail_mask * seq_loss).sum()
    monitor_loss = weights[0] * fail_loss + weights[1] * success_loss
    monitor_loss = monitor_loss / B

    avg_fail_loss = fail_loss / fail_mask.sum() if fail_mask.sum() > 0 else torch.tensor(0.0)
    avg_success_loss = success_loss / success_mask.sum() if success_mask.sum() > 0 else torch.tensor(0.0)

    return monitor_loss, avg_success_loss, avg_fail_loss


def hard_negative_loss(preds, labels, valid_mask, alpha, beta=None):
    masked_preds = torch.where(valid_mask > 0.5, preds, torch.tensor(-float("inf"), device=preds.device))
    if beta is None:
        s = masked_preds.max(dim=1).values
    else:
        s = (1.0 / beta) * torch.logsumexp(beta * masked_preds, dim=1)
    pos_loss = torch.clamp((0.5 + alpha) - s, min=0)
    neg_loss = torch.clamp(s - (0.5 - alpha), min=0)
    loss = labels * (pos_loss ** 2) + (1 - labels) * (neg_loss ** 2)
    return loss.mean()
