import torch
from torch import nn

from .base import BaseModel
from .utils import (
    aggregate_monitor_loss,
    cumsum_stopgrad,
    get_time_weight,
    hard_negative_loss,
)


class LstmModel(BaseModel):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__(input_dim, cfg)
        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.n_layers = cfg.get("n_layers", 1)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=cfg.get("dropout", 0.0))
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.n_history_steps = cfg.get("n_history_steps", -1)
        self._scale_weights(cfg.get("init_weight_scale", 1.0))

    def forward(self, batch):
        x = batch["features"]
        B, T, D = x.shape
        assert D == self.input_dim
        n = self.n_history_steps
        if n < 0:
            out, _ = self.lstm(x)
        else:
            x_padded = torch.nn.functional.pad(x, (0, 0, n, 0), mode="constant", value=0)
            x_windows = [x_padded[:, t : t + n, :] for t in range(T)]
            x_seq = torch.stack(x_windows, dim=1).reshape(B * T, n, D)
            out, _ = self.lstm(x_seq)
            out = out[:, -1, :].view(B, T, -1)
        out = self.dropout(out)
        p_seq = torch.sigmoid(self.fc(out))
        if self.cfg.get("cumsum", False):
            p_seq = cumsum_stopgrad(p_seq, dim=1)
            if self.cfg.get("rmean", False):
                normalizer = p_seq.new_ones(p_seq.shape).cumsum(dim=1)
                p_seq = p_seq / normalizer
        return p_seq

    def forward_compute_loss(self, batch, weights=None):
        valid_masks = batch["valid_masks"]
        success_labels = batch["success_labels"]
        B, T, D = batch["features"].shape
        scores = self(batch).squeeze(-1)
        time_weights = get_time_weight(self.cfg.get("use_time_weighting", False), valid_masks).to(scores)

        if self.cfg.get("cumsum", False):
            lower_thresh = 0
            seq_loss_success = torch.relu(scores - lower_thresh)
            seq_loss_fail = time_weights * (-scores)
            losses = (success_labels == 1).float()[:, None] * seq_loss_success + (
                success_labels == 0
            ).float()[:, None] * seq_loss_fail
        else:
            criterion = nn.BCELoss(reduction="none")
            losses = criterion(scores, 1 - success_labels.unsqueeze(-1).expand_as(scores))
            losses[success_labels == 0] *= time_weights[success_labels == 0]

        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses, valid_masks, success_labels, weights, self.cfg.get("one_loss_per_seq", False)
        )

        hard_neg_loss = torch.tensor(0.0, device=scores.device)
        lambda_hard = self.cfg.get("lambda_hard_neg", 0.0)
        if lambda_hard > 0:
            hard_neg_loss = hard_negative_loss(
                scores, 1 - success_labels, valid_masks,
                self.cfg.get("hard_neg_margin", 0.1), self.cfg.get("hard_neg_beta", 50.0),
            )
            hard_neg_loss = lambda_hard * hard_neg_loss
        monitor_loss += hard_neg_loss

        return monitor_loss, {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": hard_neg_loss.item(),
        }
