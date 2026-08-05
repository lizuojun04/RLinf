import torch
from torch import nn

from .base import BaseModel
from .utils import aggregate_monitor_loss, get_time_weight


class IndepModel(BaseModel):
    """Per-timestep independent projector (SAFE MLP)."""

    def __init__(self, input_dim: int, cfg: dict):
        super().__init__(input_dim, cfg)
        self.total_input_dim = input_dim * cfg.get("n_history_steps", 1)
        self.hidden_dim = cfg.get("hidden_dim", 256)

        projector = []
        n_layers = cfg.get("n_layers", 2)
        if n_layers == 1:
            projector.append(nn.Linear(self.total_input_dim, 1))
        else:
            projector.append(nn.Linear(self.total_input_dim, self.hidden_dim))
            projector.append(nn.ReLU())
            for _ in range(n_layers - 2):
                projector.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                projector.append(nn.ReLU())
            projector.append(nn.Linear(self.hidden_dim, 1))

        final_act = cfg.get("final_act_layer", "sigmoid")
        if final_act == "sigmoid":
            projector.append(nn.Sigmoid())
        elif final_act == "relu":
            projector.append(nn.ReLU())
        elif final_act == "none":
            pass
        else:
            raise ValueError(f"Unknown final activation: {final_act}")

        self.projector = nn.Sequential(*projector)

    def forward(self, batch):
        x = batch["features"]
        assert x.ndim == 3 and x.shape[-1] == self.input_dim
        x = self.projector(x)
        if self.cfg.get("cumsum", False) or self.cfg.get("rmean", False):
            x = torch.cumsum(x, dim=-2)
            if self.cfg.get("rmean", False):
                x = x / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        return x

    def forward_compute_loss(self, batch, weights=None):
        features, valid_masks, labels = batch["features"], batch["valid_masks"], batch["success_labels"]
        B, T, D = features.shape
        scores = self(batch).squeeze(-1)

        time_weights = get_time_weight(self.cfg.get("use_time_weighting", False), valid_masks).to(scores)

        higher_thresh = self.cfg.get("threshold", 50)
        lower_thresh = 0
        seq_loss_success = torch.relu(scores - lower_thresh)
        if self.cfg.get("use_threshold", False):
            seq_loss_fail = time_weights * torch.relu(higher_thresh - scores)
        else:
            seq_loss_fail = time_weights * (-scores)

        losses = (labels == 1).float()[:, None] * seq_loss_success + (
            labels == 0
        ).float()[:, None] * seq_loss_fail

        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(losses, valid_masks, labels, weights)
        return monitor_loss, {"monitor_loss": monitor_loss.item(), "success_loss": success_loss.item(), "fail_loss": fail_loss.item()}
