import abc

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self._device = "cpu"

    @abc.abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward_compute_loss(
        self, batch: dict[str, torch.Tensor], weights: list[float] | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def _scale_weights(self, scale_factor: float):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name and param is not None:
                    param.mul_(scale_factor)

    def train_epoch(self, optimizer: torch.optim.Optimizer, dataloader: DataLoader) -> float:
        device = self.get_device()
        total_losses: list[float] = []
        weights = dataloader.dataset.get_class_weights()

        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss, logs = self.forward_compute_loss(batch, weights)
            reg_loss, _ = self.compute_regularization_loss(self.cfg.get("lambda_reg", 0.0))
            total_loss = loss + reg_loss
            optimizer.zero_grad()
            total_loss.backward()
            if self.cfg.get("grad_max_norm") is not None:
                clip_grad_norm_(self.parameters(), max_norm=self.cfg["grad_max_norm"])
            optimizer.step()
            total_losses.append(total_loss.item())

        return sum(total_losses) / len(total_losses)

    def compute_regularization_loss(self, lambda_reg: float) -> tuple[torch.Tensor, dict[str, float]]:
        if lambda_reg == 0:
            return torch.tensor(0.0), {}
        reg_loss = sum(torch.sum(p ** 2) for name, p in self.named_parameters() if "bias" not in name)
        reg_loss = lambda_reg * reg_loss
        return reg_loss, {"reg_loss": reg_loss.item()}

    def to(self, device):
        self._device = device
        return super().to(device)

    def get_device(self):
        return self._device

    def get_optimizer(self):
        if self.cfg.get("optimizer", "adam") == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"])
        elif self.cfg["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg.get("weight_decay", 1e-2))
        elif self.cfg["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg["lr"])
        elif self.cfg["optimizer"] == "sgdm":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg["lr"], momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.get('optimizer')}")

        step_scheduler = StepLR(optimizer, step_size=self.cfg.get("lr_step_size", 300), gamma=self.cfg.get("lr_gamma", 1.0))
        warmup_steps = self.cfg.get("warmup_steps", 0)
        if warmup_steps > 0:
            def lr_lambda(step):
                return min((step + 1) / warmup_steps, 1.0)
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[warmup_steps])
        else:
            scheduler = step_scheduler

        return optimizer, scheduler
