import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Seed the global numpy, torch, and random RNGs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
