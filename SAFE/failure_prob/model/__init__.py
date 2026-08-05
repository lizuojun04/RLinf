from .base import BaseModel
from .indep import IndepModel
from .lstm import LstmModel

MODEL_REGISTRY = {
    "indep": IndepModel,
    "mlp": IndepModel,
    "lstm": LstmModel,
}


def get_model(model_name: str, input_dim: int, cfg: dict) -> BaseModel:
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](input_dim, cfg)


__all__ = ["BaseModel", "IndepModel", "LstmModel", "get_model"]
