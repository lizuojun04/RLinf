def move_to_device(data, device):
    """Recursively move tensors in a nested structure to ``device``."""
    if hasattr(data, "to") and data.__class__.__module__.startswith("torch"):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    return data
