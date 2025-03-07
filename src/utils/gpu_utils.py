import torch


def get_device() -> torch.device:  # Changed return type to torch.device
    """
    Determines the appropriate device (GPU or CPU) for computation.

    Returns:
        A torch.device object representing "cuda" if a GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # Return torch.device("cuda")
    else:
        return torch.device("cpu")  # Return torch.device("cpu")
