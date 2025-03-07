import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
