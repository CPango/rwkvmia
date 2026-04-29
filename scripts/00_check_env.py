import sys

try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    torch_version = torch.__version__
except Exception:
    cuda_available = False
    cuda_count = 0
    torch_version = "not installed"

try:
    import transformers
    transformers_version = transformers.__version__
except Exception:
    transformers_version = "not installed"

print("python:", sys.version)
print("torch:", torch_version)
print("transformers:", transformers_version)
print("cuda_available:", cuda_available)
print("cuda_device_count:", cuda_count)
