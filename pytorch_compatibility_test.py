import torch
print(torch.__file__)

print(f"Pytorch Version (cuda): {torch.version.cuda}")

print(f"Is Cuda Available? {torch.cuda.is_available()}")
