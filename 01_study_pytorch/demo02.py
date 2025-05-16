# Use GPU

import torch

# Check if GPU is available
print(torch.__version__)
print(f"Is CUDA available in Nvidia GPU : {torch.cuda.is_available()}")

print(f"mac mps is available : {torch.backends.mps.is_available()}")
print(f"mac mps enabled : {torch.backends.mps.is_built()}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# Define a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)
# Move the tensor to GPU
x = x.to(device)
print(x)
