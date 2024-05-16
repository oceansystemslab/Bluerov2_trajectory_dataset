import torch
import warnings

import numpy as np

def get_device(gpu=0, cpu=False):
    if not cpu:
        cpu = not torch.cuda.is_available()
        if cpu:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")

    device = torch.device(f"cuda:{gpu}" if not cpu else "cpu")
    return device

device = get_device()

print("*"*5, " NUMPY ", "*"*5)

x_np = np.array([[11041., 13359, 15023, 18177], [13359, 16165, 18177, 21995], [15023, 18177, 20453, 24747], [18177, 21995, 24747, 29945]])
y_np = np.array([[29945., -24747, -21995, 18177], [-24747, 20453, 18177, -15023], [-21995, 18177, 16165, -13359], [18177, -15023, -13359, 11041]])

print(x_np@y_np)

print("*"*5, " TORCH FROM NUMPY CPU ", "*"*5)

x = torch.tensor(x_np)
y = torch.tensor(y_np)
print(x@y)

print("*"*5, " TORCH FROM NUMPY GPU ", "*"*5)

x.to(device)
y.to(device)
print(x@y)

print("*"*5, " TORCH ", "*"*5)

x = torch.tensor([[11041., 13359, 15023, 18177], [13359, 16165, 18177, 21995], [15023, 18177, 20453, 24747], [18177, 21995, 24747, 29945]])
y = torch.tensor([[29945., -24747, -21995, 18177], [-24747, 20453, 18177, -15023], [-21995, 18177, 16165, -13359], [18177, -15023, -13359, 11041]])
print(x@y)

print("*"*5, " TORCH GPU ", "*"*5)

x.to(device)
y.to(device)
print(x@y)