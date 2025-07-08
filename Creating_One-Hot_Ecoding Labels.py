import torch.nn.functional as F
import numpy as np
import torch
y = 1
num_classes = 3

# Create the one-hot encoded vector using NumPy
one_hot_numpy = np.array([0,1,0])

# Create the one-hot encoded vector using PyTorch
one_hot_pytorch = (F.one_hot(torch.tensor(y), num_classes=3))

print("One-hot vector using NumPy:", one_hot_numpy)
print("One-hot vector using PyTorch:", one_hot_pytorch)