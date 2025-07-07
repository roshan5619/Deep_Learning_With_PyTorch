#From Regression to Multi-Class Classification
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Implement a neural network with exactly four linear layers
model = nn.Sequential(


nn.Linear(11,8),
nn.Linear(8,5),
nn.Linear(5,2),
nn.Linear(2,1),
nn.Sigmoid()
)


output = model(input_tensor)
print(output)


#Multi-Class Classification
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Update network below to perform a multi-class classification with four labels
model = nn.Sequential(
  nn.Linear(11, 20),
  nn.Linear(20, 12),
  nn.Linear(12, 6),
  nn.Linear(6, 4),
  nn.Softmax(dim=-1)
)

output = model(input_tensor)
print(output)