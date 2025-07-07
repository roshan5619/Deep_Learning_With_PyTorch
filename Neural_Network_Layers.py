import torch
import torch.nn as nn


input_layer=torch.tensor([[0.34,0.47,0.34]])
linear_layer=nn.Linear(in_features=3,out_features=2) 
output_layer=linear_layer(input_layer)

print(output_layer)

print(linear_layer.weight) #it will display the weights passed in outr linear layer
print(linear_layer.bias)#it will display the bias passed in our linear layer

#Hidden Layers and Parameters
import torch
import torch.nn as nn

input_tensor1 = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Create a container for stacking linear layers
model = nn.Sequential(nn.Linear(8, 4),
                nn.Linear(4, 1)
                )

output = model(input_tensor1)
print(output)

#TOTAL PARAMETERS
import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(9, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, 1))

total = 0

# Calculate the number of parameters in the model
for p in model.parameters():
    total += p.numel()
  
print(f"The number of parameters in the model is {total}")