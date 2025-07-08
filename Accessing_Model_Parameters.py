import torch.nn as nn


model = nn.Sequential(nn.Linear(16, 8),
                      nn.Linear(8, 2)
                     )

# Access the weight of the first linear layer
weight_0 = model[0].weight
print("Weight of the first layer:", weight_0)

# Access the bias of the second linear layer
bias_1 = model[1].bias
print("Bias of the second layer:", bias_1)

####################
lr=0.001
#Updating Weights Manually
weight0 = model[0].weight
weight1 = model[1].weight
#weight2 = model[2].weight

# Access the gradients of the weight of each linear layer
grads0 = weight0.grad
grads1 = weight1.grad
#grads2 = weight2.grad

# Update the weights using the learning rate and the gradients
weight0 = weight0-lr*grads0
weight1 = weight1-lr*grads1
#weight2 = weight2-lr*grads2


###################333
#Using PyTorch Optimizer
# Create the optimizer
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = criterion(pred, target)
loss.backward()

# Update the model's parameters using the optimizer
optimizer.step()