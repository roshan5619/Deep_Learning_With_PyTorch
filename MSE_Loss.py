

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# === Compare MSE using NumPy and PyTorch ===
y_pred = np.array([3, 5.0, 2.5, 7.0])
y_true = np.array([3.0, 4.5, 2.0, 8.0])

# NumPy MSE
mse_numpy = np.mean((y_pred - y_true) ** 2)

# PyTorch MSE
criterion = nn.MSELoss()
mse_pytorch = criterion(torch.tensor(y_pred, dtype=torch.float32),
                        torch.tensor(y_true, dtype=torch.float32))

print("MSE (NumPy):", mse_numpy)
print("MSE (PyTorch):", mse_pytorch.item())

# === Dummy dataset ===
X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)  # y = 2x + 1 + noise

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# === Simple linear regression model ===
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

# === Training loop ===
num_epochs = 5
for epoch in range(num_epochs):
    for feature, target in dataloader:
        prediction = model(feature)
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# === Show results ===
def show_results(model, dataloader):
    model.eval()
    all_features = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for feature, target in dataloader:
            prediction = model(feature)
            all_features.append(feature)
            all_predictions.append(prediction)
            all_targets.append(target)

    X_all = torch.cat(all_features).numpy()
    y_pred_all = torch.cat(all_predictions).numpy()
    y_true_all = torch.cat(all_targets).numpy()

    plt.scatter(X_all, y_true_all, label='True')
    plt.scatter(X_all, y_pred_all, label='Predicted', marker='x')
    plt.legend()
    plt.title("Model Predictions vs True Values")
    plt.xlabel("Input Feature")
    plt.ylabel("Target")
    plt.show()

# Call show_results
show_results(model, dataloader)
