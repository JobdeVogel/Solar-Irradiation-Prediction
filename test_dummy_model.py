import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys
import time

import wandb
# wandb.login()

# Generate random data
data = np.random.rand(10000, 2500).astype(np.float32)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :]

# Create an instance of the dataset and a data loader
custom_dataset = CustomDataset(data)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(2500, 2500)
        self.fc2 = nn.Linear(2500, 2500)
        self.fc3 = nn.Linear(2500, 2500)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x

# Instantiate the model
model = DummyModel()

# Define loss function and optimizer
# criterion = nn.L1Loss() #0.0777
criterion = nn.MSELoss() #0.0745
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# with wandb.init(mode="online", project="dummy"):
# Training loop
epochs = 1000

start = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    for i, inputs in enumerate(data_loader, 0):
        optimizer.zero_grad()

        # Move inputs to the GPU
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)

        # Use inputs as targets for this example
        targets = inputs

        print(outputs.shape)
        print(targets.shape)
        sys.exit()

        # Calculate loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            avg_loss = running_loss / 100
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {running_loss / 10:.5f}")
            
            # # Log the average loss to wandb
            # wandb.log({"Epoch": epoch + 1, "Batch": i + 1, "Loss": avg_loss})
            
            running_loss = 0.0

    end = time.time()
    if end - start > 30:
        break

with torch.no_grad():
    inputs = next(iter(data_loader)).to(device)

    targets = inputs
    
    outputs = model(inputs)
    criterion = nn.MSELoss()

    loss = criterion(outputs, targets)
    print(f'Test loss: {loss}')

    print("Finished Training")