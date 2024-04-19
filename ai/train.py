import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class MouseDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = StandardScaler()

        X = self.dataframe[['dx', 'dy', 'dt', 'd']].values
        X = self.scaler.fit_transform(X)  # Normalize features
        self.X = torch.tensor(X, dtype=torch.float32)
        
        y = self.dataframe[['targetX', 'targetY']].values
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MouseModel(nn.Module):
    def __init__(self):
        super(MouseModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)

dataset = MouseDataset('mouse_data.csv')
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = MouseModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1)

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Loss: {loss.item():.6f}')

for epoch in range(10000): 
    train(epoch, model, train_loader, criterion, optimizer)