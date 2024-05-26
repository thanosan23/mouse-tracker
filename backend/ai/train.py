import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from backend.config import Config
from model import MouseModel

class MouseDataset(Dataset):
    def __init__(self, csv_file, window_size):
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = MaxAbsScaler()
        self.output_scaler = MaxAbsScaler()
        self.window_size = window_size

        X = self.dataframe[['currentX', 'currentY', 'dx', 'dy', 'dt', 'd']].values
        X = self.scaler.fit_transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        y = self.dataframe[['targetX', 'targetY']].values
        y = self.output_scaler.fit_transform(y)
        self.y = torch.tensor(y, dtype=torch.float32)

        with open('output_scaler.pkl', 'wb') as f:
            pickle.dump(self.output_scaler, f)

    def __len__(self):
        return len(self.dataframe) - self.window_size + 1

    def __getitem__(self, idx):
        X_window = self.X[idx:idx + self.window_size].flatten()
        y_target = self.y[idx + self.window_size - 1]
        return X_window, y_target

window_size = Config.WINDOW_SIZE
input_size = Config.INPUT_SIZE

dataset = MouseDataset('Mouse Data.csv', window_size=window_size)
train_size = int(Config.TRAIN_SIZE * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)

model = MouseModel(input_size=input_size)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=Config.SCHEDULER_PATIENCE, factor=0.5)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)

train_losses = []
test_losses = []

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    loss = torch.tensor(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.6f}')
    train_losses.append(loss.item())

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
    test_loss /= len(test_loader.dataset)
    print(f'Average Test Loss: {test_loss:.6f}')
    test_losses.append(test_loss)
    return test_loss

for epoch in range(Config.EPOCHS):
    train(epoch, model, train_loader, criterion, optimizer)
    test_loss = evaluate(model, test_loader, criterion)
    scheduler.step(test_loss)
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

torch.save(model.state_dict(), 'mouse_model.pth')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.title('Training and Test Loss')
plt.savefig('loss.png')
