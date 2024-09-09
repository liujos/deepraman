import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import RamanNN
import os

class RamanDataset(Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.spectra = self.data.iloc[:, 1:-1].to_numpy()
        self.labels = self.data.iloc[:, -1:].to_numpy().flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectrum = torch.tensor(self.spectra[idx], dtype=torch.float32)
        spectrum = torch.reshape(spectrum, (1, -1))
        label = torch.tensor(self.labels[idx])
        return spectrum, label


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X) 
            print(f"loss: {loss:>7f} [{current}/{len(dataloader.dataset)}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"\nTest Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


data = RamanDataset('spectra/train_data.csv')
train, validation = torch.utils.data.random_split(data, [0.9, 0.1])

model = RamanNN()
learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas=(0.9, 0.95))
loss_fn = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation, batch_size=32, shuffle=True)

if not os.path.isdir('weights'):
    os.mkdir('weights')

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}:\n------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(validation_loader, model, loss_fn)
    torch.save(model.state_dict(), f"weights/weights_{t+1}")
print("Done!")


