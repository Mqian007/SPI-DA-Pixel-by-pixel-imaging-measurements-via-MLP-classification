import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from mlp_spi_da import DATA

n=235 ## n = Image pixels / c ，c is a constant
class_all = 11        # class of number
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n, 235)
        self.fc2 = nn.Linear(235, 168)         ###change number if necessary
        self.fc3 = nn.Linear(168, 148)
        self.fc4 = nn.Linear(148, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, class_all)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        x = torch.relu(self.dropout(self.fc3(x)))
        x = torch.relu(self.dropout(self.fc4(x)))
        x = torch.relu(self.dropout(self.fc5(x)))
        x = torch.relu(self.dropout(self.fc6(x)))
        x = self.fc7(x)
        return x

def star(model, train_loader, test_loader, LOSS, opt, epochs, patience=10):
    model.train()
    best_accuracy = 0
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(data)
            loss = LOSS(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        accuracy = evaluate(model, test_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model_nn.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        if (epoch + 1) % 25000 == 0:
            continue_SPI = input('Continue training? Enter 1 for yes or 0 for no: ')
            if continue_SPI == '0':
                break

def evaluate(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return 100.0 * total_correct / total_samples

device = torch.device("cpu")  ####CPU/GPU
model = MLP().to(device)

LOSS = nn.CrossEntropyLoss()   ## LOSS
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

dataset = DATA(root_dir='SPI_movement_SIM_train2')

train_, test_= train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_, batch_size=16)
test_loader = DataLoader(test_, batch_size=16)

star(model, train_loader, test_loader, LOSS, opt, epochs=50000)