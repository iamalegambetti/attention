#!/usr/bin/env python

import sys
sys.path.insert(0, '../Attention/')

import pandas as pd 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from attention.encoder import Encoder
from optimize.trainer import train_toy_regression
from optimize.tester import test_toy_regression
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

class Net(nn.Module):
    def __init__(self, n_blocks, n_heads, embed_dim, head_dim):
        super(Net, self).__init__()
        self.encoder = Encoder(n_blocks, n_heads, embed_dim, head_dim)
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

n_blocks = 1
n_heads = 4
embed_dim = 8 # this is basically the number of features
head_dim = 16
model = Net(n_blocks, n_heads, embed_dim, head_dim)

print('Training...')
epochs = 20
lr = 0.001
model = train_toy_regression(model, X_train, y_train, EPOCHS=epochs, lr=lr)
print()

print('Testing...')
test_loss = test_toy_regression(model, X_test, y_test)
print('Test Loss: ', test_loss.item())
print()

print('Benchmarking Linear Regression...')
lr = LinearRegression()
lr.fit(X_train.numpy(), y_train.numpy())
preds = lr.predict(X_test.numpy())
benchmark = mean_squared_error(y_test.numpy(), preds)
print('Benchmark Loss: ', benchmark)