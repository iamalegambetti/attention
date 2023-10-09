import torch.optim as optim
import torch.nn as nn

def train_regression(model, X_train, y_train, EPOCHS = 100, lr = 0.001):
    """
    Toy function to train a regression model using the given training data.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print("Epoch: ", epoch, "Loss: ", loss.item())

    return model