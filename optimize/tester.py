import torch
import torch.nn as nn

@torch.no_grad()
def test_regression(model, X_test, y_test):
    """
    Test a regression model using the given test data.
    """
    criterion = nn.MSELoss()
    y_pred = model(X_test)
    loss = criterion(y_pred, y_test)
    return loss