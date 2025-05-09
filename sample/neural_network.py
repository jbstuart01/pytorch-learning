import torch
import torch.nn as nn
import torch.optim as optim
import random

class AdditionModel(nn.Module):
    def __init__(self):
        super(AdditionModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # generate 1000 random [a, b] pairs with their sums
    inputs = torch.rand(1000, 2) * 1000  # numbers between 0 and 1000

    # Inputs: [a, b], Labels: [a + b]
    targets = inputs.sum(dim=1, keepdim=True)

    # separate testing vs training sets
    train_inputs = inputs[:800]
    train_targets= targets[:800]
    test_inputs = inputs[800:]
    test_targets = inputs[800:]

    # training setup
    model = AdditionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    # test on unseen inputs
    with torch.no_grad():
        preds = model(test_inputs)
        test_loss = nn.MSELoss()(preds, test_targets)
        print(f"Test Loss: {test_loss.item():.8f}")

    # training loop
    epochs = 10000
    for epoch in range(epochs):
        predictions = model(train_inputs)
        loss = criterion(predictions, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.8f}")

    with torch.no_grad():
        test_preds = model(test_inputs)
        test_loss = criterion(test_preds, test_targets)
        print(f"\nTest Loss: {test_loss.item():.4f}")

        # manual prediction
        test = torch.tensor([[123, 456]], dtype=torch.float32)
        predicted_sum = model(test).item()
        print(f"\nPredicted sum of [123, 456]: {predicted_sum:.8f}")

if __name__ == "__main__":
    main()