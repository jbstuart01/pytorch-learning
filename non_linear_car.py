import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # sample non-linear dataset
    inputs = torch.tensor([
        [0.0, 0.0],
        [3.0, 30.0],
        [5.0, 50.0],
        [8.0, 90.0],
        [10.0, 120.0],
        [15.0, 180.0],
        [20.0, 240.0],
        [25.0, 300.0],
        [30.0, 360.0]
    ])
    targets = torch.tensor([
        30000.0,
        25000.0,
        22000.0,
        16000.0,
        12000.0,
        7000.0,
        4000.0,
        2000.0,
        1000.0
    ])
    # reshape targets
    targets = targets.view(-1, 1)
    # normalize the inputs
    min_vals = inputs.min(0, keepdim=True)[0]
    max_vals = inputs.max(0, keepdim=True)[0]
    inputs = (inputs - min_vals) / (max_vals - min_vals)

    # define a simple neural network
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    # training loop
    epochs = 50000
    for epoch in range(epochs):
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save the model
    torch.save({
        'model_state': model.state_dict(),
        'min_vals': min_vals,
        'max_vals': max_vals
    }, 'car_values_nn.pth')

if __name__ == "__main__":
    main()