import torch

def main():
    # Car data: [age in years, mileage in thousands]
    inputs = torch.tensor([
        [0.0, 0.0],
        [5.0, 50.0],
        [10.0, 100.0],
        [15.0, 150.0],
        [20.0, 200.0],
        [25.0, 250.0],
        [30.0, 300.0]
    ])
    
    # Corresponding car values
    targets = torch.tensor([
        30000.0,
        22000.0,
        15000.0,
        9000.0,
        5000.0,
        2500.0,
        1000.0
    ])

    # Normalize inputs
    min_vals = inputs.min(0, keepdim=True)[0]
    max_vals = inputs.max(0, keepdim=True)[0]
    normalized_inputs = (inputs - min_vals) / (max_vals - min_vals)

    # Init weights and bias
    weights = torch.zeros(2, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)

    # initialize the optimizer
    optimizer = torch.optim.SGD([weights, bias], lr = 0.01)

    # Training
    epochs = 10000
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(normalized_inputs)):
            x = normalized_inputs[i]
            y = targets[i]
            pred = weights @ x + bias
            loss = (pred - y) ** 2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    # Input: Gene's car data (age in years, mileage in thousands)
    gene = torch.tensor([25.0, 219.875])  # Make sure values are floats
    thor = torch.tensor([13.0, 150.0])

    # Normalize using saved min/max
    gene_normalized = (gene - min_vals[0]) / (max_vals[0] - min_vals[0])
    gene_normalized = gene_normalized.view(-1)  # Ensure it's a 1D tensor

    thor_normalized = (thor - min_vals[0]) / (max_vals[0] - min_vals[0])
    thor_normalized = thor_normalized.view(-1)

    # Make prediction
    gene_prediction = weights @ gene_normalized + bias
    thor_prediction = weights @ thor_normalized + bias
    print(f"${gene_prediction.item():.2f}")
    print(f"${thor_prediction.item():.2f}")

    # save everything to use later
    torch.save({
        'weights': weights.detach(),
        'bias': bias.detach(),
        'min_vals': min_vals,
        'max_vals': max_vals
    }, 'car_values.pth')

if __name__ == "__main__":
    main()