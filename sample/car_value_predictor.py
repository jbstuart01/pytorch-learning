import torch

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
print(f"Input tensor: {inputs}")
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
print(f"Normalized input tensor: {normalized_inputs}")

# Init weights and bias
weights = torch.zeros(2, requires_grad=True)
print(f"Weights: {weights}")
bias = torch.zeros(1, requires_grad=True)
print(f"Bias: {weights}")

# Optimizer
optimizer = torch.optim.SGD([weights, bias], lr=0.001)

# Training
epochs = 5000
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
#    if (epoch + 1) % 100 == 0:
#        print(f'Epoch {epoch+1}, Loss: {total_loss / len(inputs):.2f}')

# Test: 25-year-old car, 220k miles
test_input = torch.tensor([25.0, 220.0])
test_input_norm = (test_input - min_vals[0]) / (max_vals[0] - min_vals[0])
prediction = weights @ test_input_norm + bias
print(f"\nEstimated price for 25 yrs / 220k miles: ${prediction.item():.2f}")

# Test: 13-year-old car, 151k miles
test_input = torch.tensor([13.0, 151.0])
test_input_norm = (test_input - min_vals[0]) / (max_vals[0] - min_vals[0])
prediction = weights @ test_input_norm + bias
print(f"Estimated price for 13 yrs / 151k miles: ${prediction.item():.2f}")
