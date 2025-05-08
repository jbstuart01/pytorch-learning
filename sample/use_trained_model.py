import torch

# Load saved model data
model_data = torch.load('car_values.pth')
weights = model_data['weights']
bias = model_data['bias']
min_vals = model_data['min_vals']
max_vals = model_data['max_vals']

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

print(f"Estimated price for Gene: ${gene_prediction.item():.2f}")
print(f"Estimated price for Thor: ${thor_prediction.item():.2f}")
