import torch
import torch.nn as nn

# Load saved model data
model_data = torch.load('car_values_nn.pth')

# rebuild model structure
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
model.load_state_dict(model_data['model_state'])
model.eval()

min_vals = model_data['min_vals']
max_vals = model_data['max_vals']

# normalize test input
test_input = torch.tensor([[25.0, 219.875]])
test_input_normalized = (test_input - min_vals) / (max_vals - min_vals)

# predict
prediction = model(test_input_normalized).item()
print(f"Predicted price for Gene using neural networks: ${prediction:.2f}")