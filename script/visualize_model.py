# visualize_model.py

import torch
from torchviz import make_dot
from train_model import CNNModel  # Ensure CNNModel is imported from your training script

# Create an instance of your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)

# Create a dummy input tensor with the shape your model expects
dummy_input = torch.randn(1, 3, 224, 224)

# Run the model on the dummy input
output = model(dummy_input)

# Use make_dot to generate the graph
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as a PNG file
dot.render("CNN_Model_Visualization", format="png")
print("CNN_Model_Visualization.png has been generated.")
