import torch
from train_model import CNNModel 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)  # Your CNN model
dummy_input = torch.randn(1, 3, 224, 224)  # Dummy image
torch.onnx.export(model, dummy_input, "model.onnx")
