# prototype_app.py

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from train_model import CNNModel  # Assumes CNNModel is defined here
import os

st.title("Cataract Detection Prototype (PyTorch)")
st.write("Upload a fundus image to classify it as 'Cataract' or 'Normal'.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=400)
    
    # Define the same transforms used in training (except data augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Button to trigger prediction
    if st.button("Predict"):
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load the trained model
        model = CNNModel().to(device)
        
        # Adjust the path to your saved model file as needed
        # Ensure correct path to the model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'cataract_model.pth'))

        # Debugging: Print the path
        # print("Resolved Model Path:", model_path)

        # Check if the file exists before loading
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        # model_path = os.path.join(os.getcwd(), '..', 'models', 'cataract_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Set model to evaluation mode
        model.eval()
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor.to(device))
            probability = output.item()  # Sigmoid output
            
        # Interpret the result
        result = "Cataract" if probability > 0.5 else "Normal"
        
        # Display the prediction and probability
        st.write(f"**Prediction:** {result}")
        st.write(f"**Probability:** {probability:.4f}")
