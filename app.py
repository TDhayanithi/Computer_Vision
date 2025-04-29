import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import LungCancerModel  # Import your actual model

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = LungCancerModel(num_classes=3)
model.load_state_dict(torch.load("lung_model_pt_final.pth", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# Class names
classes = ['Benign', 'Malignant', 'Normal']

# Streamlit UI
st.title("Lung Cancer Classification")
st.write("Upload a lung scan image to predict whether it's **Benign**, **Malignant**, or **Normal**.")

uploaded_file = st.file_uploader("Choose a lung image...", type=["jpg", "jpeg", "png"])

# Image preprocessing and prediction
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

# Display prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Lung Image", use_container_width=True)
    st.write("Classifying...")
    prediction = predict(image)
    st.success(f"Prediction: **{prediction}**")
