import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from model import LungCancerModel  # Import the LungCancerModel from model.py
import os

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model_path = "lung_model_pt_final.pth"  # Path to the trained model file
model = LungCancerModel(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # Load model state dict
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Welcome message
st.title("Lung Cancer Detection Chatbot")
st.write("Hello! Welcome to the Lung Cancer Detection Assistant.")
st.write("I will guide you through the process to assess your condition.")

# Step 1: Collect personal details (name, age)
name = st.text_input("What's your name?")
age = st.number_input("How old are you?", min_value=1)

if name and age:
    st.write(f"Hello {name}, age {age}! Let's get started.")

    # Step 2: Ask about symptoms
    symptoms = st.text_area("Please describe your symptoms (e.g., coughing, chest pain, shortness of breath, etc.)")

    if symptoms:
        st.write("Thank you for sharing your symptoms. Now, let's proceed to the next step.")

        # Step 3: Upload an image
        uploaded_file = st.file_uploader("Upload an X-ray or CT scan image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Open image for prediction
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Step 4: Predict using the lung classifier
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                class_names = ["Benign", "Malignant", "Normal"]  # Class names for prediction
                prediction = class_names[predicted.item()]

            # Step 5: Display prediction result
            st.write(f"Prediction: {prediction}")

            # Step 6: Suggestions
            if prediction == "Malignant":
                st.write("Based on the prediction, it seems you may have a malignant condition. We recommend that you meet a doctor immediately for further consultation and possibly a biopsy.")
            elif prediction == "Benign":
                st.write("It seems like the condition might be benign. However, we still suggest you follow up with a healthcare professional for further guidance.")
            else:
                st.write("The condition seems normal. However, if you're experiencing persistent symptoms, we recommend checking with a healthcare provider.")

        else:
            st.write("Please upload an image to continue.")
