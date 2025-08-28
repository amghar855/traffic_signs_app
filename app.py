import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO("best.pt")

st.title("🚦 Traffic Sign Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Make predictions
    results = model.predict(image)

    # Draw bounding boxes and display results
    res_plotted = results[0].plot()  # returns a numpy array
    st.image(res_plotted, caption="📍 Prediction Result", use_column_width=True)
