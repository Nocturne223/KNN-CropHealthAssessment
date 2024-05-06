import os
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Set page title and favicon
st.set_page_config(page_title="Crop Health Assessment App", page_icon="🌱")

# Define function to enhance design and layout
def enhance_ui():
    # Set page layout
    st.title("Crop Health Assessment App")
    st.write("Welcome to the Crop Health Assessment App! Upload an image of a plant to analyze its health.")
    
    # Add image uploader
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    # Add interactive elements
    st.sidebar.subheader("Analysis Settings")
    threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Adjust the confidence threshold for classification")
    crop_type = st.sidebar.selectbox("Select Crop Type", ["Cauliflower", "Pepper", "Sugarcane", "Lettuce"], help="Choose the type of crop for analysis")
    
    # Add informative tooltips
    st.info("ℹ️ Tip: Adjust the confidence threshold to control the sensitivity of the analysis.")
    st.info("For optimal results, ensure that the uploaded image is clear and properly centered on the plant of interest.")

# Run the app
if __name__ == "__main__":
    enhance_ui()