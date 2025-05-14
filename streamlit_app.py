# streamlit_app.py
"""
Streamlit web interface for barcode and QR code detection
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from detector import Detector
from test_generator import TestGenerator

# Set page config
st.set_page_config(
    page_title="Barcode & QR Code Detector",
    page_icon="📱",
    layout="wide"
)


# Initialize detector
@st.cache_resource
def get_detector():
    return Detector()


def main():
    st.title("📱 Barcode & QR Code Detection System")
    st.write("Upload an image to detect and decode barcodes and QR codes")

    # Sidebar with options
    st.sidebar.title("Options")

    # Generate test images button
    if st.sidebar.button("Generate Test Images"):
        with st.spinner("Generating test images..."):
            generator = TestGenerator()
            generator.create_simple_test_set()
        st.sidebar.success("Test images generated!")

    # Option to use test images
    test_images = []
    if os.path.exists('test_images'):
        test_images = [f for f in os.listdir('test_images') if f.endswith(('.png', '.jpg', '.jpeg'))]

    if test_images:
        st.sidebar.subheader("Try Test Images")
        selected_test = st.sidebar.selectbox("Select a test image:", ['None'] + test_images)

        if selected_test != 'None':
            st.sidebar.image(f'test_images/{selected_test}', caption=selected_test, width=200)
            if st.sidebar.button("Process Test Image"):
                process_image(f'test_images/{selected_test}')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        image.save(temp_path)

        # Process button
        if st.button("🔍 Detect Codes", type="primary"):
            process_image(temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_image(image_path):
    """Process an image and display results"""
    detector = get_detector()

    with st.spinner("Processing image..."):
        result = detector.detect_codes(image_path)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Detection Results")
        st.metric("Processing Time", f"{result['processing_time']:.3f} seconds")
        st.metric("Barcode Regions", len(result['barcode_regions']))
        st.metric("QR Code Regions", len(result['qr_regions']))
        st.metric("Total Codes", result['total_codes'])

        if result['detected_codes']:
            st.subheader("🔍 Detected Codes")
            for i, code in enumerate(result['detected_codes'], 1):
                with st.expander(f"{i}. {code['type']}"):
                    st.code(code['data'])
                    st.write(f"Category: {code['category']}")
        else:
            st.info("No codes detected")

    with col2:
        # Show visualization if available
        vis_path = result.get('visualization_path') or os.path.splitext(os.path.basename(image_path))[0] + '_detected.jpg'
        if os.path.exists(vis_path):
            st.subheader("🎯 Visualization")
            st.image(vis_path, caption="Detected codes highlighted", use_column_width=True)

    # Download button for visualization
    if os.path.exists(vis_path):
        with open(vis_path, "rb") as file:
            st.download_button(
                label="📥 Download Visualization",
                data=file.read(),
                file_name=os.path.basename(vis_path),
                mime="image/jpeg"
            )


if __name__ == "__main__":
    main()