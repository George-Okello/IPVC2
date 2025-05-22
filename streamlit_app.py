# simple_streamlit_app.py
"""
Simplified Streamlit web interface that avoids column nesting issues
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from detector import Detector
from batch_detector import BatchDetector
from test_generator import TestGenerator
import zipfile
import shutil
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Barcode & QR Code Detector",
    page_icon="📱",
    layout="wide"
)

# Initialize session state
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None


# Initialize detectors
@st.cache_resource
def get_detectors():
    return Detector(), BatchDetector()


def main():
    st.title("📱 Barcode & QR Code Detection System")
    st.markdown("### Process single images or entire folders with complexity analysis")

    # Sidebar
    with st.sidebar:
        st.title("🔧 Options")

        # Tab selection
        tab_options = ["Single Image", "Batch Processing", "Performance Analysis"]
        selected_tab = st.radio("Choose Mode:", tab_options)

        # Generate test images button
        if st.button("🎯 Generate Test Images"):
            with st.spinner("Generating test images..."):
                generator = TestGenerator()
                generator.create_simple_test_set()
            st.success("✅ Test images generated!")

    # Main content based on selected tab
    if selected_tab == "Single Image":
        single_image_tab()
    elif selected_tab == "Batch Processing":
        batch_processing_tab()
    elif selected_tab == "Performance Analysis":
        performance_analysis_tab()


def single_image_tab():
    """Single image processing tab"""
    st.header("🖼️ Single Image Detection")

    detector, _ = get_detectors()

    # Test image selector
    test_images = []
    if os.path.exists('test_images'):
        test_images = [f for f in os.listdir('test_images') if f.endswith(('.png', '.jpg', '.jpeg'))]

    if test_images:
        st.subheader("Try a Test Image")
        selected_test = st.selectbox("Select test image:", ['None'] + test_images)

        if selected_test != 'None':
            test_path = f'test_images/{selected_test}'
            st.image(test_path, caption=selected_test, width=400)

            if st.button("🔍 Process Test Image", type="primary"):
                process_single_image(test_path, detector)

    st.subheader("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        image.save(temp_path)

        if st.button("🔍 Detect Codes", type="primary"):
            process_single_image(temp_path, detector)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def batch_processing_tab():
    """Batch processing tab"""
    st.header("📁 Batch Processing")

    _, batch_detector = get_detectors()

    st.subheader("Upload Images")

    # Option 1: Upload multiple files
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    # Option 2: Upload ZIP file
    st.subheader("Or Upload ZIP File")
    zip_file = st.file_uploader("Choose a ZIP file", type=['zip'])

    # Processing options
    st.subheader("Processing Options")
    max_images = st.number_input("Maximum images to process", min_value=1, max_value=100, value=20)

    # Process buttons
    if uploaded_files:
        st.write(f"📁 Selected {len(uploaded_files)} files")
        if st.button("🚀 Process Multiple Files", type="primary"):
            process_multiple_files(uploaded_files, batch_detector, max_images)

    elif zip_file:
        st.write(f"📦 ZIP file: {zip_file.name}")
        if st.button("🚀 Process ZIP File", type="primary"):
            process_zip_file(zip_file, batch_detector, max_images)

    # Process test images folder
    if os.path.exists('test_images'):
        st.subheader("Process Test Images")
        if st.button("🔍 Process Test Images Folder"):
            process_test_folder(batch_detector, max_images)


def process_single_image(image_path, detector):
    """Process a single image and display results"""
    with st.spinner("Processing image..."):
        result = detector.detect_codes(image_path)

    # Display results
    st.subheader("📊 Detection Results")

    # Show metrics
    st.write(f"⏱️ **Processing Time:** {result['processing_time']:.3f} seconds")
    st.write(f"🔢 **Total Codes Found:** {result['total_codes']}")
    st.write(f"📊 **Barcode Regions:** {len(result['barcode_regions'])}")
    st.write(f"🎯 **QR Code Regions:** {len(result['qr_regions'])}")

    # Show detected codes
    if result['detected_codes']:
        st.subheader("🔍 Detected Codes")
        for i, code in enumerate(result['detected_codes'], 1):
            with st.expander(f"{i}. {code['type']} - {code['category']}"):
                st.code(code['data'])

                # Show additional details
                if 'rotation' in code:
                    st.write(f"🔄 **Rotation:** {code['rotation']}°")
                if 'preprocess' in code:
                    st.write(f"🔧 **Preprocessing:** {code['preprocess']}")
                if 'image_variant' in code:
                    st.write(f"🖼️ **Image Variant:** {code['image_variant']}")
                if 'detection_method' in code:
                    st.write(f"⚙️ **Detection Method:** {code['detection_method']}")
    else:
        st.info("No codes detected in this image")

    # Show visualization if available
    vis_path = result.get('visualization_path')
    if vis_path and os.path.exists(vis_path):
        st.subheader("🎯 Visualization")
        st.image(vis_path, caption="Detected codes highlighted", width=600)

        # Download button
        with open(vis_path, "rb") as file:
            st.download_button(
                label="📥 Download Visualization",
                data=file.read(),
                file_name=os.path.basename(vis_path),
                mime="image/jpeg"
            )


def process_multiple_files(uploaded_files, batch_detector, max_images):
    """Process multiple uploaded files"""
    # Create temporary directory
    temp_dir = "temp_batch_upload"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Save uploaded files
        for uploaded_file in uploaded_files[:max_images]:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Process the folder
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            results = batch_detector.process_folder(temp_dir, max_images)
            st.session_state.batch_results = results

        st.success(f"✅ Processed {results['batch_info']['processed_successfully']} images successfully!")
        display_batch_results(results)

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_zip_file(zip_file, batch_detector, max_images):
    """Process uploaded ZIP file"""
    temp_dir = "temp_zip_extract"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Extract ZIP file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process the extracted folder
        with st.spinner("Processing ZIP file contents..."):
            results = batch_detector.process_folder(temp_dir, max_images)
            st.session_state.batch_results = results

        st.success(f"✅ Processed {results['batch_info']['processed_successfully']} images from ZIP!")
        display_batch_results(results)

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_test_folder(batch_detector, max_images):
    """Process test images folder"""
    with st.spinner("Processing test images folder..."):
        results = batch_detector.process_folder("test_images", max_images)
        st.session_state.batch_results = results

    st.success(f"✅ Processed {results['batch_info']['processed_successfully']} test images!")
    display_batch_results(results)


def display_batch_results(results):
    """Display batch processing results"""
    if not results or 'error' in results:
        st.error("No valid results to display")
        return

    # Summary metrics
    st.subheader("📈 Batch Summary")

    batch_info = results['batch_info']

    # Display summary information
    st.write(f"📁 **Total Images:** {batch_info['total_images']}")
    st.write(f"✅ **Successfully Processed:** {batch_info['processed_successfully']}")
    st.write(f"❌ **Failed:** {batch_info['failed_images']}")
    st.write(f"⏱️ **Total Processing Time:** {batch_info['total_batch_time']:.2f} seconds")
    st.write(f"📊 **Average Time per Image:** {batch_info['average_time_per_image']:.3f} seconds")

    # Performance analysis
    if 'performance_analysis' in results:
        perf = results['performance_analysis']

        st.subheader("⚡ Performance Analysis")
        timing_stats = perf['timing_stats']
        efficiency = perf['efficiency_metrics']

        st.write(f"🚀 **Fastest Image:** {timing_stats['fastest_time']:.3f} seconds")
        st.write(f"🐌 **Slowest Image:** {timing_stats['slowest_time']:.3f} seconds")
        st.write(f"📈 **Average Processing Time:** {timing_stats['average_time']:.3f} seconds")
        st.write(f"🎯 **Detection Success Rate:** {efficiency['detection_success_rate']:.1f}%")
        st.write(f"📊 **Images with Codes Found:** {efficiency['images_with_codes_found']}")


def performance_analysis_tab():
    """Performance analysis tab"""
    st.header("📊 Performance Analysis")

    if st.session_state.batch_results is None:
        st.info("👆 Process a batch of images first to see performance analysis")
        return

    results = st.session_state.batch_results

    if 'error' in results:
        st.error("No valid batch results available")
        return

    # Create analysis dashboard
    create_performance_dashboard(results)


def create_performance_dashboard(results):
    """Create performance dashboard"""
    # Prepare data
    detailed_results = results.get('detailed_results', [])
    if not detailed_results:
        st.error("No detailed results available")
        return

    # Convert to DataFrame
    data = []
    for result in detailed_results:
        row = {
            'filename': result['file_metadata']['filename'],
            'processing_time': result['processing_time'],
            'file_size_mb': result['file_metadata']['file_size_mb'],
            'megapixels': result['file_metadata']['megapixels'],
            'codes_found': result['total_codes'],
            'strategies_used': result['complexity_indicators']['detection_strategies_used'],
            'preprocessing_complexity': result['complexity_indicators']['preprocessing_complexity'],
        }
        data.append(row)

    df = pd.DataFrame(data)
    df_sorted = df.sort_values('processing_time')

    # 1. Complexity Ranking Chart
    st.subheader("🏆 Time Complexity Ranking")

    fig_ranking = px.bar(
        df_sorted,
        x='processing_time',
        y='filename',
        color='processing_time',
        color_continuous_scale='RdYlGn_r',
        orientation='h',
        title='Processing Time by Image (Red = High Complexity, Green = Low Complexity)',
        labels={'processing_time': 'Processing Time (seconds)', 'filename': 'Image File'}
    )
    fig_ranking.update_layout(height=max(400, len(df) * 25))
    st.plotly_chart(fig_ranking, use_container_width=True)

    # 2. Scatter Plot Analysis
    st.subheader("🔗 Processing Time vs Image Size")
    fig_scatter = px.scatter(
        df,
        x='megapixels',
        y='processing_time',
        color='strategies_used',
        size='codes_found',
        hover_data=['filename', 'preprocessing_complexity'],
        title='Processing Time vs Image Size',
        labels={'megapixels': 'Image Size (Megapixels)', 'processing_time': 'Processing Time (seconds)'},
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Detection Success Analysis
    st.subheader("🎯 Detection Success Analysis")
    success_data = {
        'Category': ['Found Codes', 'No Codes Found'],
        'Count': [len(df[df['codes_found'] > 0]), len(df[df['codes_found'] == 0])]
    }
    fig_pie = px.pie(
        success_data,
        values='Count',
        names='Category',
        title='Code Detection Success Rate'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Detailed Results Table
    st.subheader("📋 Detailed Results Table")

    # Add ranking column
    df_display = df_sorted.copy()
    df_display.insert(0, 'complexity_rank', range(1, len(df_display) + 1))

    # Format columns for better display
    df_display['processing_time'] = df_display['processing_time'].round(3)
    df_display['file_size_mb'] = df_display['file_size_mb'].round(2)
    df_display['megapixels'] = df_display['megapixels'].round(1)

    st.dataframe(df_display, use_container_width=True)

    # 5. Performance Rankings
    st.subheader("🏆 Performance Rankings")

    # Use tabs instead of columns to avoid nesting
    fastest_tab, slowest_tab = st.tabs(["🚀 Fastest Images", "🐌 Slowest Images"])

    with fastest_tab:
        st.write("**Top 5 Fastest Processing Times (Low Complexity):**")
        top_5 = df_sorted.head(5)[['filename', 'processing_time', 'megapixels', 'codes_found', 'strategies_used']]
        st.dataframe(top_5, use_container_width=True)

    with slowest_tab:
        st.write("**Top 5 Slowest Processing Times (High Complexity):**")
        bottom_5 = df_sorted.tail(5)[['filename', 'processing_time', 'megapixels', 'codes_found', 'strategies_used']]
        st.dataframe(bottom_5, use_container_width=True)

    # 6. Download Options
    st.subheader("📥 Download Results")

    # Prepare download data
    csv_data = df_sorted.to_csv(index=False)
    json_data = json.dumps(results, indent=2, default=str)

    # Download buttons
    st.download_button(
        label="📊 Download CSV Report",
        data=csv_data,
        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.download_button(
        label="📋 Download JSON Report",
        data=json_data,
        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    # 7. Performance Insights
    st.subheader("💡 Performance Insights")

    # Calculate insights
    perf = results.get('performance_analysis', {})
    if perf:
        insights = []

        # Time spread analysis
        time_ratio = perf['timing_stats']['slowest_time'] / perf['timing_stats']['fastest_time']
        if time_ratio > 10:
            insights.append(
                f"⚠️ **High complexity variation:** Slowest image took {time_ratio:.1f}x longer than fastest")

        # Correlation insights
        corr = perf.get('correlations', {})
        if corr.get('time_vs_megapixels', 0) > 0.7:
            insights.append("📏 **Strong correlation:** Processing time increases significantly with image size")
        elif corr.get('time_vs_megapixels', 0) < 0.3:
            insights.append("🎯 **Weak size correlation:** Image size is not the main complexity factor")

        if corr.get('time_vs_strategies', 0) > 0.5:
            insights.append("🔄 **Strategy dependency:** More detection strategies needed for difficult images")

        # Success rate insights
        success_rate = perf['efficiency_metrics']['detection_success_rate']
        if success_rate < 70:
            insights.append(f"📉 **Low detection rate:** Only {success_rate:.1f}% of images had detectable codes")
        elif success_rate > 90:
            insights.append(f"✅ **High detection rate:** {success_rate:.1f}% of images had detectable codes")

        # Display insights
        for insight in insights:
            st.markdown(insight)

        if not insights:
            st.info("📈 Performance appears consistent across all processed images")


if __name__ == "__main__":
    main()